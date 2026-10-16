"""既知チェス駒との幾何照合。自己位置と残差を検証し、独立した観測で駒を補正する。

駒の番号は画像の固有IDではない。一対一対応が曖昧なら補正しない。
2点では駒の移動と機体のずれを完全には分離できないため、間隔も検査する。
"""
import itertools
import math
from research.world import wrap
from research.landmark_refinement import LandmarkRefiner


def transform(point, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + c * point[0] - s * point[1],
            pose[1] + s * point[0] + c * point[1])


def fit_pose(local, world):
    """縮尺を変えない2次元の最小二乗剛体変換。"""
    a = [sum(p[i] for p in local) / len(local) for i in (0, 1)]
    b = [sum(p[i] for p in world) / len(world) for i in (0, 1)]
    dot = cross = 0.0
    for p, q in zip(local, world):
        px, py, qx, qy = p[0]-a[0], p[1]-a[1], q[0]-b[0], q[1]-b[1]
        dot += px*qx + py*qy
        cross += px*qy - py*qx
    angle = math.atan2(cross, dot)
    rotated = transform(a, (0, 0, angle))
    return b[0]-rotated[0], b[1]-rotated[1], angle


class LandmarkLocalizer:
    def __init__(self, landmarks, config):
        self.landmarks = {str(k): tuple(v) for k, v in landmarks.items()}
        self.config = config
        self.previous = None
        self.streak = 0
        self.last_fix_time = None
        self.last_fix_distance = 0.0
        self.last_fix_ids = set()
        self.report = dict(status='UNVERIFIED', matches=[], pairs=[], residual_mm=None, seen_ids=[], seen_count=0)
        self.suspect_counts = {}
        # 6駒を周回して確認するため、フレームをまたいで認識済みIDを保持する。
        self.seen_counts = {}
        self.seen_ids = set()
        self.refiner = LandmarkRefiner(self.landmarks, config)
        self.pose_references = []
        self.last_observation_time = None
        self.refinement_pending = False

    def trusted(self, now, travelled):
        return (self.last_fix_time is not None
                and now - self.last_fix_time <= self.config.localization_max_age_seconds
                and travelled - self.last_fix_distance <= self.config.localization_max_travel_mm)

    def _matches(self, observations, pose, limit=None):
        """最近傍を一対一照合する。同じ駒への競合/同距離なら採用しない。"""
        cfg = self.config
        limit = cfg.chess_match_mm if limit is None else limit
        proposals = []
        for index, obs in enumerate(observations):
            point = transform((obs['local_x'], obs['local_y']), pose)
            choices = sorted((math.dist(point, p), k) for k, p in self.landmarks.items())
            if not choices or choices[0][0] > limit:
                continue
            if len(choices) > 1 and choices[1][0] - choices[0][0] < cfg.chess_ambiguity_mm:
                continue
            proposals.append((choices[0][0], index, choices[0][1]))
        matches = []
        for error, index, key in sorted(proposals):
            competing = [e for e, i, k in proposals if k == key and i != index]
            if competing and min(competing) - error < cfg.chess_ambiguity_mm:
                continue
            obs = observations[index]
            matches.append(dict(id=key, index=index, local=(obs['local_x'], obs['local_y']),
                                map=self.landmarks[key], error_before_mm=error))
        return matches

    def _update_pose(self, observations, pose, now, travelled):
        cfg = self.config
        matches = self._matches(observations, pose)

        # 自己位置補正は2駒以上必要だが、「その駒を見た」事実は1駒でも蓄積する。
        # 1フレームの誤検知で完了扱いしないよう、同じIDが複数フレームで
        # 保存地図の近傍に対応した場合だけ seen_ids へ昇格させる。
        matched_now = set()
        for match in matches:
            if match['error_before_mm'] <= cfg.chess_seen_match_mm:
                key = match['id']
                matched_now.add(key)
                self.seen_counts[key] = self.seen_counts.get(key, 0) + 1
                if self.seen_counts[key] >= cfg.chess_seen_confirm_frames:
                    self.seen_ids.add(key)
        # 連続性を要求する。今フレームで見えていない未確定IDはカウントを戻す。
        for key in list(self.seen_counts):
            if key not in matched_now and key not in self.seen_ids:
                self.seen_counts[key] = 0

        pairs = []
        for a, b in itertools.combinations(matches, 2):
            measured, mapped = math.dist(a['local'], b['local']), math.dist(a['map'], b['map'])
            pairs.append(dict(ids=[a['id'], b['id']], observed_mm=measured,
                              map_mm=mapped, error_mm=measured-mapped))
        report = dict(status='NEED 2 CHESS', matches=matches, pairs=pairs,
                      residual_mm=None, corrected=False, suspect_ids=[],
                      seen_ids=sorted(self.seen_ids), seen_count=len(self.seen_ids))
        candidates = {}
        for a, b in itertools.combinations(range(len(matches)), 2):
            selected = [matches[a], matches[b]]
            if min(math.dist(selected[0]['local'], selected[1]['local']),
                   math.dist(selected[0]['map'], selected[1]['map'])) < cfg.chess_min_baseline_mm:
                continue
            if abs(math.dist(selected[0]['local'], selected[1]['local']) -
                   math.dist(selected[0]['map'], selected[1]['map'])) > cfg.chess_pair_tolerance_mm:
                continue
            estimate = fit_pose([m['local'] for m in selected], [m['map'] for m in selected])
            inliers = tuple(i for i, m in enumerate(matches)
                            if math.dist(transform(m['local'], estimate), m['map']) <= cfg.chess_residual_mm)
            required = 2 if len(matches) == 2 else 3
            if len(inliers) < required:
                continue
            chosen = [matches[i] for i in inliers]
            estimate = fit_pose([m['local'] for m in chosen], [m['map'] for m in chosen])
            errors = [math.dist(transform(m['local'], estimate), m['map']) for m in chosen]
            residual = math.sqrt(sum(e*e for e in errors) / len(errors))
            if (max(errors) > cfg.chess_residual_mm
                    or math.dist(estimate[:2], pose[:2]) > cfg.chess_max_shift_mm
                    or abs(wrap(estimate[2]-pose[2])) > math.radians(cfg.chess_max_turn_degrees)):
                continue
            candidates[inliers] = (residual, estimate)
        if not candidates:
            report['status'] = 'GEOMETRY MISMATCH' if len(matches) >= 2 else 'NEED 2 CHESS'
            self.previous, self.streak = None, 0
            if len(matches) >= 2:
                self.last_fix_time = None
            self.report = report
            return pose, report
        ranked = sorted(candidates, key=lambda k: (-len(k), candidates[k][0], k))
        best = ranked[0]
        if len(ranked) > 1 and len(ranked[1]) == len(best):
            report['status'] = 'AMBIGUOUS GEOMETRY'
            self.previous, self.streak = None, 0
            if len(matches) >= 2:
                self.last_fix_time = None
            self.report = report
            return pose, report
        residual, estimate = candidates[best]
        signature = tuple(sorted(matches[i]['id'] for i in best))
        stable = (self.previous is not None and self.previous[0] == signature
                  and now - self.previous[2] < 4.0
                  and math.dist(estimate[:2], self.previous[1][:2]) <= 12
                  and abs(wrap(estimate[2] - self.previous[1][2])) <= math.radians(4))
        self.streak = self.streak + 1 if stable else 1
        self.previous = signature, estimate, now
        report.update(status='CHECKING CHESS', residual_mm=residual, inlier_ids=list(signature))
        for i, m in enumerate(matches):
            m['residual_mm'] = math.dist(transform(m['local'], estimate), m['map'])
            bad = i not in best and m['residual_mm'] > cfg.chess_residual_mm
            self.suspect_counts[m['id']] = self.suspect_counts.get(m['id'], 0) + 1 if bad else 0
            if self.suspect_counts[m['id']] >= 3:
                report['suspect_ids'].append(m['id'])
        if self.streak >= cfg.chess_confirm_frames:
            report.update(status='POSE VERIFIED', corrected=True)
            self.last_fix_time, self.last_fix_distance = now, travelled
            self.last_fix_ids = set(signature)
            pose = estimate
        self.report = report
        return pose, report

    @staticmethod
    def _propagate(reference, odometry):
        """保存した独立姿勢へ、生のタイヤオドメトリの相対変位だけを加える。"""
        old_raw, fixed = reference['raw'], reference['pose']
        dx, dy = odometry[0] - old_raw[0], odometry[1] - old_raw[1]
        c, sn = math.cos(old_raw[2]), math.sin(old_raw[2])
        point = transform((c * dx + sn * dy, -sn * dx + c * dy), fixed)
        return (*point, wrap(fixed[2] + odometry[2] - old_raw[2]))

    def _independent_pose(self, key, report, odometry, now, travelled):
        inliers = set(report.get('inlier_ids', []))
        # 3駒以上で自己位置が確定した場合、対象を必ず外して再計算する。
        # 2駒だけの幾何では、どちらがずれたか判別できない。
        if report['corrected'] and len(inliers) >= 3:
            anchors = [m for m in report['matches'] if m['id'] in inliers and m['id'] != key]
            if max(math.dist(a['map'], b['map']) for a, b in itertools.combinations(anchors, 2)) >= self.config.chess_min_baseline_mm:
                estimate = fit_pose([m['local'] for m in anchors], [m['map'] for m in anchors])
                return estimate, {m['id'] for m in anchors}
        # 視野に1駒しか入らなくても、直近に「別の駒」で確定した姿勢なら使える。
        # 現在の補正済みposeを使うと対象自身の誤差を循環参照するため、rawで伝播する。
        for ref in reversed(self.pose_references):
            if (key not in ref['ids']
                    and 0 <= now - ref['time'] <= self.config.chess_refine_pose_age_seconds
                    and 0 <= travelled - ref['travelled'] <= self.config.chess_refine_pose_travel_mm):
                return self._propagate(ref, odometry), set(ref['ids'])
        return None, set()

    def update(self, observations, pose, now, travelled=0.0, odometry=None):
        """1画像ずつ更新する。補正は安定した標本の平均が揃った時だけ行う。"""
        if self.last_observation_time is not None and now <= self.last_observation_time:
            # 同じ画像の再投入では確認数/平均/補正を増やさない。
            duplicate = dict(self.report, corrected=False, landmark_updates=[], duplicate_frame=True)
            return pose, duplicate
        self.last_observation_time = now
        observations = [o for o in observations
                        if all(isinstance(o.get(k), (int, float)) and math.isfinite(o[k])
                               for k in ('local_x', 'local_y'))]
        odometry = pose if odometry is None else odometry
        corrected, report = self._update_pose(observations, pose, now, travelled)
        updates, states = [], []
        cfg = self.config
        if cfg.chess_refine_enabled:
            for key, current in self.landmarks.items():
                independent, anchors = self._independent_pose(key, report, odometry, now, travelled)
                if independent is None:
                    continue
                limit = max(cfg.chess_match_mm, cfg.chess_refine_max_shift_mm + cfg.chess_refine_outlier_mm)
                match = next((m for m in self._matches(observations, independent, limit) if m['id'] == key), None)
                if match is None or not 60 <= math.hypot(*match['local']) <= cfg.chess_refine_far_mm:
                    continue
                point = transform(match['local'], independent)
                proposal, state = self.refiner.offer(key, point, current, now, anchors)
                states.append(state)
                if proposal is not None:
                    # 平均位置でも他の駒とのIDの近接/入れ替わりを許さない。
                    alternatives = [math.dist(proposal['after'], p) for k, p in self.landmarks.items() if k != key]
                    if alternatives and min(alternatives) - proposal['shift_mm'] < cfg.chess_ambiguity_mm:
                        state['status'] = 'AMBIGUOUS ID'
                        continue
                    updates.append(proposal)

        proposed_ids = {u['id'] for u in updates}
        # 同時に直す駒を互いの基準にする更新は採用しない。
        dependent = {u['id'] for u in updates if proposed_ids.intersection(u['anchor_ids'])}
        if dependent:
            updates = [u for u in updates if u['id'] not in dependent]
            for state in states:
                if state['id'] in dependent:
                    state['status'] = 'NEED INDEPENDENT ANCHORS'

        changed = {u['id'] for u in updates}
        if changed:
            for update in updates:
                key = update['id']
                self.landmarks[key] = tuple(update['after'])
                self.seen_ids.add(key)
                self.seen_counts[key] = cfg.chess_seen_confirm_frames
                self.suspect_counts[key] = 0
            self.refiner.commit(updates)
            self.previous, self.streak = None, 0
            self.pose_references = [r for r in self.pose_references if not changed.intersection(r['ids'])]
            # 更新前の対象座標を含んで解いた姿勢は、同じフレームでは採用しない。
            if changed.intersection(self.last_fix_ids):
                self.last_fix_time = None
            if changed.intersection(report.get('inlier_ids', [])):
                corrected = pose
                report.update(corrected=False, status='CHESS UPDATED / RECHECK POSE')
                self.last_fix_time = None

        doubtful = {s['id'] for s in states if s['status'] not in ('STABLE', 'READY')
                    and s.get('shift_mm', 0) >= cfg.chess_refine_min_shift_mm}
        if doubtful.intersection(report.get('inlier_ids', [])):
            # 平均化待ちの駒で自己位置だけ先に動かし、キューブを誤投影しない。
            corrected = pose
            report.update(corrected=False, status='CHECKING CHESS POSITION')
            self.last_fix_time = None

        if report['corrected']:
            ids = tuple(report['inlier_ids'])
            groups = [tuple(key for key in ids if key != omitted) for omitted in ids] if len(ids) >= 3 else []
            groups.append(ids)
            for group in groups:
                anchors = [m for m in report['matches'] if m['id'] in group]
                if max(math.dist(a['map'], b['map']) for a, b in itertools.combinations(anchors, 2)) < cfg.chess_min_baseline_mm:
                    continue
                fixed = fit_pose([m['local'] for m in anchors], [m['map'] for m in anchors])
                self.pose_references = [r for r in self.pose_references if r['ids'] != group]
                self.pose_references.append(dict(ids=group, raw=tuple(odometry), pose=tuple(fixed),
                                                 time=now, travelled=travelled))
            self.pose_references = self.pose_references[-32:]
        self.refinement_pending = any(s['status'] == 'COLLECTING'
                                      and s.get('shift_mm', 0) >= cfg.chess_refine_min_shift_mm for s in states)
        report.update(landmark_updates=updates, refinement=states, refinement_pending=self.refinement_pending,
                      landmarks={k: list(p) for k, p in self.landmarks.items()},
                      seen_ids=sorted(self.seen_ids), seen_count=len(self.seen_ids))
        self.report = report
        return corrected, report

    def observe_frame(self, slam, frame, raw, pose, now, travelled):
        """同じ停止画像でチェスを先に処理し、キューブ投影に補正後の姿勢を渡す。"""
        detections = slam.detect_chess_pieces(frame)
        observations = []
        for detection in detections:
            if detection['confidence'] < self.config.chess_confidence:
                continue
            obs = slam._estimate_landmark_observation(detection, frame.shape[1], *pose)
            if obs is not None:
                # 既存チェス校正はCozmo基準点からの実測。offsetを二重加算しない。
                observations.append(obs)
        corrected, report = self.update(observations, pose, now, travelled, odometry=raw)
        if report['corrected']:
            slam._set_estimated_pose(*raw, *corrected)
        boxes = [dict(box=d['box'], label='Chess: ' + report['status']) for d in detections]
        if not observations and detections:
            report['status'] = 'CHESS UNCALIBRATED / CLIPPED'
        return corrected, report, boxes
