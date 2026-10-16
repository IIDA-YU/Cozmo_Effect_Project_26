"""停止画像の連続性、近距離bbox、自己位置の整合からキューブ位置を確定する。

全体モデルは同じ停止位置で確定可能。距離が不確かな場合は再観測する。
旧マーカー専用の接近量補正はmarkerモードだけに残す。
"""
import math
import statistics

from research.cube_mapping import classify_cube
from research.world import wrap


def _unit(angle):
    return math.cos(angle), math.sin(angle)


def _triangulate_rays(samples, minimum_angle_degrees):
    """複数の2D視線の最小二乗交点を返す。条件が悪ければNone。"""
    if len(samples) < 2:
        return None

    minimum_angle = math.radians(minimum_angle_degrees)
    separated = False
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if abs(wrap(samples[i]['bearing'] - samples[j]['bearing'])) >= minimum_angle:
                separated = True
                break
        if separated:
            break
    if not separated:
        return None

    # A = Σ(I - uu^T), b = Σ(I - uu^T)o
    a00 = a01 = a11 = b0 = b1 = 0.0
    for sample in samples:
        ox, oy = sample['origin']
        ux, uy = _unit(sample['bearing'])
        m00 = 1.0 - ux * ux
        m01 = -ux * uy
        m11 = 1.0 - uy * uy
        a00 += m00
        a01 += m01
        a11 += m11
        b0 += m00 * ox + m01 * oy
        b1 += m01 * ox + m11 * oy

    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-6:
        return None
    x = (b0 * a11 - b1 * a01) / det
    y = (a00 * b1 - a01 * b0) / det

    residuals = []
    depths = []
    for sample in samples:
        ox, oy = sample['origin']
        ux, uy = _unit(sample['bearing'])
        dx, dy = x - ox, y - oy
        depth = dx * ux + dy * uy
        # 視線の後方に交点がある解は物理的に使わない。
        if depth <= 20:
            return None
        residuals.append(abs(dx * uy - dy * ux))
        depths.append(depth)

    residual = statistics.median(residuals)
    if residual > 55.0:
        return None
    return (x, y), max(8.0, min(35.0, residual + 8.0)), statistics.median(depths)


class CubeReviews:
    """候補の追跡・複数視点測距・確定台帳とのID対応。"""

    def __init__(self, config):
        self.config = config
        self.items = []
        self.sequence = 0
        self.accepted_items = []
        self.now = 0.0

    def _add_geometry_sample(self, item, obs):
        """同じ停止地点の連写を1視点として扱い、幾何情報を蓄積する。"""
        origin = obs.get('camera_origin')
        bearing = obs.get('bearing_world')
        distance = obs.get('distance')
        if origin is None or bearing is None or distance is None:
            return
        if not obs.get('pose_verified', True):
            return
        if obs.get('confidence', 0) < self.config.cube_candidate_confidence:
            return

        origin = (float(origin[0]), float(origin[1]))
        bearing = float(bearing)
        distance = float(distance)
        baseline = self.config.cube_viewpoint_baseline_mm

        # viewpointは確定条件用。50mm以上離れて初めて別視点と数える。
        if not item['viewpoints'] or all(math.dist(origin, p) >= baseline for p in item['viewpoints']):
            item['viewpoints'].append(origin)
            item['viewpoints'] = item['viewpoints'][-6:]

        # 三角測量/走行補正には少し細かい移動も使うが、同じ停止位置の連写は除外。
        if item['ray_samples'] and math.dist(origin, item['ray_samples'][-1]['origin']) < 18.0:
            return

        sample = dict(origin=origin, bearing=bearing, distance=distance,
                      travelled_mm=obs.get('travelled_mm'))
        if item.get('approach_anchor') is None:
            item['approach_anchor'] = dict(sample)

        # 正面に近い接近では、走行距離と見かけ距離の差から倍率を求める。
        if item['range_anchors']:
            previous = item['range_anchors'][-1]
            angle_difference = abs(wrap(bearing - previous['bearing']))
            dx = origin[0] - previous['origin'][0]
            dy = origin[1] - previous['origin'][1]
            ux, uy = _unit(bearing)
            approach = dx * ux + dy * uy
            lateral = abs(dx * uy - dy * ux)
            visual_drop = previous['distance'] - distance
            if (
                angle_difference <= math.radians(14)
                and approach >= 25.0
                and visual_drop >= 12.0
                and lateral <= max(30.0, approach * 0.70)
            ):
                correction = approach / visual_drop
                if 0.25 <= correction <= 2.50:
                    item['range_scale_samples'].append(correction)
                    item['range_scale_samples'] = item['range_scale_samples'][-5:]

        item['range_anchors'].append(sample)
        item['range_anchors'] = item['range_anchors'][-5:]
        item['ray_samples'].append(sample)
        item['ray_samples'] = item['ray_samples'][-6:]

    def _refine_observation(self, item, obs):
        """三角測量または走行量補正で現在観測の位置を更新する。"""
        self._add_geometry_sample(item, obs)
        cfg = self.config
        whole_cube = cfg.cube_detection_region == 'whole_cube'
        if whole_cube and obs.get('range_reliable', False):
            return

        # 遠距離bboxの絶対距離は信用せず、接近後はCozmoのオドメトリ移動量を
        # 主情報として初期視線上の距離を再構成する。近距離のbbox距離は
        # 「残り距離」の補助にだけ使うため、遠方で30cm級の誤差があっても
        # 地図上へそのまま投影しない。
        anchor = item.get('approach_anchor')
        origin = obs.get('camera_origin')
        bearing = obs.get('bearing_world')
        box_ratio = obs.get('box_height_ratio', 0.0)
        if not whole_cube and anchor and origin is not None and bearing is not None:
            ux, uy = _unit(anchor['bearing'])
            dx, dy = origin[0] - anchor['origin'][0], origin[1] - anchor['origin'][1]
            forward_progress = dx * ux + dy * uy
            lateral = abs(dx * uy - dy * ux)
            if (box_ratio >= cfg.cube_approach_stop_box_height_ratio
                    and forward_progress >= cfg.cube_odometry_min_approach_mm
                    and lateral <= max(45.0, forward_progress * 0.55)):
                correction = (statistics.median(item['range_scale_samples'])
                              if item['range_scale_samples'] else 1.0)
                visual_remaining = float(obs.get('distance') or cfg.cube_approach_stop_mm) * correction
                # 近接していることは画像占有率で分かるため、異常に大きなbbox距離は上限を掛ける。
                remaining = max(cfg.cube_map_near_mm + 8.0,
                                min(visual_remaining, cfg.cube_approach_stop_mm))
                anchor_range = forward_progress + remaining
                point = (anchor['origin'][0] + anchor_range * ux,
                         anchor['origin'][1] + anchor_range * uy)
                current_range = math.dist(origin, point)
                if current_range <= cfg.cube_confirm_far_mm:
                    obs['point'] = point
                    obs['sigma_mm'] = max(10.0, min(28.0, current_range * 0.10))
                    obs['refinement'] = 'ODOMETRY APPROACH'
                    obs['refined_range_mm'] = current_range
                    obs['status'] = 'READY'
                    obs['range_reliable'] = True
                    return

        triangulated = _triangulate_rays(
            item['ray_samples'], cfg.cube_triangulation_min_angle_degrees
        )
        if triangulated is not None and obs.get('refinement') != 'ODOMETRY APPROACH':
            point, sigma, _ = triangulated
            origin = obs.get('camera_origin')
            if origin is not None:
                current_range = math.dist(origin, point)
                if current_range <= cfg.cube_map_far_mm:
                    obs['point'] = point
                    obs['sigma_mm'] = sigma
                    obs['refinement'] = 'TRIANGULATED'
                    obs['refined_range_mm'] = current_range
                    if cfg.cube_map_near_mm <= current_range <= cfg.cube_confirm_far_mm:
                        obs['status'] = 'READY'
                        obs['range_reliable'] = True
                    return

        if not whole_cube and item['range_scale_samples'] and obs.get('camera_origin') is not None and obs.get('bearing_world') is not None:
            correction = statistics.median(item['range_scale_samples'])
            corrected_range = obs['distance'] * correction
            if cfg.cube_map_near_mm <= corrected_range <= cfg.cube_map_far_mm:
                ox, oy = obs['camera_origin']
                ux, uy = _unit(obs['bearing_world'])
                obs['point'] = (ox + corrected_range * ux, oy + corrected_range * uy)
                obs['sigma_mm'] = max(12.0, corrected_range * 0.08)
                obs['refinement'] = 'ODOMETRY RANGE'
                obs['refined_range_mm'] = corrected_range
                if corrected_range <= cfg.cube_confirm_far_mm:
                    obs['status'] = 'READY'
                    obs['range_reliable'] = True

    def update(self, observations, grid, now):
        self.now = now
        cfg = self.config
        self.items = [
            v for v in self.items
            if v['confirmed'] or now - v['last_seen'] <= cfg.cube_candidate_ttl_seconds
        ]
        self.accepted_items = []
        accepted, used = [], set()
        prior = list(self.items)

        for obs in sorted(observations, key=lambda o: -o.get('confidence', 0)):
            point = obs.get('point')
            if point is None:
                obs['review_status'] = classify_cube(obs, grid)
                continue

            # 同一フレームで非常に近い2枠が残った場合の最後の保険。
            if any(
                v['id'] in used and math.dist(point, v['point']) < cfg.cube_width_mm * 0.45
                for v in self.items
            ):
                obs['review_status'] = 'DUPLICATE FACE'
                continue

            sigma = obs.get('sigma_mm', 20.0)
            matches = []
            for item in prior:
                gate = min(
                    cfg.cube_track_max_match_mm,
                    max(cfg.cube_candidate_match_mm, 1.5 * (sigma + item.get('sigma_mm', 20.0))),
                )
                d = math.dist(point, item['point'])
                score = None
                if d <= gate:
                    score = d
                else:
                    # 遠距離では投影距離が大きくずれても、同じ停止位置付近から
                    # 同じ方角を見ている候補は同一キューブとして追跡する。
                    bearing = obs.get('bearing_world')
                    origin = obs.get('camera_origin')
                    previous_bearing = item.get('last_bearing_world')
                    previous_origin = item.get('last_camera_origin')
                    if (bearing is not None and origin is not None
                            and previous_bearing is not None and previous_origin is not None
                            and abs(wrap(bearing - previous_bearing)) <= math.radians(cfg.cube_bearing_match_degrees)
                            and math.dist(origin, previous_origin) <= cfg.cube_bearing_match_origin_mm):
                        score = gate * 0.75 + abs(wrap(bearing - previous_bearing)) * 20.0
                if item['id'] not in used and score is not None:
                    matches.append((score, item['id'], item))
            matches.sort(key=lambda v: v[:2])

            if len(matches) > 1 and matches[1][0] - matches[0][0] < cfg.cube_track_ambiguity_mm:
                obs['review_status'] = 'AMBIGUOUS CUBE'
                continue

            item = matches[0][2] if matches else None
            if item is None:
                self.sequence += 1
                item = dict(
                    id=self.sequence,
                    point=list(point),
                    hits=0,
                    ready_hits=0,
                    retry_after=0.0,
                    attempts=0,
                    confirmed=False,
                    samples=[],
                    last_seen=now,
                    record_id=None,
                    viewpoints=[],
                    ray_samples=[],
                    range_anchors=[],
                    range_scale_samples=[],
                    approach_anchor=None,
                    last_bearing_world=None,
                    last_camera_origin=None,
                    last_box_height_ratio=0.0,
                    last_range_reliable=False,
                    last_distance=None,
                    last_travelled_mm=0.0,
                )
                self.items.append(item)

            used.add(item['id'])
            self._refine_observation(item, obs)
            point = obs.get('point')
            status = classify_cube(obs, grid)
            obs['review_status'] = status
            sigma = obs.get('sigma_mm', sigma)

            stable = math.dist(point, item['point']) <= cfg.cube_merge_mm
            eligible = (
                status == 'MAPPED'
                and obs.get('range_reliable', obs.get('status') == 'READY')
                and obs.get('pose_verified', True)
                and obs.get('confidence', 0) >= cfg.cube_review_confirm_confidence
            )

            if not stable or now - item['last_seen'] > 4.0:
                item['ready_hits'], item['samples'] = 0, []

            item['ready_hits'] = item['ready_hits'] + 1 if eligible else 0
            item.update(
                visual_confirmed=obs.get('visual_confirmed', item.get('visual_confirmed', True)),
                pose_verified=obs.get('pose_verified', True),
                confidence=obs.get('confidence', 0),
                last_seen=now,
                status=status,
                hits=item['hits'] + 1,
                sigma_mm=sigma,
                refinement=obs.get('refinement', 'VISUAL RANGE'),
                last_bearing_world=obs.get('bearing_world', item.get('last_bearing_world')),
                last_camera_origin=obs.get('camera_origin', item.get('last_camera_origin')),
                last_box_height_ratio=obs.get('box_height_ratio', item.get('last_box_height_ratio', 0.0)),
                last_range_reliable=obs.get('range_reliable', False),
                last_distance=obs.get('refined_range_mm', obs.get('distance')),
                last_travelled_mm=obs.get('travelled_mm', item.get('last_travelled_mm', 0.0)),
            )
            if eligible:
                item['samples'].append((float(point[0]), float(point[1]), max(5.0, sigma)))
                item['samples'] = item['samples'][-7:]

            item['point'] = list(point)
            required_frames = max(cfg.cube_confirm_frames, cfg.cube_review_confirm_frames)
            geometry_capable = bool(item['ray_samples']) or (
                obs.get('camera_origin') is not None and obs.get('bearing_world') is not None
            )
            # 独自テスト検出器など幾何情報を返さない旧インターフェースでは
            # 従来互換にする。実機ResearchPerceptionは必ず幾何情報を返す。
            viewpoints_ok = (
                len(item['viewpoints']) >= cfg.cube_confirm_viewpoints
                if geometry_capable else True
            )
            confirmed_now = (eligible and item['ready_hits'] >= required_frames
                             and viewpoints_ok and item['visual_confirmed'])

            if confirmed_now:
                weights = [1 / s[2] ** 2 for s in item['samples']]
                item['estimate'] = [
                    round(sum(s[i] * w for s, w in zip(item['samples'], weights)) / sum(weights), 6)
                    for i in (0, 1)
                ]
                item['confirmed'] = True
                self.accepted_items.append(item)
                accepted.append(tuple(item['estimate']))

            obs['track_id'] = item['id']
            if confirmed_now:
                obs['review_status'] = 'MAPPED'
            elif not obs.get('pose_verified', True):
                obs['review_status'] = 'POSE UNVERIFIED'
            elif status == 'MAPPED' and not viewpoints_ok:
                obs['review_status'] = f'VIEW {len(item["viewpoints"])}/{cfg.cube_confirm_viewpoints}'
            elif status == 'MAPPED':
                obs['review_status'] = 'CHECKING'
            else:
                obs['review_status'] = status
            item['status'] = obs['review_status']

        for item in self.items:
            if item['id'] not in used:
                item['ready_hits'], item['samples'] = 0, []
        return accepted

    def reset_after_pose_fix(self, before, after):
        """座標系補正後は、古い視線幾何を混ぜずに再観測する。"""
        angle = wrap(after[2] - before[2])
        c, s = math.cos(angle), math.sin(angle)
        for item in self.items:
            x, y = item['point'][0] - before[0], item['point'][1] - before[1]
            item['point'] = [after[0] + c * x - s * y, after[1] + s * x + c * y]
            item['ready_hits'], item['samples'] = 0, []
            item['viewpoints'] = []
            item['ray_samples'] = []
            item['range_anchors'] = []
            item['range_scale_samples'] = []
            item['approach_anchor'] = None

    def unresolved(self, world):
        known = world.cubes()
        return [
            item for item in self.items
            if not item['confirmed']
            and self.now - item['last_seen'] <= self.config.cube_candidate_ttl_seconds
            and not any(
                math.dist(item['point'], (c['x'], c['y'])) <= self.config.cube_merge_mm
                for c in known
            )
        ]

    def discard_lost_candidates(self, observations):
        """停止画像で繰り返し見失った未確認物だけを削除する。確定物は保持。"""
        lost_ids = {o.get('track_id') for o in observations}
        self.items = [item for item in self.items
                      if item['id'] not in lost_ids or item['confirmed']
                      or item.get('visual_confirmed', False)]

    def previously_confirmed(self, observation):
        """確定物の一意な再観測なら、弱い枠だけで未確認へ戻さない。"""
        point = observation.get('point')
        if point is None or not observation.get('range_reliable', False):
            return False
        ranked = sorted((math.dist(point, item['point']), item['id'], item) for item in self.items)
        if not ranked or ranked[0][0] > self.config.cube_merge_mm:
            return False
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < self.config.cube_track_ambiguity_mm:
            return False
        item = ranked[0][2]
        return item['confirmed'] and item.get('record_id') is not None


class CubeVerifier:
    """キューブ候補を最優先で追跡し、短い接近と停止観測を繰り返す。

    遠距離のbbox距離ではなく、まず検出方向を信頼して近づく。見失っても
    数秒間・一定走行距離までは最後の方向を疑い続ける。十分近づいた後は
    odometry + 近距離観測、必要なら別視点の三角測量で位置を確定する。
    """

    def __init__(self, reviews):
        self.reviews = reviews
        self.active_id = None
        self.since = 0.0
        self.looks = 0
        self.visited = {}

    def reset_route(self, planner):
        if planner.mode == 'verify':
            planner.path, planner.goal, planner.heading = [], None, None
            planner.mode = 'coverage'
        planner.active_verify_point = None
        self.active_id = None
        self.looks = 0

    @staticmethod
    def _heading_to_item(pose, item):
        bearing = item.get('last_bearing_world')
        if bearing is not None:
            return float(bearing)
        return math.atan2(item['point'][1] - pose[1], item['point'][0] - pose[0])

    def _side_viewpoint(self, pose, planner, item, now):
        """近接後に別角度から確認するための安全な観測点を作る。"""
        cfg, world = planner.config, planner.world
        grid = world.grid
        previous, start = grid.reachable(pose[:2], planner.obstacles())
        if start is None:
            return None
        visited = self.visited.setdefault(item['id'], set())
        visited.update(k for k in previous if math.dist(pose[:2], grid.cells[k]) < 35)
        desired = max(cfg.cube_review_distance_mm, planner._cube_safe_radius() + 30)
        candidates = [
            k for k in previous
            if k not in visited and math.dist(grid.cells[k], item['point']) >= desired
        ]
        if not candidates:
            item['attempts'] += 1
            item['retry_after'] = now + cfg.cube_review_retry_seconds
            visited.clear()
            self.reset_route(planner)
            return None

        current_view = self._heading_to_item(pose, item)
        preferred = math.radians(cfg.cube_viewpoint_preferred_angle_degrees)

        def score(key):
            position = grid.cells[key]
            target_distance = math.dist(position, item['point'])
            route_cost = 0.15 * len(grid.route(previous, key)) * grid.cell
            candidate_view = math.atan2(item['point'][1] - position[1], item['point'][0] - position[0])
            separation = abs(wrap(candidate_view - current_view))
            angle_penalty = max(0.0, preferred - separation) * 140.0
            return abs(target_distance - desired) + route_cost + angle_penalty, key

        key = min(candidates, key=score)
        planner.path = planner._smooth_path(
            pose[:2], [grid.cells[k] for k in grid.route(previous, key)]
        )
        planner.goal = grid.cells[key]
        planner.heading = math.atan2(
            item['point'][1] - planner.goal[1], item['point'][0] - planner.goal[0]
        )
        visited.add(key)
        return planner.command(pose)

    def command(self, pose, planner, now, travelled_mm=0.0):
        cfg, world = planner.config, planner.world
        pending = self.reviews.unresolved(world)
        # 通常はsessionの停止ループが解決する。別呼出し経路でも未確認のまま動かさない。
        if any(not v.get('visual_confirmed', True) for v in pending):
            planner.motion_target = planner.motion_heading = None
            return 0, 0
        # 存在確認済みでも自己位置が未検証なら、チェス再観測を先に行う。
        pending = [v for v in pending if v.get('pose_verified', True)]
        item = next((v for v in pending if v['id'] == self.active_id), None)
        if item is None:
            self.reset_route(planner)
            candidates = [v for v in pending if now >= v['retry_after']]
            if not candidates:
                return None
            # 最近見えた候補を最優先。その次に試行回数と距離。
            item = min(candidates, key=lambda v: (
                -(v.get('last_seen', 0.0)), v['attempts'], math.dist(pose[:2], v['point'])
            ))
            self.active_id, self.since, self.looks = item['id'], now, 0
            planner.cancel_landmark_route()
            planner.path, planner.goal, planner.heading = [], None, None
            planner.mode = 'verify'

        planner.active_verify_point = tuple(item['point'])

        if now - self.since >= cfg.cube_review_timeout_seconds:
            item['attempts'] += 1
            item['retry_after'] = now + cfg.cube_review_retry_seconds
            self.reset_route(planner)
            return None

        if planner.path:
            return planner.command(pose)

        age = max(0.0, now - item.get('last_seen', now))
        heading = self._heading_to_item(pose, item)
        delta = wrap(heading - pose[2])
        if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
            return planner._turn_command(delta, pose[2])

        visually_close = item.get('last_box_height_ratio', 0.0) >= cfg.cube_approach_stop_box_height_ratio
        range_close = (item.get('last_range_reliable', False)
                       and item.get('last_distance') is not None
                       and item['last_distance'] <= cfg.cube_approach_stop_mm)

        # 推定点が既に衝突安全半径の内側なら、距離が怪しくても前進しない。
        estimated_too_close = (
            item.get('point') is not None
            and math.dist(pose[:2], item['point']) <= planner._cube_safe_radius() + 10.0
        )
        if estimated_too_close and not (visually_close or range_close):
            item['attempts'] += 1
            item['retry_after'] = now + cfg.cube_review_retry_seconds
            self.reset_route(planner)
            return None

        # まだ遠い/距離不明なら、最後に見えた方向へ短く前進して必ず再停止する。
        # 見失っても猶予時間内かつ最後に見えた地点から75mm以内なら疑い続ける。
        moved_since_seen = max(0.0, travelled_mm - float(item.get('last_travelled_mm', travelled_mm) or 0.0))
        can_chase_lost = age <= cfg.cube_lost_grace_seconds and moved_since_seen <= cfg.cube_lost_forward_mm
        if not visually_close and not range_close and (age < 1.2 or can_chase_lost):
            step = cfg.cube_approach_step_mm
            # active_verify_point は planner.obstacles で一時候補から除外される。
            if planner.can_advance(pose, step):
                planner.motion_heading = heading
                planner.motion_target = (
                    pose[0] + step * math.cos(heading),
                    pose[1] + step * math.sin(heading),
                )
                speed = min(cfg.drive_speed_mm_s, 35.0)
                return speed, speed

        # 十分近い場合は2回停止観測してから横へ回り、2視点目を作る。
        if visually_close or range_close:
            self.looks += 1
            if self.looks < 2:
                return 0, 0
            self.looks = 0
            return self._side_viewpoint(pose, planner, item, now)

        # 見失い猶予を使い切ったら、その場で回り続けず別視点へ移る。
        return self._side_viewpoint(pose, planner, item, now)
