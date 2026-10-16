"""探索実験の実行。走行・観測・共有・記録を一つのループで順番に処理。"""
import datetime
import json
import math
import time
from pathlib import Path

from research.world import World,save_json
from research.exchange import EncounterExchange
from research.perception import ResearchPerception
from research.planner import Explorer
from research.interface import MapWindow
from research.cube_review import CubeReviews, CubeVerifier
from research.cube_stop import CubeStopLock
from research.motion import run_motion
from research.localization import LandmarkLocalizer
from research.cube_calibration import load_calibration, calibrate_stopped
from telecommunication.bluetooth_communication import BluetoothNetwork


def _attempt_reached_start(output):
    """過去の同一試行で、実際の探索開始まで進んだかを確認する。

    探索開始済みの記録は研究データとして残す必要があるため、
    同じ trial_id / robot_id で上書きしない。
    """
    exploration_path = output / 'exploration.json'
    if exploration_path.exists():
        try:
            data = json.loads(exploration_path.read_text(encoding='utf-8'))
            if data.get('metrics', {}).get('started_utc'):
                return True
        except (OSError, json.JSONDecodeError, TypeError):
            # exploration.json が途中までしか書かれていない場合は、
            # events.jsonl も確認して安全側に判定する。
            pass

    events_path = output / 'events.jsonl'
    if events_path.exists():
        try:
            with events_path.open('r', encoding='utf-8') as events:
                for line in events:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get('event') == 'start':
                        return True
        except OSError:
            # 読み取れない既存記録は上書きしない。
            return True

    return False


def _next_archive_path(output):
    """開始前に失敗した記録の退避先を重複しない名前で作る。"""
    archive_root = output.parent / '_failed_attempts'
    archive_root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    candidate = archive_root / f'{output.name}_{stamp}'
    number = 2
    while candidate.exists():
        candidate = archive_root / f'{output.name}_{stamp}_{number}'
        number += 1
    return candidate


def prepare_output_for_run(setup, config):
    """同じ試行IDの結果フォルダが残っている場合を安全に処理する。

    - 探索開始前に失敗したフォルダ: 自動で _failed_attempts へ退避
    - 探索開始済みのフォルダ: 上書きせず、次の試行IDを要求

    この確認は controller から Cozmo Wi-Fi へ切り替える前にも呼び出す。
    """
    output = config.setup_path.parent / 'results' / setup['trial_id'] / config.robot_id

    if not output.exists():
        return output

    if _attempt_reached_start(output):
        raise RuntimeError(
            'この試行IDでは既に探索を開始した記録が存在します。\n'
            '過去の実験データを上書きしないため、メニューの '
            '「3:次の試行ID」を選択してから、もう一度「2:探索実験」を実行してください。\n'
            f'既存の記録: {output}'
        )

    archive = _next_archive_path(output)
    output.rename(archive)
    print('前回の探索開始前エラー記録を退避しました:')
    print(f'  {archive}')
    return output


def _confirm_cube_positions(frame_candidates, minimum_frames, match_mm):
    """複数フレームのキューブ候補をまとめ、安定して見えた位置だけ返す。

    旧実装は perception.py 内で「直前フレームの位置と30mm以内」を
    3回連続で満たす必要があり、カメラ枠は出ているのに地図登録されない
    ケースがあった。ここでは各フレームの候補をクラスタ化し、
    minimum_frames 枚以上で見えたクラスタの平均座標を採用する。
    """
    minimum_frames = max(1, int(minimum_frames))
    match_mm = float(match_mm)
    clusters = []

    for frame_index, candidates in enumerate(frame_candidates):
        used_clusters = set()
        for point in candidates:
            x, y = float(point[0]), float(point[1])
            matches = []
            for index, cluster in enumerate(clusters):
                if index in used_clusters or frame_index in cluster['frames']:
                    continue
                cx = cluster['sum_x'] / cluster['count']
                cy = cluster['sum_y'] / cluster['count']
                distance = math.hypot(x - cx, y - cy)
                if distance <= match_mm:
                    matches.append((distance, index))

            if matches:
                _, index = min(matches)
                cluster = clusters[index]
                cluster['sum_x'] += x
                cluster['sum_y'] += y
                cluster['count'] += 1
                cluster['frames'].add(frame_index)
                used_clusters.add(index)
            else:
                clusters.append({
                    'sum_x': x,
                    'sum_y': y,
                    'count': 1,
                    'frames': {frame_index},
                })
                used_clusters.add(len(clusters) - 1)

    confirmed = []
    for cluster in clusters:
        if len(cluster['frames']) < minimum_frames:
            continue
        confirmed.append((
            cluster['sum_x'] / cluster['count'],
            cluster['sum_y'] / cluster['count'],
        ))
    return confirmed


class TrialInterrupted(Exception):
    """完了とは別に記録する、時間・電池による実験中断。"""


class ResearchSession:
    def __init__(self, cli, camera, slam, setup, config, battery):
        self.cli,self.camera,self.slam = cli,camera,slam
        self.setup,self.config,self.battery = setup,config,battery
        self.world = World(setup,config.robot_id)
        self.planner = Explorer(self.world,config)
        load_calibration(config)
        self.perception = ResearchPerception(config)
        # Cozmo とキューブを、実寸を地図縮尺へ合わせて表示する。
        self.window = MapWindow(
            setup['landmarks'],
            config.robot_length_mm,
            config.robot_width_mm,
            config.cube_width_mm,
        )

        # 参加機体が自分だけなら Bluetooth は完全に使用しない。
        # 2台探索のときだけ、setup に含まれる相手機体の COM ポートを開く。
        peers = set(setup['robots']) - {config.robot_id}
        ports = (
            {peer: config.bluetooth_ports[peer] for peer in peers}
            if config.sharing_mode != 'none'
            else {}
        )
        self.network = BluetoothNetwork(ports)
        self.exchange = EncounterExchange(self.world,self.network,config.sharing_mode,
                                           config.encounter_cooldown_seconds,self.log)
        self.robot_count = len(setup['robots'])
        self.active_robots = tuple(setup['robots'])
        # controller 側でも事前確認するが、直接 ResearchSession を使った場合にも
        # 安全になるよう、ここでも同じ試行IDの既存記録を確認する。
        self.output = prepare_output_for_run(setup, config)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (self.output/'events.jsonl').open('w',encoding='utf-8')
        self.started = None
        self.started_utc = None
        self.paused = False
        self.last_frame = -1
        self.previous_raw = None
        self.distance = 0.0
        self.first_all_known = None
        self.first_local_all = None
        self.last_saved = 0.0
        self.marker_ids = []
        self.interventions = 0
        self.reason = 'not_started'
        self.completion_reason = None
        self.exploration_finished_seconds = None
        self.exploration_finished_distance_mm = None
        self.exploration_coverage_ratio = None
        self.return_started = None
        self.return_started_distance = None
        self.returned_home = False
        self.return_home_seconds = None
        self.return_home_distance_mm = None
        self.last_coverage = 0
        self.last_trail_coverage = 0
        self.last_chess_seen = 0
        self.stale_progress_since = 0.0

        # 距離条件などで永続登録を保留した候補。地図上では輪郭のみ表示する。
        self.live_cube_candidates = []
        self.cube_reviews = CubeReviews(config)
        self.cube_verifier = CubeVerifier(self.cube_reviews)
        self.cube_stop_active = False
        self.cube_stop_started = None
        self.cube_stop_seconds = 0.0
        self.cube_stop_count = 0
        self.mapping_status = 'Waiting for a stopped observation'
        self.last_mapping_warning = None
        self._last_action = None
        self.localizer = (LandmarkLocalizer(setup['landmarks'], config)
                          if hasattr(slam, 'detect_chess_pieces') else None)
        self.localization_scan_since = None
        self.localization_status = 'UNVERIFIED'
        if self.localizer is not None:
            self.world.landmarks = self.localizer.landmarks
        self.last_observed_pose = None



    def _register_detected_cubes(self, candidates):
        """確定した追跡IDを登録し、再観測なら同じ台帳IDの位置を更新する。"""
        elapsed = 0.0 if self.started is None else time.monotonic() - self.started
        reviews = getattr(self, 'cube_reviews', None)
        if reviews is None:
            for x, y in candidates:
                self.world.add_cube(x, y, elapsed)
            return
        for item in reviews.accepted_items:
            x, y = item['estimate']
            old_id = item['record_id']
            old = dict(self.world.records[old_id]) if old_id else None
            key = self.world.upsert_cube(x, y, elapsed, old_id)
            if key is None:
                continue
            item['record_id'] = key
            if old is None:
                self.log('cube_discovered', cube_id=key, x=x, y=y, track_id=item['id'])
            elif self.world.records[key] != old:
                self.log('cube_position_updated', cube_id=key, x=x, y=y,
                         previous=[old['x'], old['y']], revision=self.world.records[key]['revision'])

    def _pose_is_verified(self):
        localizer = getattr(self, 'localizer', None)
        return localizer is None or (localizer.trusted(time.monotonic(), self.distance)
            and localizer.report['status'] not in ('GEOMETRY MISMATCH', 'AMBIGUOUS GEOMETRY'))

    def _recheck_localization(self):
        """位置未確定でもその場旋回へ固定せず、周回探索へ回復させる。

        v6までは45秒間20度ずつ同じ場所で回り、2駒同時照合できないと例外で
        終了していた。v7では中央の全周スキャンはplannerへ任せ、中央以外では
        未確認チェスの内側観測点へ移動する。ここでは状態を記録するだけにする。
        """
        if self._pose_is_verified():
            self.localization_scan_since = None
            return False
        now = time.monotonic()
        if self.localization_scan_since is None:
            self.localization_scan_since = now
            self.log('localization_recheck', reason=self.localizer.report.get('status', 'UNVERIFIED'))
        elif now - self.localization_scan_since > self.config.localization_scan_timeout_seconds:
            # エラー終了せず、周回しながら再取得を続ける。ログだけ周期的に残す。
            self.log('localization_roaming_recovery',
                     reason=self.localizer.report.get('status', 'UNVERIFIED'),
                     chess_seen=len(getattr(self.localizer, 'seen_ids', set())))
            self.localization_scan_since = now
        return False

    def log(self, event, **values):
        elapsed = 0.0 if self.started is None else time.monotonic()-self.started
        if event in ('exchange_received','exchange_success'):
            values['known_record_ids'] = sorted(self.world.records)
        self.events.write(json.dumps(dict(event=event,elapsed=elapsed,**values),ensure_ascii=False,allow_nan=False)+'\n')
        self.events.flush()
        if event.startswith('exchange_') or event=='cube_discovered':
            print(event,values)

    def pose(self):
        p = self.cli.pose
        if p is None:
            raise RuntimeError('機体座標を取得できません')
        raw = p.position.x,p.position.y,p.rotation.angle_z.radians
        if not all(math.isfinite(v) for v in raw):
            raise RuntimeError('機体座標が不正です')
        if self.previous_raw is not None:
            distance = math.dist(raw[:2],self.previous_raw[:2])
            if distance>100:
                raise RuntimeError('座標が急変しました。初期位置を再設定してください')
            self.distance += distance
        self.previous_raw = raw
        return self.slam._get_estimated_pose(*raw)

    def key(self, status):
        pose = self.pose() if self.started else None
        self.window.draw(
            self.world,
            poses=self.setup['robots'],
            current=pose,
            status=status,
            target=self.planner.goal,
            cube_candidates=self.cube_reviews.unresolved(self.world),
            mapping_status=self.mapping_status,
            localization=self.localizer.report if self.localizer else None,
            route=self.planner.return_path if self.planner.mode == 'return' else self.planner.path,
            view=(self.config.view_near_mm, self.config.view_far_mm,
                  self.config.view_fov_degrees, self.config.camera_forward_mm),
        )
        self.camera.display_once()
        key = self.camera.poll_key(10)
        if key in (ord('q'),27):
            raise KeyboardInterrupt
        if key==ord('p') and self.started:
            self.paused = not self.paused
            self.log('pause' if self.paused else 'resume')
            self.cli.drive_wheels(0,0)
        if key==ord('i') and self.started:
            self.cli.drive_wheels(0, 0)
            self.interventions += 1
            self.log('intervention',count=self.interventions)
            self.paused = True
        return key

    def await_start(self):
        """Enter が押されたら、待ち時間なしで探索を開始する。

        以前は placement_wait_seconds（既定30秒）が経過するまで Enter を
        受け付けなかった。実験準備は操作者が確認してから Enter を押すため、
        現在は時間待ちを行わない。2台探索だけは Bluetooth 接続完了を確認する。
        """
        cfg = self.config
        if self.robot_count == 1:
            print('【1台探索】キューブとCozmoの配置を確認し、Enterで探索を開始してください。')
            print('Bluetooth通信は使用しません。30秒の自動待機はありません。')
        else:
            print('【2台協調探索】全機とキューブの配置を確認してください。')
            print('Bluetooth接続後、各PCでEnterを押すと探索を開始します。')

        while True:
            ready = all(self.network.status.get(peer)=='connected' for peer in self.network.links)
            if self.robot_count == 1:
                status = 'READY / SOLO / ENTER to start'
            else:
                status = f'READY / BT={ready} / ENTER when all PCs ready'

            key = self.key(status + ' / K: cube calibration')
            if key == ord('k'):
                calibrate_stopped(self.cli, self.camera, self.perception, cfg)
                continue
            if key not in (10,13):
                continue

            if not ready:
                print('Bluetooth接続待ち:',self.network.status)
                continue

            try:
                self.cli.acknowledge_placement()
            except RuntimeError as error:
                print(error)
                continue

            p = self.cli.pose
            if p is None:
                print('Cozmoの姿勢情報を待っています。')
                continue

            initial = self.setup['robots'][cfg.robot_id]
            self.slam._set_estimated_pose(
                p.position.x, p.position.y, p.rotation.angle_z.radians,
                initial['x'], initial['y'], initial['theta'],
            )

            # Enter を押した時点を実験開始時刻とする。
            self.started = time.monotonic()
            self.started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            self.stale_progress_since = self.started

            run_setup = dict(self.setup)
            run_setup['run_conditions'] = {
                'robot_count': self.robot_count,
                'active_robots': list(self.active_robots),
                'bluetooth_enabled': bool(self.network.links),
                'sharing_mode': cfg.sharing_mode,
                'start_method': 'enter_key',
                'exploration_revision': cfg.exploration_revision,
                'observation_frames': max(cfg.observation_frames, cfg.marker_confirm_frames),
                'cube_calibration': dict(focal_px=cfg.cube_focal_px,
                                         detection_region=cfg.cube_detection_region,
                                         model_sha256=self.perception.model_sha256 if hasattr(self.perception, 'model_sha256') else None,
                                         distance_scale=cfg.cube_distance_scale,
                                         camera_forward_mm=cfg.camera_forward_mm,
                                         confirm_far_mm=cfg.cube_confirm_far_mm,
                                         confirm_viewpoints=cfg.cube_confirm_viewpoints),
            }
            save_json(self.output/'setup.json',run_setup)
            self.log(
                'start',
                started_utc=self.started_utc,
                robot_count=self.robot_count,
                active_robots=list(self.active_robots),
                bluetooth_enabled=bool(self.network.links),
                start_method='enter_key',
            )
            return

    @staticmethod
    def _raw_pose_snapshot(cli):
        """Cozmo本体のオドメトリを、走行確認用に軽量なtupleで取得する。"""
        p = cli.pose
        if p is None:
            return None
        return (p.position.x, p.position.y, p.rotation.angle_z.radians)

    def _run_motion_burst(self, command, status_prefix="MOVING", use_target=True):
        """観測間の走行を、目標角度・地点で止まれる専用処理へ渡す。"""
        if command == (0, 0) or getattr(self, 'cube_stop_active', False):
            self.cli.drive_wheels(0, 0)
            return
        if hasattr(self.camera, 'clear_detections'):
            # 停止中に推論した古い枠を、移動中の別画像へ重ね続けない。
            self.camera.clear_detections('research')
        action = 'TURN' if command[0] * command[1] < 0 else 'ADVANCE'
        if self.planner.mode == 'verify':
            status_prefix = 'VERIFYING CUBE'
        if self.planner.mode == 'inspection':
            action = 'SCAN' if action == 'TURN' and not self.planner.path else action
        if action != self._last_action:
            print({'TURN': '↪ 目標方向を向きます', 'ADVANCE': '→ 観測地点へ前進します',
                   'SCAN': '↪ 巡回地点で周囲を確認します'}[action])
            self._last_action = action

        def process_events(left, right):
            self.key(f'{status_prefix} / {action} / cubes={len(self.world.cubes())} / P pause / Q stop')
            self.exchange.tick()

        def keep_running():
            if self.paused:
                return False
            if self.return_started is not None:
                return time.monotonic() < self.return_started + self.config.return_home_timeout_seconds
            return not self.trial_time_expired()

        result = run_motion(
            self.cli, self.config, command, self.pose, process_events, keep_running,
            target=self.planner.motion_target if use_target else None,
            heading=self.planner.motion_heading if use_target else None,
            can_advance=self.planner.can_advance,
            clock=time,
        )
        self.log('motion', action=action, **result)
        if result['stop_reason'] == 'path_blocked' and self.planner.mode != 'return':
            self.planner.path, self.planner.goal = [], None
            self.planner.inspection_target_key = None
            self.planner.inspection_scan_queue = []
        if result['duration'] > 0.25 and result['moved_mm'] < 0.5 and result['turned_degrees'] < 0.5:
            print('車輪命令に対して位置・向きの変化がありません。実機の車輪と床面を確認してください。')

    def _map_camera_detections(self, candidates, boxes):
        """同じ採否結果を、50mm角の地図登録とカメラ枠の両方へ反映する。"""
        observations = getattr(self.perception, 'cube_observations', None)
        if observations is None:
            # 検出器を差し替えた場合にも従来の戻り値を利用できる。
            return [point for point in candidates if self.world.grid.inside(point)]
        # 差し替え検出器・単体試験でも利用できるよう初回に生成する。
        if not hasattr(self, 'cube_reviews'):
            self.cube_reviews = CubeReviews(self.config)
        accepted = self.cube_reviews.update(observations, self.world.grid, time.monotonic())
        self.live_cube_candidates = [v['point'] for v in self.cube_reviews.unresolved(self.world)]
        rejected = []
        for observation in observations:
            status = observation['review_status']
            point = observation['point']
            if status != 'MAPPED':
                rejected.append(dict(reason=status, point=point, distance=observation['distance']))
            distance = observation['distance']
            refined = observation.get('refined_range_mm')
            if refined is not None:
                distance_text = f' ~{refined:.0f}mm({observation.get("refinement", "refined")})'
            else:
                distance_text = '' if distance is None else f' ~{distance:.0f}mm'
            name = 'Cube' if observation.get('visual_confirmed', status == 'MAPPED') else 'Cube? STOP'
            boxes[observation['box_index']]['label'] = f'{name} {status}{distance_text}'

        self.mapping_status = f'DETECTED={len(observations)} / MAPPED={len(accepted)}'
        if rejected:
            first = rejected[0]
            self.mapping_status += ' / ' + first['reason']
            if first['point'] is not None:
                self.mapping_status += ' ({:.0f},{:.0f})mm'.format(*first['point'])
            signature = tuple(item['reason'] for item in rejected)
            if signature != self.last_mapping_warning:
                self.log('cube_mapping_rejected', detections=rejected)
                print('📦 キューブ候補を表示・再確認:', self.mapping_status)
            self.last_mapping_warning = signature
        else:
            self.last_mapping_warning = None
        return accepted

    def trial_time_expired(self):
        """0秒は上限なし。指定された時間での中断は完了条件とは別扱い。"""
        return (self.started is not None and self.config.trial_limit_seconds > 0
                and time.monotonic() - self.started >= self.config.trial_limit_seconds)

    def _set_cube_stop(self, active):
        now = time.monotonic()
        if active and not self.cube_stop_active:
            self.cube_stop_started = now
            self.cube_stop_count += 1
            self.cli.drive_wheels(0, 0)
            self.log('cube_stop_start')
        elif not active and self.cube_stop_active:
            seconds = now - self.cube_stop_started
            self.cube_stop_seconds += seconds
            self.cube_stop_started = None
            self.log('cube_stop_end', seconds=seconds)
        self.cube_stop_active = active

    def observe_stopped(self, record=True):
        """新しい停止画像でチェス→自己位置→キューブの順に確認する。

        帰還中はrecord=Falseとして探索実績を増やさず、位置補正と回避を続ける。
        """
        self.cli.drive_wheels(0, 0)
        stopped = time.monotonic()
        count = 0
        visible = None
        deadline = stopped + max(8, self.config.frame_timeout_seconds * 3)
        markers = []
        robot_seen = False
        marker_seen = set()
        close_cube_seen = False

        frame_cube_candidates = []
        all_occluders = []
        stop_lock = CubeStopLock(self.config)

        required_frames = max(self.config.observation_frames, self.config.marker_confirm_frames,
                              self.config.cube_confirm_frames)
        chess_pending = False
        while (count < required_frames or stop_lock.pending
               or (chess_pending and count < self.config.chess_refine_window_frames)):

            status = (f'CUBE? STOP / pending={stop_lock.pending}' if stop_lock.pending else
                      f'CHESS AVERAGING {count}' if chess_pending else
                      f'OBSERVING {count}/{required_frames}')
            if self.cube_stop_active and time.monotonic() - self.cube_stop_started > self.config.cube_hold_warning_seconds:
                status += ' / WAITING FOR STABLE DETECTION'
            self.key(status + f' / cubes={len(self.world.cubes())}: Q stop / P pause')
            self.exchange.tick()
            if self.cli.latched_reason:
                raise RuntimeError(self.cli.latched_reason)
            percent = self.battery.percent
            if percent is not None and percent < self.config.minimum_battery_percent:
                raise TrialInterrupted('low_battery')
            if record and self.trial_time_expired():
                raise TrialInterrupted('time_limit')
            if not record and self.return_started is not None and time.monotonic() >= self.return_started + self.config.return_home_timeout_seconds:
                raise TrialInterrupted('return_home_timeout')

            if self.paused:
                # 一時停止中の画像を探索結果へ入れず、再開後に撮り直す。
                stopped = time.monotonic()
                deadline = stopped + max(8, self.config.frame_timeout_seconds * 3)
                count, visible = 0, None
                chess_pending = False
                frame_cube_candidates.clear()
                all_occluders.clear()
                robot_seen, close_cube_seen = False, False
                marker_seen.clear()
                stop_lock = CubeStopLock(self.config)
                self._set_cube_stop(False)
                continue
            frame, number, stamp = self.camera.get_frame_sample()
            if frame is None or number == self.last_frame or stamp < stopped + self.config.observation_settle_seconds:
                if time.monotonic() > deadline:
                    raise RuntimeError('新しいカメラ画像が取得できません')
                continue
            if time.monotonic() - stamp > self.config.frame_timeout_seconds:
                raise RuntimeError('画像の受信が停止しました')

            self.last_frame = number
            pose = self.pose()
            chess_boxes = []
            if self.localizer is not None:
                before = pose
                raw = self._raw_pose_snapshot(self.cli)
                pose, report, chess_boxes = self.localizer.observe_frame(
                    self.slam, frame, raw, pose, time.monotonic(), self.distance)
                chess_pending = report.get('refinement_pending', False)
                if report.get('landmark_updates'):
                    # 未確認駒への観測方向・経路・遮蔽を、新しい確定座標で作り直す。
                    self.planner.path, self.planner.goal = [], None
                    self.planner.cancel_landmark_route()
                    self.planner.landmark_views.attempts.clear()
                    visible = None
                    all_occluders.clear()
                    for update in report['landmark_updates']:
                        self.log('chess_position_updated', **update)
                    if self.planner.mode == 'return':
                        self.planner.start_return(pose, tuple(self.setup['robots'][self.config.robot_id][k]
                                                             for k in ('x', 'y', 'theta')))
                self.localization_status = report['status']
                self.log('chess_localization', **report, pose=list(pose))
                if report['corrected'] and (math.dist(before[:2], pose[:2]) > 3.0
                        or abs(pose[2] - before[2]) > math.radians(1)):
                    self.cube_reviews.reset_after_pose_fix(before, pose)
                    # 古い姿勢の候補・遮蔽・経路と、新しい姿勢を混ぜない。
                    visible = None
                    all_occluders.clear()
                    self.planner.path, self.planner.goal = [], None
                    if self.planner.mode == 'return':
                        self.planner.start_return(pose, tuple(self.setup['robots'][self.config.robot_id][k]
                                                             for k in ('x', 'y', 'theta')))
            candidates, markers, boxes, occluders = self.perception.detect(frame, pose)
            boxes.extend(chess_boxes)
            observations = getattr(self.perception, 'cube_observations', [])
            for observation in observations:
                observation['previously_confirmed'] = self.cube_reviews.previously_confirmed(observation)
            stop_lock.update(observations)
            self.cube_reviews.discard_lost_candidates(stop_lock.lost)
            if stop_lock.lost:
                self.log('cube_stop_lost', candidates=len(stop_lock.lost))
            self._set_cube_stop(bool(stop_lock.pending))
            for observation in getattr(self.perception, 'cube_observations', []):
                observation['pose_verified'] = self._pose_is_verified()
                # Cozmoの累積オドメトリ走行量。接近前後の差をキューブ測距へ使う。
                observation['travelled_mm'] = self.distance
            robot_seen = robot_seen or self.perception.robot_in_view
            marker_seen.update(markers)
            close_cube_seen = close_cube_seen or getattr(self.perception, 'close_cube_unlocalized', False)
            candidates = self._map_camera_detections(candidates, boxes)
            self.camera.set_detections('research', boxes)

            frame_cube_candidates.append(candidates)
            # 確認に時間がかかっても画像・遮蔽リストを無制限に増やさない。
            frame_cube_candidates = frame_cube_candidates[-required_frames:]

            if record:
                self._register_detected_cubes(candidates)
            self.mapping_status += ' / ' + self.localization_status

            # 未確定候補は画面から消えても保持する。地図外でも表示し、
            # 位置と信頼度が確認できるまでは共有する確定台帳へ入れない。
            if hasattr(self, 'cube_reviews'):
                self.live_cube_candidates = [v['point'] for v in self.cube_reviews.unresolved(self.world)]
            else:
                self.live_cube_candidates = list(getattr(self.perception, 'preview_cubes', candidates))

            # 一度見えた未確定候補も回避する。画面から消えても突っ込まない。
            all_occluders = list(occluders)
            self.planner.set_transient_obstacles(all_occluders + [
                (*point, self.config.cube_width_mm / 2) for point in self.live_cube_candidates])

            for mid in markers:
                peer = next(
                    (
                        name
                        for name, robot_pose in self.setup['robots'].items()
                        if robot_pose['marker_id'] == mid
                        and name != self.config.robot_id
                    ),
                    None,
                )
                if peer:
                    self.exchange.seen(peer, mid)

            cfg = self.config
            cpose = (
                pose[0] + cfg.camera_forward_mm * math.cos(pose[2]),
                pose[1] + cfg.camera_forward_mm * math.sin(pose[2]),
                pose[2],
            )
            cells = self.world.grid.visible(
                cpose,
                cfg.view_near_mm,
                cfg.view_far_mm,
                math.radians(cfg.view_fov_degrees),
                occluders + self.planner.occluders(),
            )

            # 全確認フレームで観測可能なセルだけ探索済みにする。
            # 一時的にキューブが見えた背後は occluder で除外される。
            if not self._pose_is_verified():
                cells = set()
            visible = cells if visible is None else visible & cells
            count += 1
            deadline = time.monotonic() + 8

        # 登録自体は各フレームの検知直後に済んでいる。
        # ここでは候補表示の整理だけを行う。
        # 確定済みの近くにある候補は、MapWindowで二重表示されない。
        # 未確定候補だけは次の描画まで残し、「カメラでは見えている」ことを
        # 地図上でも確認できるようにする。
        if frame_cube_candidates:
            known = self.world.cubes()
            self.live_cube_candidates = [
                point
                for point in self.live_cube_candidates
                if not any(
                    math.hypot(point[0] - cube['x'], point[1] - cube['y'])
                    <= self.config.cube_merge_mm
                    for cube in known
                )
            ]

        self.perception.robot_in_view = robot_seen
        self.perception.close_cube_unlocalized = close_cube_seen
        self.marker_ids = [
            mid
            for mid in sorted(marker_seen)
            if mid != self.setup['robots'][self.config.robot_id]['marker_id']
        ]
        if not self.paused and not self.marker_ids and not self.perception.robot_in_view and not close_cube_seen:
            if record and self._pose_is_verified():
                # 停止地点を実走行軌跡として残し、仮想フェロモンを更新する。
                # 回転だけを続けると同じセルの濃度が上がるため、planner は
                # 次に低フェロモンの離れた地点を選びやすくなる。
                self.world.evaporate_pheromones(self.config.pheromone_evaporation_rate)
                self.world.observe(pose, visible or set())
                self.world.deposit_pheromone(
                    pose,
                    self.config.pheromone_deposit,
                    self.config.pheromone_spread_cells,
                )

        self.log(
            'observation',
            frames=count,
            observation_seconds=time.monotonic() - stopped,
            mapping_status=self.mapping_status,
            localization=self.localizer.report if self.localizer else None,
            x=pose[0],
            y=pose[1],
            theta=pose[2],
            local_covered=len(self.world.local_covered),
            known_covered=len(self.world.covered),
            cubes=len(self.world.cubes()),
            cube_candidates=sum(len(items) for items in frame_cube_candidates),
            distance_mm=self.distance,
            trail_coverage_ratio=self.trail_coverage_ratio(),
            pheromone_cells=len(self.world.pheromones),
        )
        self.last_observed_pose = tuple(pose)
        return pose

    def coverage_ratio(self):
        """現在までに探索済みになった地図セルの割合を 0.0～1.0 で返す。"""
        total = len(self.world.grid.cells)
        return 0.0 if total == 0 else len(self.world.covered) / total

    def trail_coverage_ratio(self):
        """自機+共有済み他機の軌跡から求めた物理的な巡回率。"""
        return self.world.trail_coverage_ratio(self.config.trail_coverage_radius_mm)

    def recognized_chess_ids(self):
        """今回の試行で複数フレーム確認できたチェス駒ID。"""
        localizer = getattr(self, 'localizer', None)
        if localizer is None:
            return set()
        return set(getattr(localizer, 'seen_ids', set()))

    def required_chess_count(self):
        return int(self.config.required_chess_landmarks)

    def unseen_chess_ids(self):
        setup = getattr(self, 'setup', None) or getattr(getattr(self, 'world', None), 'setup', {})
        all_ids = {str(v) for v in setup.get('landmarks', {})}
        return all_ids - self.recognized_chess_ids()

    def exploration_goal_reached(self):
        """6駒確認 + マップ面積95%確認で探索終了。

        実走行率はフェロモン探索の性能指標として保存するが、終了条件には
        しない。これにより研究条件を「6境界ランドマーク確認と95%面探索」に
        明確化する。
        """
        return (
            self.coverage_ratio() >= self.config.return_home_coverage_ratio
            and len(self.setup['landmarks']) >= self.required_chess_count()
            and not self.unseen_chess_ids()
            and not getattr(self, 'cube_stop_active', False)
            and not getattr(getattr(self, 'localizer', None), 'refinement_pending', False)
        )

    def mark_exploration_finished(self, reason):
        """帰還を始める前の探索性能を固定して記録する。"""
        if self.exploration_finished_seconds is not None:
            return

        self.completion_reason = reason
        self.exploration_finished_seconds = time.monotonic() - self.started
        self.exploration_finished_distance_mm = self.distance
        self.exploration_coverage_ratio = self.coverage_ratio()
        self.log(
            'exploration_finished',
            reason=reason,
            coverage_ratio=self.exploration_coverage_ratio,
            trail_coverage_ratio=self.trail_coverage_ratio(),
            covered_cells=len(self.world.covered),
            total_cells=len(self.world.grid.cells),
            cubes=len(self.world.cubes()),
            chess_seen=sorted(self.recognized_chess_ids()),
            chess_seen_count=len(self.recognized_chess_ids()),
            distance_mm=self.exploration_finished_distance_mm,
        )

    def return_home(self):
        """探索終了後、setup.json に保存された初期姿勢へ自動で戻る。

        帰還中は新しい探索セルやキューブを追加しない。これにより、
        「探索に要した時間・距離」と「帰還に要した時間・距離」を分離できる。
        """
        initial = self.setup['robots'][self.config.robot_id]
        home_pose = (initial['x'], initial['y'], initial['theta'])
        pose = self.pose()

        if not self.planner.start_return(pose, home_pose):
            self.reason = 'return_home_unreachable'
            self.log('return_home_failed', reason=self.reason)
            return False

        self.return_started = time.monotonic()
        self.return_started_distance = self.distance
        deadline = self.return_started + self.config.return_home_timeout_seconds
        self.log(
            'return_home_start',
            x=pose[0],
            y=pose[1],
            home_x=home_pose[0],
            home_y=home_pose[1],
        )
        print('🏠 探索完了。初期位置へ戻ります。')

        while time.monotonic() < deadline:
            self.exchange.tick()
            if self.cli.latched_reason:
                raise RuntimeError(self.cli.latched_reason)

            percent = self.battery.percent
            if percent is not None and percent < self.config.minimum_battery_percent:
                self.reason = 'low_battery_during_return'
                return False

            if self.paused:
                self.cli.drive_wheels(0, 0)
                self.key('RETURN PAUSED / P: resume / Q: stop')
                continue

            pose = self.observe_stopped(record=False)
            if self._recheck_localization():
                continue
            command = self.planner.return_command(pose)

            if command is None:
                self.cli.drive_wheels(0, 0)
                self.returned_home = True
                self.return_home_seconds = time.monotonic() - self.return_started
                self.return_home_distance_mm = self.distance - self.return_started_distance
                self.reason = 'returned_home'
                self.log(
                    'return_home_complete',
                    seconds=self.return_home_seconds,
                    distance_mm=self.return_home_distance_mm,
                    x=pose[0],
                    y=pose[1],
                    theta=pose[2],
                )
                print('✅ 初期位置へ戻りました。')
                return True

            if command == (0, 0):
                self.cli.drive_wheels(0, 0)
                self.reason = 'return_home_path_blocked'
                self.log('return_home_failed', reason=self.reason)
                return False

            self._run_motion_burst(command, status_prefix='RETURNING HOME')

        self.cli.drive_wheels(0, 0)
        self.reason = 'return_home_timeout'
        self.return_home_seconds = time.monotonic() - self.return_started
        self.return_home_distance_mm = self.distance - self.return_started_distance
        self.log(
            'return_home_failed',
            reason=self.reason,
            seconds=self.return_home_seconds,
            distance_mm=self.return_home_distance_mm,
        )
        return False

    def save(self):
        elapsed = 0 if self.started is None else time.monotonic() - self.started
        data = self.world.export()
        data['cube_candidates'] = self.cube_reviews.unresolved(self.world)
        data['localization'] = self.localizer.report if self.localizer else {}
        data['chess_position_updates'] = list(getattr(getattr(self.localizer, 'refiner', None), 'history', []))
        # 観測後に走行して終了した場合にも、保存時点の現在地を残す。
        try:
            data['current_pose'] = list(self.pose()) if self.started is not None else None
        except (RuntimeError, AttributeError, TypeError):
            data['current_pose'] = list(self.last_observed_pose) if self.last_observed_pose else None

        data['metrics'] = dict(
            elapsed_seconds=elapsed,
            distance_mm=self.distance,
            first_all_known_seconds=self.first_all_known,
            first_local_all_seconds=self.first_local_all,
            expected_cubes=self.config.expected_cubes,
            covered_cells=len(self.world.covered),
            total_cells=len(self.world.grid.cells),
            coverage_ratio=self.coverage_ratio(),
            chess_seen=sorted(self.recognized_chess_ids()),
            chess_seen_count=len(self.recognized_chess_ids()),
            required_chess_count=self.required_chess_count(),
            chess_unseen=sorted(self.unseen_chess_ids()),
            exploration_complete=self.completion_reason is not None,
            cube_stop_count=self.cube_stop_count,
            cube_stop_seconds=self.cube_stop_seconds + (time.monotonic() - self.cube_stop_started if self.cube_stop_active else 0),
            exploration_revision=self.config.exploration_revision,
            trail_covered_cells=len(self.world.trail_coverage_cells(self.config.trail_coverage_radius_mm)),
            total_walkable_cells=len(self.world.grid.walkable),
            trail_coverage_ratio=self.trail_coverage_ratio(),
            pheromone_cells=len(self.world.pheromones),
            completion_reason=self.completion_reason,
            exploration_finished_seconds=self.exploration_finished_seconds,
            exploration_finished_distance_mm=self.exploration_finished_distance_mm,
            exploration_coverage_ratio=self.exploration_coverage_ratio,
            returned_home=self.returned_home,
            return_home_seconds=self.return_home_seconds,
            return_home_distance_mm=self.return_home_distance_mm,
            interventions=self.interventions,
            reason=self.reason,
            sharing_mode=self.config.sharing_mode,
            started_utc=self.started_utc,
            robot_count=self.robot_count,
            active_robots=list(self.active_robots),
            bluetooth_enabled=bool(self.network.links),
        )
        save_json(self.output / 'exploration.json', data)

    def run(self):
        self.reason = 'running'
        try:
            self.cli.set_head_angle(self.config.head_angle)
            self.cli.set_lift_height(0.0)
            self.network.start()
            self.await_start()

            while not self.trial_time_expired():
                self.exchange.tick()
                if self.cli.latched_reason:
                    raise RuntimeError(self.cli.latched_reason)

                percent = self.battery.percent
                if percent is not None and percent < self.config.minimum_battery_percent:
                    self.reason = 'low_battery'
                    break

                mode_text = (
                    'VERIFY CUBE' if self.planner.mode == 'verify' else
                    'CHESS PATROL' if self.planner.mode == 'landmark_search' else
                    'EXPLORE'
                )
                self.key(
                    'PAUSED: P resume'
                    if self.paused
                    else (
                        f'{mode_text} / {self.config.robot_id} / '
                        f'cubes={len(self.world.cubes())}/{self.config.expected_cubes or "?"} / '
                        f'chess={len(self.recognized_chess_ids())}/{self.required_chess_count()} / '
                        f'view={self.coverage_ratio() * 100:.1f}% / '
                        f'route={self.trail_coverage_ratio() * 100:.1f}%'
                    )
                )

                if self.paused:
                    self.cli.drive_wheels(0, 0)
                    continue

                pose = self.observe_stopped()
                expected = self.config.expected_cubes
                elapsed = time.monotonic() - self.started

                if (
                    expected > 0
                    and len(self.world.cubes()) >= expected
                    and self.first_all_known is None
                ):
                    self.first_all_known = elapsed
                    self.log('all_known', cube_count=len(self.world.cubes()))

                local_groups = [
                    group
                    for group in self.world.cubes()
                    if any(key in self.world.local_found for key in group['aliases'])
                ]
                if (
                    expected > 0
                    and len(local_groups) >= expected
                    and self.first_local_all is None
                ):
                    self.first_local_all = elapsed
                    self.log('all_locally_found')


                if self._recheck_localization():
                    continue

                # 6駒確認 + 視認率95%を満たした場合だけ帰還する。
                if not self.paused and self.exploration_goal_reached():
                    self.mark_exploration_finished('six_chess_and_95pct_view')
                    self.return_home()
                    break

                expected = self.config.expected_cubes
                missing_expected_cubes = (
                    expected > 0 and len(self.world.cubes()) < expected
                )

                trail_cells = len(self.world.trail_coverage_cells(self.config.trail_coverage_radius_mm))
                chess_seen_count = len(self.recognized_chess_ids())
                progressed = (
                    len(self.world.covered) > self.last_coverage
                    or trail_cells > self.last_trail_coverage
                    or chess_seen_count > self.last_chess_seen
                )
                if progressed:
                    self.last_coverage = max(self.last_coverage, len(self.world.covered))
                    self.last_trail_coverage = max(self.last_trail_coverage, trail_cells)
                    self.last_chess_seen = max(self.last_chess_seen, chess_seen_count)
                    self.stale_progress_since = time.monotonic()
                elif time.monotonic() - self.stale_progress_since > self.config.exploration_stall_timeout_seconds:
                    # v7では「6駒 + 95%」未達のまま停滞だけを理由に探索完了扱いしない。
                    # 現在の目標を捨てて再計画し、別の場所へ動く。
                    self.log('exploration_stall_recovery',
                             coverage_ratio=self.coverage_ratio(),
                             chess_seen=sorted(self.recognized_chess_ids()))
                    self.planner.path, self.planner.goal = [], None
                    self.planner.landmark_target_id = None
                    self.planner.landmark_waiting_observation = False
                    self.cube_verifier.reset_route(self.planner)
                    self.stale_progress_since = time.monotonic()


                if self.paused:
                    continue

                if not self.world.grid.inside(pose[:2]):
                    raise RuntimeError(
                        '機体位置が地図範囲外です。座標と初期位置を確認してください'
                    )

                pending_cubes = bool(self.cube_reviews.unresolved(self.world))
                # 他機体は必ず譲る。近接キューブは「その候補を確認中」なら
                # 逃げずに停止観測し、確認中でないときだけ回避する。
                yield_needed = (bool(self.marker_ids) or self.perception.robot_in_view
                                or (getattr(self.perception, 'close_cube_unlocalized', False)
                                    and not pending_cubes))
                if yield_needed:
                    command = (self.config.turn_speed_mm_s, -self.config.turn_speed_mm_s)
                    self.cube_verifier.reset_route(self.planner)
                    self.planner.goal = None
                else:
                    # 1. 存在確認済みキューブのうち、位置が未確定の物を再観測する。
                    if self._pose_is_verified():
                        command = self.cube_verifier.command(
                            pose, self.planner, time.monotonic(), self.distance
                        )
                    else:
                        self.cube_verifier.reset_route(self.planner)
                        command = None
                    if command is not None:
                        self.mapping_status = 'VERIFYING CUBE / ' + self.mapping_status.removeprefix('VERIFYING CUBE / ')
                    else:
                        # 2. 未確認チェスが残るなら、中央スキャンまたは周回観測点へ移動
                        unseen = self.unseen_chess_ids()
                        if not unseen and not self._pose_is_verified():
                            # 全駒の既視認フラグがあっても自己位置は永久には信用しない。
                            unseen = set(self.setup['landmarks'])
                        if unseen:
                            old_view = self.planner.landmark_target_id, self.planner.goal
                            command = self.planner.landmark_command(pose, unseen)
                            new_view = self.planner.landmark_target_id, self.planner.goal
                            if new_view != old_view and self.planner.landmark_target_id is not None:
                                self.log('chess_viewpoint', landmark_id=new_view[0],
                                         target=new_view[1], clear_view=self.planner.landmark_view_clear,
                                         missing=sorted(self.unseen_chess_ids()))
                        else:
                            # 3. 6駒確認後は95%面積を埋めるフェロモン探索
                            self.planner.landmark_command(pose, set())
                            command = self.planner.command(pose)

                if command is None:
                    if self.exploration_goal_reached():
                        self.mark_exploration_finished('six_chess_and_95pct_view')
                        self.return_home()
                        break
                    # 終了条件未達なら、淵で外向きに回り続けず中央方向へ回復する。
                    command = self.planner.recovery_command(pose)

                self.reason = 'running'
                self._run_motion_burst(command, use_target=not yield_needed)

                if time.monotonic() - self.last_saved > 2:
                    self.save()
                    self.last_saved = time.monotonic()

            if self.reason == 'running':
                self.reason = 'time_limit'

        except TrialInterrupted as error:
            self.reason = str(error)
        except KeyboardInterrupt:
            self.reason = 'user_stop'
        except Exception as error:
            self.reason = 'error: ' + str(error)
            self.log('error', reason=str(error))
            raise
        finally:
            try:
                self.cli.drive_wheels(0, 0)
            finally:
                self._set_cube_stop(False)
                self.network.stop()
                self.save()
                self.log('end', reason=self.reason)
                self.events.close()
                self.window.close()
                print('実験記録:', self.output)
