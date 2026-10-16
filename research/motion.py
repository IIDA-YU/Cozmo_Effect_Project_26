"""目標角度・距離で止まる短区間走行。観測と経路選択は別モジュール。"""
import math
import time

from research.world import wrap


def run_motion(cli, config, command, get_pose, process_events, keep_running,
               target=None, heading=None, can_advance=None, clock=time):
    """40mm以下の前進／目標方向への旋回を行い、必ず停止して戻る。

    SafetyClientへ短い間隔で命令を送る。経路計算や画像推論は走行中に行わない。
    時計・機体・姿勢取得を渡せるため、実機なしのオドメトリ試験もできる。
    """
    left, right = command
    start_time = clock.monotonic()
    start = get_pose()
    turning = left * right < 0
    if turning:
        speed = max(25.0, abs(left), abs(right))
        left, right = math.copysign(speed, left), math.copysign(speed, right)
    # 旧い短区間設定でも動き出せる。ただし到達したら直ちに止める。
    duration = max(0.35 if turning else 0.45, config.motion_pulse_seconds)
    end_time = start_time + duration
    initial_delta = None if heading is None else wrap(heading - start[2])
    reason = 'time_slice'
    try:
        while (left or right) and clock.monotonic() < end_time and keep_running():
            pose = get_pose()
            if turning and heading is not None:
                delta = wrap(heading - pose[2])
                # 符号反転も検出し、行き過ぎたまま旋回し続けない。
                if abs(delta) <= math.radians(config.turn_tolerance_degrees) or delta * initial_delta < 0:
                    reason = 'heading_reached'
                    break
            elif not turning:
                travelled = math.dist(start[:2], pose[:2])
                if travelled >= config.motion_step_mm:
                    reason = 'distance_slice'
                    break
                if target is not None:
                    if math.dist(pose[:2], target) <= 10.0:
                        reason = 'waypoint_reached'
                        break
                    direction = math.atan2(target[1] - pose[1], target[0] - pose[0])
                    if abs(wrap(direction - pose[2])) > math.radians(10):
                        reason = 'realign'
                        break
                # 次の命令区間と停止余裕の分だけ、実際の向きでも障害物を確認する。
                lookahead = max(abs(left), abs(right)) * 0.12 + 5.0
                if can_advance is not None and not can_advance(pose, lookahead):
                    reason = 'path_blocked'
                    break

            cli.drive_wheels(left, right)
            process_events(left, right)
            if cli.latched_reason:
                raise RuntimeError(cli.latched_reason)
            clock.sleep(0.04)
    finally:
        cli.drive_wheels(0, 0)

    end = get_pose()
    return dict(left_speed=left, right_speed=right,
                duration=clock.monotonic() - start_time,
                moved_mm=math.dist(start[:2], end[:2]),
                turned_degrees=abs(math.degrees(wrap(end[2] - start[2]))),
                stop_reason=reason)
