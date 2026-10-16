"""ant-COZMO アプリケーションの起動処理。

main.py から呼び出され、次の流れを担当します。

1. config.py の設定値を確認する
2. 実行モードを決める
3. setup.json の読み書きを行う
4. Cozmo と接続し、必要なモジュールを起動する

探索・通信・安全制御などの具体的な処理は、このファイルへ直接書かず、
research/ や各専用モジュールを呼び出して使用します。
"""

import argparse
import json
import math
import threading
import uuid

from application.config import CONFIG
from application.setup_upgrade import upgrade_exploration_conditions
from research.world import Grid, fingerprint, save_json


# ----------------------------------------------------------------------
# setup.json で全 PC に共有する実験条件
# ----------------------------------------------------------------------
# ここにある項目は、setup モードで保存した値を run モードで読み込みます。
# そのため、各 PC で異なる値のまま実験が始まることを防げます。
SHARED_FIELDS = (
    "chess_refine_enabled", "chess_refine_min_samples", "chess_refine_window_frames",
    "chess_refine_window_seconds", "chess_refine_sample_interval_seconds",
    "chess_refine_min_span_seconds", "chess_refine_spread_mm", "chess_refine_outlier_mm",
    "chess_refine_min_shift_mm", "chess_refine_max_shift_mm", "chess_refine_cooldown_seconds",
    "chess_refine_pose_age_seconds", "chess_refine_pose_travel_mm", "chess_refine_far_mm",
    "cube_detection_region", "cube_duplicate_iou", "cube_stopped_match_iou",
    "cube_stop_missing_frames", "cube_hold_warning_seconds",
    "landmark_view_min_mm", "landmark_view_max_mm",
    "landmark_vantage_separation_mm", "landmark_occlusion_margin_mm",
    "cube_track_max_match_mm", "cube_track_ambiguity_mm", "cube_candidate_ttl_seconds",
    "cube_lost_grace_seconds", "cube_lost_forward_mm", "cube_approach_step_mm",
    "cube_approach_stop_mm", "cube_approach_stop_box_height_ratio",
    "cube_bearing_match_degrees", "cube_bearing_match_origin_mm", "cube_odometry_min_approach_mm",
    "chess_confidence", "chess_match_mm", "chess_ambiguity_mm", "chess_min_baseline_mm",
    "chess_pair_tolerance_mm", "chess_residual_mm", "chess_max_shift_mm",
    "chess_max_turn_degrees", "chess_confirm_frames",
    "required_chess_landmarks", "chess_seen_confirm_frames", "chess_seen_match_mm",
    "center_scan_radius_mm", "center_scan_step_degrees", "edge_scan_guard_mm",
    "edge_inward_half_angle_degrees", "landmark_vantage_distance_mm",
    "localization_max_age_seconds", "localization_max_travel_mm", "localization_scan_timeout_seconds",

    "sharing_mode",
    "placement_wait_seconds",
    "trial_limit_seconds",
    "expected_cubes",
    "cell_mm",
    "boundary_margin_mm",
    "cube_merge_mm",
    "cube_width_mm",
    "cube_confidence",
    "cube_candidate_confidence",
    "cube_review_confirm_confidence",
    "cube_review_confirm_frames",
    "cube_review_distance_mm",
    "cube_review_timeout_seconds",
    "cube_review_retry_seconds",
    "cube_confirm_frames",
    "cube_confirm_viewpoints",
    "cube_viewpoint_baseline_mm",
    "cube_triangulation_min_angle_degrees",
    "cube_viewpoint_preferred_angle_degrees",
    "cube_candidate_match_mm",
    "cube_lost_grace_seconds",
    "cube_lost_forward_mm",
    "cube_approach_step_mm",
    "cube_approach_stop_mm",
    "cube_bearing_match_degrees",
    "cube_bearing_match_origin_mm",
    "cube_odometry_min_approach_mm",
    "chess_seen_match_mm",
    "center_scan_radius_mm",
    "center_scan_step_degrees",
    "edge_scan_guard_mm",
    "edge_inward_half_angle_degrees",
    "landmark_vantage_distance_mm",
    "cube_map_near_mm",
    "cube_confirm_far_mm",
    "cube_map_far_mm",
    "cube_candidate_projection_mm",
    "cube_emergency_box_height_ratio",
    "cube_face_vertical_overlap",
    "cube_face_gap_ratio",
    "cube_face_size_ratio",
    "cube_face_combined_width_ratio",
    "cube_aspect_min",
    "cube_aspect_max",
    "cube_safety_margin_mm",
    "view_near_mm",
    "view_far_mm",
    "view_fov_degrees",
    "cube_search_patrol_start_ratio",
    "inspection_waypoint_spacing_mm",
    "inspection_reached_mm",
    "inspection_scan_step_degrees",
    "transient_cube_obstacle_enabled",
    "head_angle",
    "drive_speed_mm_s",
    "turn_speed_mm_s",
    "motion_pulse_seconds",
    "motion_step_mm",
    "turn_tolerance_degrees",
    "observation_frames",
    "observation_settle_seconds",
    "exploration_revision",
    "pheromone_deposit",
    "pheromone_evaporation_rate",
    "pheromone_spread_cells",
    "pheromone_route_weight",
    "trail_route_weight",
    "exploration_waypoint_spacing_mm",
    "exploration_min_move_mm",
    "trail_coverage_radius_mm",
    "return_home_trail_ratio",
    "exploration_stall_timeout_seconds",
    "return_home_coverage_ratio",
    "return_home_timeout_seconds",
    "return_home_position_tolerance_mm",
    "return_home_angle_tolerance_degrees",
)

# 有限の正数である必要がある設定項目。
POSITIVE_NUMBER_FIELDS = (
    "cube_hold_warning_seconds", "landmark_view_min_mm", "landmark_view_max_mm",
    "landmark_vantage_separation_mm", "landmark_occlusion_margin_mm",
    "cell_mm",
    "boundary_margin_mm",
    "cube_merge_mm",
    "cube_width_mm",
    "cube_focal_px",
    "cube_distance_scale",
    "cube_candidate_match_mm",
    "cube_lost_grace_seconds",
    "cube_lost_forward_mm",
    "cube_approach_step_mm",
    "cube_approach_stop_mm",
    "cube_bearing_match_degrees",
    "cube_bearing_match_origin_mm",
    "cube_odometry_min_approach_mm",
    "chess_seen_match_mm",
    "center_scan_radius_mm",
    "center_scan_step_degrees",
    "edge_scan_guard_mm",
    "edge_inward_half_angle_degrees",
    "landmark_vantage_distance_mm",
    "cube_map_near_mm",
    "cube_confirm_far_mm",
    "cube_map_far_mm",
    "cube_candidate_projection_mm",
    "cube_viewpoint_baseline_mm",
    "cube_triangulation_min_angle_degrees",
    "cube_viewpoint_preferred_angle_degrees",
    "cube_aspect_min",
    "cube_aspect_max",
    "cube_safety_margin_mm",
    "robot_length_mm",
    "robot_width_mm",
    "initial_pose_adjust_mm",
    "initial_pose_adjust_degrees",
    "inspection_waypoint_spacing_mm",
    "inspection_reached_mm",
    "inspection_scan_step_degrees",
    "pheromone_deposit",
    "pheromone_route_weight",
    "trail_route_weight",
    "exploration_waypoint_spacing_mm",
    "exploration_min_move_mm",
    "trail_coverage_radius_mm",
    "exploration_stall_timeout_seconds",
    "drive_speed_mm_s",
    "turn_speed_mm_s",
    "motion_step_mm",
    "turn_tolerance_degrees",
    "observation_settle_seconds",
    "return_home_timeout_seconds",
    "return_home_position_tolerance_mm",
    "return_home_angle_tolerance_degrees",
)

# キーボード入力と内部モード名の対応。
MODE_CHOICES = {
    "1": "setup",
    "2": "run",
    "3": "new-trial",
    "4": "bluetooth-test",
    "5": "view-results",
}


# ----------------------------------------------------------------------
# 設定値の確認
# ----------------------------------------------------------------------
def validate_config(config):
    """config.py の設定値に明らかな間違いがないか確認する。"""

    # 自分の robot_id が参加機体一覧に存在するか。
    if config.robot_id not in config.robot_markers:
        raise ValueError("robot_idがrobot_markersにありません")

    # 情報共有モードは 3 種類だけを許可する。
    if config.sharing_mode not in ("none", "cubes", "full"):
        raise ValueError("sharing_modeはnone/cubes/full")

    # ArUco の DICT_4X4_50 を使うため、ID は 0～49 の整数で重複不可。
    markers = list(config.robot_markers.values())
    marker_is_invalid = (
        len(set(markers)) != len(markers)
        or any(type(marker) != int or not 0 <= marker < 50 for marker in markers)
    )
    if marker_is_invalid:
        raise ValueError("DICT_4X4_50のマーカーIDを重複なく設定してください")

    # 走行命令を長く出し続けすぎないようにする。
    # 実機で確実に動き始めるよう、現在は0.45秒程度を標準とする。
    if not 0 < config.motion_pulse_seconds <= 1.0:
        raise ValueError("motion_pulse_secondsは0超〜1.0秒")
    if not 0 < config.motion_step_mm <= 50:
        raise ValueError("motion_step_mmは0超〜50mm以下")
    if not 1 <= config.turn_tolerance_degrees <= 7:
        raise ValueError("turn_tolerance_degreesは1〜7度")
    if type(config.observation_frames) is not int or not 1 <= config.observation_frames <= 3:
        raise ValueError("observation_framesは1〜3枚")

    # カメラの観測距離は near < far である必要がある。
    if not 0 < config.view_near_mm < config.view_far_mm:
        raise ValueError("観測距離の大小関係が不正です")

    # 視野角の異常値を防ぐ。
    if not 0 < config.view_fov_degrees < 120:
        raise ValueError("視野角は0超〜120度未満")

    # キューブ登録距離・縦横比・確認フレーム数。
    if not 0 < config.cube_map_near_mm < config.cube_confirm_far_mm < config.cube_map_far_mm:
        raise ValueError("cube_map_near_mm < cube_confirm_far_mm < cube_map_far_mm にしてください")
    if not 0 < config.cube_candidate_projection_mm <= config.cube_confirm_far_mm:
        raise ValueError("cube_candidate_projection_mmはcube_confirm_far_mm以下にしてください")
    if not 0 < config.cube_aspect_min < config.cube_aspect_max:
        raise ValueError("cube_aspect_min / cube_aspect_max の大小関係が不正です")
    if type(config.cube_confirm_frames) is not int or not 1 <= config.cube_confirm_frames <= 3:
        raise ValueError("cube_confirm_framesは1〜3の整数にしてください")
    if config.cube_detection_region not in ('whole_cube', 'marker'):
        raise ValueError('cube_detection_regionはwhole_cubeまたはmarkerにしてください')
    if type(config.cube_confirm_viewpoints) is not int or not 1 <= config.cube_confirm_viewpoints <= 4:
        raise ValueError("cube_confirm_viewpointsは1〜4の整数にしてください")
    for name in ('cube_duplicate_iou', 'cube_stopped_match_iou'):
        value = getattr(config, name)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(name + 'は0超1未満にしてください')
    if type(config.cube_stop_missing_frames) is not int or config.cube_stop_missing_frames < 2:
        raise ValueError('cube_stop_missing_framesは2以上の整数にしてください')
    if config.landmark_view_min_mm >= config.landmark_view_max_mm:
        raise ValueError('チェス観測距離はmin < maxにしてください')
    for name in ('cube_emergency_box_height_ratio', 'cube_approach_stop_box_height_ratio',
                 'cube_face_vertical_overlap', 'cube_face_gap_ratio'):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(name + 'は0超1未満にしてください')
    if not 1 <= config.cube_face_size_ratio <= 4:
        raise ValueError('cube_face_size_ratioは1〜4にしてください')
    if not 1.2 <= config.cube_face_combined_width_ratio <= 3.0:
        raise ValueError('cube_face_combined_width_ratioは1.2〜3.0にしてください')

    # キューブ不足時に巡回確認へ移る探索率。
    if not 0 <= config.cube_search_patrol_start_ratio <= 1:
        raise ValueError("cube_search_patrol_start_ratioは0〜1にしてください")

    # 探索終了として帰還を始める探索率。
    if not 0 < config.cube_candidate_confidence <= config.cube_review_confirm_confidence <= config.cube_confidence <= 1:
        raise ValueError('キューブ信頼度は候補 <= 再確認 <= 高信頼度、かつ0～1に設定してください')
    if type(config.cube_review_confirm_frames) is not int or config.cube_review_confirm_frames < 2:
        raise ValueError('cube_review_confirm_framesは2以上の整数にしてください')
    for name in ('cube_review_distance_mm', 'cube_review_timeout_seconds', 'cube_review_retry_seconds'):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
            raise ValueError(name + 'は有限の正数にしてください')

    if not 0 < config.return_home_coverage_ratio <= 1:
        raise ValueError("return_home_coverage_ratioは0超〜1以下")
    if not 0 < config.return_home_trail_ratio <= 1:
        raise ValueError("return_home_trail_ratioは0超〜1以下")
    if not 0 <= config.pheromone_evaporation_rate < 1:
        raise ValueError("pheromone_evaporation_rateは0以上1未満")
    if type(config.pheromone_spread_cells) is not int or not 0 <= config.pheromone_spread_cells <= 3:
        raise ValueError("pheromone_spread_cellsは0〜3の整数")

    for name in ('cube_track_max_match_mm', 'cube_track_ambiguity_mm', 'cube_candidate_ttl_seconds',
                 'chess_match_mm', 'chess_ambiguity_mm', 'chess_min_baseline_mm',
                 'chess_pair_tolerance_mm', 'chess_residual_mm', 'chess_max_shift_mm',
                 'chess_max_turn_degrees', 'localization_max_age_seconds',
                 'localization_max_travel_mm', 'localization_scan_timeout_seconds'):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(name + 'は有限の正数にしてください')
    if not 0 < config.chess_confidence <= 1:
        raise ValueError('chess_confidenceは0超〜1にしてください')
    if type(config.chess_confirm_frames) is not int or not 2 <= config.chess_confirm_frames <= 3:
        raise ValueError('chess_confirm_framesは2〜3にしてください')
    if type(config.required_chess_landmarks) is not int or config.required_chess_landmarks != 6:
        raise ValueError('この研究ではrequired_chess_landmarksを6にしてください')
    if type(config.chess_seen_confirm_frames) is not int or not 1 <= config.chess_seen_confirm_frames <= 5:
        raise ValueError('chess_seen_confirm_framesは1〜5の整数にしてください')

    if type(config.chess_refine_enabled) is not bool:
        raise ValueError('chess_refine_enabledはboolにしてください')
    for name in ('chess_refine_min_samples', 'chess_refine_window_frames'):
        if type(getattr(config, name)) is not int or not 4 <= getattr(config, name) <= 60:
            raise ValueError(name + 'は4〜60の整数にしてください')
    if config.chess_refine_window_frames < config.chess_refine_min_samples:
        raise ValueError('平均化の窓は必要標本数以上にしてください')
    for name in SHARED_FIELDS:
        if name.startswith('chess_refine_') and name not in (
                'chess_refine_enabled', 'chess_refine_min_samples', 'chess_refine_window_frames'):
            value = getattr(config, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + 'は有限の正数にしてください')
    if not (config.chess_refine_min_shift_mm < config.chess_refine_max_shift_mm
            and config.chess_refine_spread_mm <= config.chess_refine_outlier_mm
            and config.chess_refine_min_span_seconds < config.chess_refine_window_seconds
            and config.chess_refine_min_span_seconds <=
                (config.chess_refine_window_frames - 1) * config.chess_refine_sample_interval_seconds):
        raise ValueError('チェス補正の閾値・時間窓の大小関係を確認してください')

    # 時間・個数として不正な値を防ぐ。
    if (
        config.placement_wait_seconds < 0
        or not math.isfinite(config.trial_limit_seconds)
        or config.trial_limit_seconds < 0
        or config.expected_cubes < 0
    ):
        raise ValueError("待機時間・試行時間・荷物数を確認してください")

    # 距離・速度などは「有限な正の数」である必要がある。
    for name in POSITIVE_NUMBER_FIELDS:
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + "には有限の正数が必要です")


def validate_ports(config, setup):
    """Bluetooth 通信に必要な COM ポート設定を確認する。"""

    # setup.json に参加する機体のうち、自分以外を通信相手とする。
    peers = set(setup["robots"]) - {config.robot_id}

    # 情報共有をしない場合、または相手機体がいない場合は確認不要。
    if config.sharing_mode == "none" or not peers:
        return

    # 相手機体すべてに対して COM ポートが指定されている必要がある。
    if set(config.bluetooth_ports) != peers:
        raise ValueError("bluetooth_portsは自分以外の参加機体をすべて指定してください")

    ports = list(config.bluetooth_ports.values())

    # 未設定・空文字・重複した COM ポートは使用しない。
    has_invalid_port = any(
        not isinstance(port, str) or port in ("", "CHANGE_ME")
        for port in ports
    )
    has_duplicate_port = len(set(ports)) != len(ports)

    if has_invalid_port or has_duplicate_port:
        raise ValueError("bluetooth_portsを実際のBluetooth COMポートへ変更してください")


def validate_bluetooth_runtime(config, setup):
    """Bluetooth通信に必要なPythonライブラリが使えるか確認する。

    結果フォルダやCozmo接続を始める前に確認することで、
    依存ライブラリ不足だけで途中の実験データが残ることを防ぎます。
    """
    peers = set(setup["robots"]) - {config.robot_id}

    # 情報共有なし、または通信相手がいない場合は pyserial を使わない。
    if config.sharing_mode == "none" or not peers:
        return

    from telecommunication.bluetooth_communication import require_pyserial

    require_pyserial()


# ----------------------------------------------------------------------
# setup.json の作成 / 読み込み
# ----------------------------------------------------------------------
def make_setup(slam, poses, config):
    """SLAM の地図と初期姿勢から setup.json 用データを作る。"""

    # JSON に保存しやすいよう、ランドマーク ID を文字列へ変換する。
    landmarks = {
        str(marker_id): [float(x), float(y)]
        for marker_id, (x, y) in slam.true_landmarks.items()
    }

    return dict(
        schema_version=1,
        units="mm",
        angle_unit="rad",
        axes="x-forward,y-left,theta-CCW",
        map_id=fingerprint(landmarks),
        trial_id=uuid.uuid4().hex,
        landmarks=landmarks,
        robots=poses,
        cell_mm=config.cell_mm,
        margin_mm=config.boundary_margin_mm,
        cube_merge_mm=config.cube_merge_mm,
        conditions={name: getattr(config, name) for name in SHARED_FIELDS},
    )


def load_setup(config):
    """setup.json を読み込み、内容を検証して config に反映する。"""

    data = json.loads(config.setup_path.read_text(encoding="utf-8"))

    if data.get('initial_poses_require_review'):
        raise ValueError('実測地図を更新したため、先にメニュー1で初期姿勢を設定し直してください')

    # setup.json の基本形式を確認する。
    if (
        data.get("schema_version") != 1
        or data.get("units") != "mm"
        or data.get("angle_unit") != "rad"
    ):
        raise ValueError("setup.jsonの形式が不正です")

    # 保存された地図 ID と、実際のランドマーク座標が一致するか確認する。
    if data["map_id"] != fingerprint(data["landmarks"]):
        raise ValueError("setup.jsonの地図IDと座標が一致しません")

    # trial_id は uuid.uuid4().hex と同じ 32 桁の 16 進文字列を想定する。
    trial_id = data["trial_id"]
    trial_id_is_invalid = (
        not isinstance(trial_id, str)
        or len(trial_id) != 32
        or not all(char in "0123456789abcdef" for char in trial_id)
    )
    if trial_id_is_invalid:
        raise ValueError("試行IDが不正です")

    # この研究段階ではチェス駒 6 個の地図を前提としている。
    if len(data["landmarks"]) != 6:
        raise ValueError("チェス6個の共通地図が必要です")

    # setup.json の参加機体と config.py の機体一覧を一致させる。
    if set(data["robots"]) != set(config.robot_markers):
        raise ValueError("参加機体一覧とrobot_markersが一致しません")

    # 地図情報から走行可能範囲を作る。
    grid = Grid(
        data["landmarks"].values(),
        data["cell_mm"],
        data["margin_mm"],
    )

    # 各 Cozmo の初期位置・向き・マーカー ID を確認する。
    for robot_id, pose in data["robots"].items():
        marker_is_wrong = pose["marker_id"] != config.robot_markers[robot_id]
        pose_is_invalid = any(
            not math.isfinite(pose[key])
            for key in ("x", "y", "theta")
        )

        if marker_is_wrong or pose_is_invalid:
            raise ValueError("初期姿勢またはマーカーIDが不正です")

        # 初期配置はチェス駒で囲まれた地図内なら許可する。
        # 自律走行時の boundary_margin_mm は探索経路側で引き続き使用する。
        if not grid.inside((pose["x"], pose["y"])):
            raise ValueError("初期位置が地図範囲外です")

    # setup.json に保存された共通実験条件を、この PC の config へ反映する。
    # 新しく追加した帰還設定が古い setup.json に無い場合は、config.py の
    # 現在値を使うことで地図・初期姿勢を作り直さず利用できる。
    upgrade_exploration_conditions(data, config)
    for name in SHARED_FIELDS:
        if name not in data["conditions"]:
            continue

        # cube_width_mm=44 は旧版の仮値。実物は50mmと確定したため、
        # 古い setup.json の値で現在の正しい実寸を上書きしない。
        if name == "cube_width_mm" and data["conditions"][name] != config.cube_width_mm:
            print(
                f"旧setup.jsonのcube_width_mm={data['conditions'][name]}mmは使用せず、"
                f"実物サイズ {config.cube_width_mm}mm を使用します。"
            )
            continue

        # 旧版では同一キューブ判定が60mmで、50mm角のキューブを隣接して
        # 置くと別物まで統合する可能性があった。cube_merge_mmは地図形状そのもの
        # には影響しないため、古いsetupでも現在の45mm設定を優先する。
        if name == "cube_merge_mm" and data["conditions"][name] != config.cube_merge_mm:
            print(
                f"旧setup.jsonのcube_merge_mm={data['conditions'][name]}mmは使用せず、"
                f"現在値 {config.cube_merge_mm}mm を使用します。"
            )
            continue

        # 表示は即時、確定は複数画像で行う。旧版の1画像確定へ戻さない。
        if name == "cube_confirm_frames" and data["conditions"][name] != config.cube_confirm_frames:
            print(
                f"旧setup.jsonのcube_confirm_frames={data['conditions'][name]}は使用せず、"
                f"現在値 {config.cube_confirm_frames} を使用します。"
            )
            continue

        setattr(config, name, data["conditions"][name])

    # World は top-level の cube_merge_mm を参照するため、実行時データも
    # 現在の設定値へ合わせる。setup.jsonファイル自体は書き換えない。
    data["cube_merge_mm"] = config.cube_merge_mm

    # 地図作成に使用する重要条件が一致しているか念のため確認する。
    map_conditions = (
        data["cell_mm"],
        data["margin_mm"],
        data["cube_merge_mm"],
    )
    config_conditions = (
        config.cell_mm,
        config.boundary_margin_mm,
        config.cube_merge_mm,
    )
    if map_conditions != config_conditions:
        raise ValueError("setup.json内の地図条件が矛盾しています")

    # setup.json の値を反映した後、もう一度設定全体を確認する。
    validate_config(config)

    # 実際に使用した条件を実験記録と通信ハッシュへ反映する。
    data['conditions'] = {name: getattr(config, name) for name in SHARED_FIELDS}

    return data


# ----------------------------------------------------------------------
# 実行モードに関する補助処理
# ----------------------------------------------------------------------
def _choose_mode(command_line_mode):
    """コマンドライン指定、またはキーボード入力から実行モードを決める。"""

    if command_line_mode is not None:
        return command_line_mode

    choice = input(
        "1:地図・初期姿勢の設定 / 2:探索実験 / "
        "3:次の試行ID / 4:Bluetoothテスト / "
        "5:検知結果・マップ確認 > "
    ).strip()

    return MODE_CHOICES.get(choice)


def _choose_experiment_robot_count(setup):
    """探索実験で使用する Cozmo の台数を選ぶ。

    setup.json 自体は 2 台分の初期位置を保持したままにし、
    実験を開始するときだけ 1 台 / 2 台を切り替える。
    これにより、同じ地図・同じ初期位置条件で比較しやすくする。
    """
    available = len(setup["robots"])
    if available < 2:
        # setup.json に 1 台しか登録されていない場合は自動的に 1 台探索。
        return 1

    while True:
        choice = input(
            "探索するCozmoの台数を選択してください。\n"
            "1: 1台探索（Bluetooth通信なし）\n"
            "2: 2台協調探索（Bluetooth通信あり） > "
        ).strip()

        if choice in ("1", "2"):
            return int(choice)

        print("1 または 2 を入力してください。")


def _make_experiment_setup(setup, config, robot_count):
    """実験時だけ使用する参加機体一覧を作る。

    元の setup.json は変更しない。1台探索では、このPCに接続している
    ``config.robot_id`` だけを参加機体として扱うため、Bluetooth待機や
    相手機体との情報共有は発生しない。
    """
    if config.robot_id not in setup["robots"]:
        raise ValueError("このPCのrobot_idがsetup.jsonにありません")

    if robot_count == 1:
        active_ids = [config.robot_id]
    elif robot_count == 2:
        if len(setup["robots"]) != 2:
            raise ValueError("2台探索にはsetup.jsonへ2台のCozmoを登録してください")
        active_ids = list(setup["robots"])
    else:
        raise ValueError("探索台数は1台または2台です")

    runtime_setup = dict(setup)
    runtime_setup["robots"] = {
        robot_id: setup["robots"][robot_id]
        for robot_id in active_ids
    }
    runtime_setup["experiment_robot_count"] = robot_count
    runtime_setup["active_robots"] = active_ids
    return runtime_setup


def _run_bluetooth_test(config):
    """Bluetooth の接続確認ツールを起動する。"""

    from tools.bluetooth_test import run

    run(config)


def _create_new_trial(config):
    """前回の初期姿勢を再利用して、新しい試行 ID を作る。"""

    previous = json.loads(config.setup_path.read_text(encoding="utf-8"))

    # 台数が変わった場合は、以前の初期姿勢をそのまま使わない。
    if set(previous["robots"]) != set(config.robot_markers):
        raise ValueError("台数変更時はsetupモードで初期姿勢を設定し直してください")

    previous["trial_id"] = uuid.uuid4().hex
    previous["conditions"] = {
        name: getattr(config, name)
        for name in SHARED_FIELDS
    }
    previous.update(
        cell_mm=config.cell_mm,
        margin_mm=config.boundary_margin_mm,
        cube_merge_mm=config.cube_merge_mm,
    )

    save_json(config.setup_path, previous)
    print("新しいsetup.jsonを全PCへコピーしてください。前回の探索履歴は読み込みません。")


# ----------------------------------------------------------------------
# Cozmo 接続後の setup / run 処理
# ----------------------------------------------------------------------
def _run_robot_mode(mode, setup, config):
    """Cozmo に接続し、setup または run モードを実行する。"""

    # 起動に時間のかかるモジュールは、実際に必要なときだけ読み込む。
    from camera.camera_movie import CameraSettings, CameraViewer
    from connect.network_check import cozmo_session
    from hardware.battery import BatteryMonitor
    from algorithm.mapping.visual_slam import CozmoSlamController
    from research.safety import SafetyClient

    # Cozmo の Wi-Fi に接続し、終了時には元の Wi-Fi へ戻す。
    with cozmo_session(config.cozmo_ssid, config.return_ssid) as raw:
        raw.conn_lock = threading.RLock()

        battery = BatteryMonitor(raw)
        camera = CameraViewer(
            raw,
            battery_monitor=battery,
            enhance_image=True,
            camera_settings=CameraSettings(
                exposure_ms=5,
                gain=1.0,
                auto_exposure=False,
            ),
        )

        cli = None

        try:
            battery.start()
            camera.start(background_display=False)

            # SafetyClient を通して Cozmo を操作する。
            cli = SafetyClient(raw, camera, config)

            slam = CozmoSlamController(
                model_path=config.chess_model_path,
                camera_viewer=camera,
                json_filename=str(config.map_path),
            )

            if mode == "setup":
                _run_setup_mode(cli, camera, slam, config)
            else:
                _run_experiment_mode(cli, camera, slam, setup, config, battery)

        finally:
            # 途中で例外が起きても、各機能を必ず安全に停止する。
            try:
                if cli is not None:
                    cli.stop()
            finally:
                try:
                    camera.stop()
                finally:
                    battery.stop()


def _run_setup_mode(cli, camera, slam, config):
    """チェス地図を作成し、各 Cozmo の初期姿勢を setup.json に保存する。"""

    # この画面は setup モードでしか使わないため、ここで読み込む。
    # 関数内で import した名前は、その関数の外からは使えない点に注意。
    from research.interface import MapWindow, choose_initial_poses

    # mapping_only=True なので、ここでは地図作成だけを行う。
    if not slam.run(cli, mapping_only=True):
        return

    window = MapWindow(
        slam.true_landmarks,
        robot_length_mm=config.robot_length_mm,
        robot_width_mm=config.robot_width_mm,
    )

    try:
        grid = Grid(
            slam.true_landmarks.values(),
            config.cell_mm,
            config.boundary_margin_mm,
        )

        poses = choose_initial_poses(
            window,
            grid,
            config.robot_markers,
            camera,
            adjust_step_mm=config.initial_pose_adjust_mm,
            adjust_angle_degrees=config.initial_pose_adjust_degrees,
        )

        if poses:
            setup_data = make_setup(slam, poses, config)
            save_json(config.setup_path, setup_data)
            print(
                "設定完了。setup.jsonとslam_data.jsonを全PCへ配布して"
                "runモードで開始します。"
            )

    finally:
        window.close()


def _run_experiment_mode(cli, camera, slam, setup, config, battery):
    """配布済み setup.json を基準にして探索実験を開始する。"""

    # ResearchSession は run モードでしか使わないため、ここで読み込む。
    from research.session import ResearchSession

    # 地図に埋め込まれたカメラ校正も復元する。座標の正本は下のsetup。
    if config.map_path.exists():
        slam.load_map_data(str(config.map_path))
    if not slam.distance_scales:
        slam._load_distance_scales()
    if not slam.distance_scales:
        raise ValueError('チェスの距離校正がありません。地図作成時のCキーで校正してください')
    if abs(slam.TARGET_HEAD_ANGLE - config.head_angle) > 0.001:
        raise ValueError('探索時の首角度とチェス校正時の角度が違います。head_angleを揃えてください')

    # 探索時の基準地図は全 PC 共通の setup.json。
    # 各 PC の個別 SLAM で別々の原点を作らない。
    slam.true_landmarks = {
        int(marker_id): tuple(position)
        for marker_id, position in setup["landmarks"].items()
    }

    ResearchSession(
        cli,
        camera,
        slam,
        setup,
        config,
        battery,
    ).run()


# ----------------------------------------------------------------------
# main.py から呼ばれる入口
# ----------------------------------------------------------------------
def run_application():
    """アプリケーションを起動する。"""

    parser = argparse.ArgumentParser(
        description="ant-COZMO 第1段階：協調探索"
    )
    parser.add_argument(
        "--mode",
        choices=("setup", "run", "new-trial", "bluetooth-test", "view-results"),
    )
    args = parser.parse_args()

    config = CONFIG

    # まず config.py 単体の設定ミスを確認する。
    validate_config(config)

    # コマンドライン指定がなければ、番号入力でモードを選択する。
    mode = _choose_mode(args.mode)

    # Bluetooth 接続確認だけを行うモード。
    if mode == "bluetooth-test":
        _run_bluetooth_test(config)
        return

    # 保存済み探索結果を、Cozmoへ接続せず確認するモード。
    if mode == "view-results":
        from research.result_viewer import run_result_viewer

        run_result_viewer(config)
        return

    # 以前の地図・初期姿勢を使って新しい試行を作るモード。
    if mode == "new-trial":
        _create_new_trial(config)
        return

    # setup / run 以外が入力された場合は何もせず終了する。
    if mode not in ("setup", "run"):
        return

    # run モードでは、共通の setup.json を読み込む。
    setup = load_setup(config) if mode == "run" else None

    # 探索実験では、同じ setup.json を使ったまま 1台 / 2台を選択する。
    # 1台探索では runtime_setup の参加機体を自分だけにするため、
    # Bluetooth の COM ポート確認・接続待機は行われない。
    if setup:
        robot_count = _choose_experiment_robot_count(setup)
        setup = _make_experiment_setup(setup, config, robot_count)

        print(
            f"探索条件: {robot_count}台 / "
            + ("Bluetoothなし" if robot_count == 1 else "Bluetooth協調通信あり")
        )

        # 実験開始前に Bluetooth と結果保存先を確認する。
        # 結果フォルダの重複は Cozmo Wi-Fi へ切り替える前に処理する。
        validate_ports(config, setup)
        validate_bluetooth_runtime(config, setup)

        from research.session import prepare_output_for_run
        prepare_output_for_run(setup, config)

    # 地図作成は mapper_id に指定した 1 台だけで行う。
    if mode == "setup" and config.robot_id != config.mapper_id:
        raise ValueError("地図作成はmapper_idの1台だけで行ってください")

    _run_robot_mode(mode, setup, config)
