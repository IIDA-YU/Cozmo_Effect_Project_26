"""ant-COZMO アプリケーションの設定。

このファイルには、主に「PC / Cozmo ごとに変更する値」をまとめています。
共通地図や実験条件は setup.json に保存され、各 PC で同じ内容を使用します。

研究時に設定を変更するときは、基本的に ``ApplicationConfig`` の初期値だけを
変更してください。処理そのものは controller.py や research/ 側に置きます。
"""

from dataclasses import dataclass, field
from pathlib import Path


# プロジェクトのルートディレクトリ。
# application/config.py から見て 1 つ上の階層を指す。
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ApplicationConfig:
    """アプリケーション全体で使用する設定値。

    設定を用途ごとに分けて並べています。
    観測・走行の値は今回の修正版の初期値です。
    """

    # ------------------------------------------------------------------
    # 1. Cozmo / PC の識別設定
    # ------------------------------------------------------------------
    robot_id: str = "Cozmo-A"

    # 新しいチェス地図を作成できる機体。
    mapper_id: str = "Cozmo-A"

    # Cozmo 接続用 Wi-Fi と、処理終了後に戻す Wi-Fi。
    cozmo_ssid: str = "Cozmo_613F59"
    return_ssid: str = "ctc-g-171bde"

    # 各 Cozmo に取り付ける ArUco マーカー ID。
    robot_markers: dict = field(
        default_factory=lambda: {
            "Cozmo-A": 1,
            "Cozmo-B": 2,
        }
    )

    # PC ごとに「相手の機体 ID : この PC で開く Bluetooth SPP COM ポート」を設定する。
    # 例:
    #   Cozmo-A 側 -> {"Cozmo-B": "COM5"}
    #   Cozmo-B 側 -> {"Cozmo-A": "COM7"}
    # COM 番号は Windows のデバイスマネージャーなどで実機確認する。
    bluetooth_ports: dict = field(
        default_factory=lambda: {
            "Cozmo-B": "CHANGE_ME",
        }
    )

    # ------------------------------------------------------------------
    # 2. 実験条件
    # ------------------------------------------------------------------
    # none  : 情報共有なし
    # cubes : 荷物情報のみ共有
    # full  : 荷物情報 + 探索フェロモンを共有
    sharing_mode: str = "full"

    # 互換性のため setup.json に残している旧設定。
    # 現在の探索開始は待ち時間を使わず、Enter を押した瞬間に開始する。
    placement_wait_seconds: float = 0.0

    # 0: 時間だけでは終了しない。6駒再確認 + 95%観測まで探索する。
    # 正数を指定した実験は、その秒数で「未完了の打切り」として記録する。
    trial_limit_seconds: float = 0.0

    # マップ内に存在すると分かっているキューブ総数。
    # 0 にすると総数不明として扱う。
    expected_cubes: int = 0

    # ------------------------------------------------------------------
    # 3. 地図 / 座標に関する設定
    # ------------------------------------------------------------------
    # グリッド 1 マスの大きさ [mm]。
    cell_mm: float = 50.0

    # 地図の枠から確保する安全距離 [mm]。
    # Cozmo の半径 + 位置推定誤差を考慮して実機に合わせる。
    boundary_margin_mm: float = 70.0

    # 近い位置で検出されたキューブを同一物としてまとめる距離 [mm]。
    cube_merge_mm: float = 45.0

    # Cozmo 本体のおおよその実寸 [mm]。
    # 初期位置設定画面では、この大きさを地図の縮尺に合わせて表示する。
    robot_length_mm: float = 90.0
    robot_width_mm: float = 60.0

    # 初期姿勢を微調整するときの1回あたりの移動量・回転量。
    initial_pose_adjust_mm: float = 5.0
    initial_pose_adjust_degrees: float = 2.0

    # ------------------------------------------------------------------
    # 4. キューブ検出に関する設定
    # ------------------------------------------------------------------
    # 研究地図ではPower Cubeを50mm角として扱い、回避・描画の寸法を統一する。
    # 学習時のラベルが本体全体を囲むモデルを使用する。
    cube_width_mm: float = 50.0
    cube_detection_region: str = "whole_cube"
    # 全体枠は隣接していても融合しない。大きく重なる重複枠だけ除去する。
    cube_duplicate_iou: float = 0.65
    cube_stopped_match_iou: float = 0.35
    # 新しい停止画像で6枚連続見えない候補は誤検知/見失いとして解除する。
    cube_stop_missing_frames: int = 6
    cube_hold_warning_seconds: float = 8.0

    # 画像幅 320 px を基準にした焦点距離の仮値。
    # 実機で距離を測定して校正する必要がある。
    cube_focal_px: float = 235.0

    # カメラ距離推定の補正倍率。
    cube_distance_scale: float = 1.0

    # 学習モデル内でキューブとして扱うクラス名。
    cube_class_names: tuple = ("cube", "powercube")

    # モデルのラベル名が異なる場合は、起動時の表示を確認して上記を変更する。
    cube_confidence: float = 0.65
    # 弱い検知も候補として描画し、近づいて再確認する。
    cube_candidate_confidence: float = 0.35
    cube_review_confirm_confidence: float = 0.45
    cube_review_confirm_frames: int = 2
    cube_review_distance_mm: float = 170.0
    cube_review_timeout_seconds: float = 24.0
    cube_review_retry_seconds: float = 5.0

    # 同じ停止位置の独立した2画像で確定できる。単発検知では確定しない。
    # 地図への確定登録には、別途、距離と自己位置の整合を要求する。
    cube_confirm_frames: int = 2
    cube_confirm_viewpoints: int = 1
    cube_viewpoint_baseline_mm: float = 50.0
    cube_triangulation_min_angle_degrees: float = 8.0
    cube_viewpoint_preferred_angle_degrees: float = 18.0
    cube_candidate_match_mm: float = 85.0

    # 追跡IDの照合上限。空間的に近い全キューブを一括統合する値ではない。
    cube_track_max_match_mm: float = 100.0
    cube_track_ambiguity_mm: float = 12.0
    # 遠距離候補は一瞬見失っても同じ方向を疑い続けるため長めに保持する。
    cube_candidate_ttl_seconds: float = 24.0

    # キューブを見つけたら通常探索より優先し、短い前進→停止観測を繰り返す。
    cube_lost_grace_seconds: float = 6.0
    cube_lost_forward_mm: float = 75.0
    cube_approach_step_mm: float = 45.0
    cube_approach_stop_mm: float = 120.0
    cube_approach_stop_box_height_ratio: float = 0.35
    cube_bearing_match_degrees: float = 10.0
    cube_bearing_match_origin_mm: float = 180.0
    cube_odometry_min_approach_mm: float = 45.0

    # 探索中のチェス照合。保存地図と実測が整合した場合のみ自己位置を更新する。
    chess_confidence: float = 0.55
    chess_match_mm: float = 100.0
    chess_ambiguity_mm: float = 15.0
    chess_min_baseline_mm: float = 80.0
    chess_pair_tolerance_mm: float = 25.0
    chess_residual_mm: float = 18.0
    chess_max_shift_mm: float = 80.0
    chess_max_turn_degrees: float = 20.0
    chess_confirm_frames: int = 2

    # チェス座標は、対象駒を使わない自己位置からの安定した平均で更新する。
    chess_refine_enabled: bool = True
    chess_refine_min_samples: int = 6
    chess_refine_window_frames: int = 12
    chess_refine_window_seconds: float = 8.0
    chess_refine_sample_interval_seconds: float = 0.06
    chess_refine_min_span_seconds: float = 0.4
    chess_refine_spread_mm: float = 8.0
    chess_refine_outlier_mm: float = 18.0
    chess_refine_min_shift_mm: float = 8.0
    chess_refine_max_shift_mm: float = 120.0  # 初期座標からの累積上限
    chess_refine_cooldown_seconds: float = 10.0
    chess_refine_pose_age_seconds: float = 20.0
    chess_refine_pose_travel_mm: float = 100.0
    chess_refine_far_mm: float = 600.0

    # 探索完了条件として、地図境界を作る6駒を個別に確認する。
    # 各駒は1フレームだけでなく複数フレームで同じ地図座標へ対応した場合に確認済みとする。
    required_chess_landmarks: int = 6
    chess_seen_confirm_frames: int = 2
    chess_seen_match_mm: float = 90.0

    # 中央付近だけはその場全周確認を許可する。淵では内向き/対象駒向きに限定する。
    center_scan_radius_mm: float = 180.0
    center_scan_step_degrees: float = 45.0
    edge_scan_guard_mm: float = 150.0
    edge_inward_half_angle_degrees: float = 85.0
    landmark_vantage_distance_mm: float = 180.0
    landmark_view_min_mm: float = 100.0
    landmark_view_max_mm: float = 400.0
    landmark_vantage_separation_mm: float = 65.0
    landmark_occlusion_margin_mm: float = 10.0
    localization_max_age_seconds: float = 30.0
    localization_max_travel_mm: float = 150.0
    localization_scan_timeout_seconds: float = 45.0

    # 単眼bbox測距は遠距離ほど誤差が急増する。
    # 320mm以内だけを「距離を使って確定可能」とし、それより遠い検出は
    # 方向候補として扱ってCozmo自身が近づいて再観測する。
    cube_map_near_mm: float = 35.0
    cube_confirm_far_mm: float = 320.0
    cube_map_far_mm: float = 700.0
    cube_candidate_projection_mm: float = 300.0

    # 画面を大きく占有する物体は距離式を信用せず、安全側に停止・旋回する。
    cube_emergency_box_height_ratio: float = 0.45

    # 1個のPower Cubeを角から見たとき、隣り合う2面のマーカーが
    # 2検出になる場合を画像上で1候補へまとめる条件。
    cube_face_vertical_overlap: float = 0.42
    cube_face_gap_ratio: float = 0.45
    cube_face_size_ratio: float = 2.2
    cube_face_combined_width_ratio: float = 1.95

    # 斜めから見たPower Cubeの検出枠も許容する縦横比。
    cube_aspect_min: float = 0.40
    cube_aspect_max: float = 2.40

    # キューブ回避時に、Cozmoとキューブの実寸に加えて確保する余裕 [mm]。
    cube_safety_margin_mm: float = 15.0

    # ------------------------------------------------------------------
    # 5. カメラの観測範囲
    # ------------------------------------------------------------------
    view_near_mm: float = 80.0
    view_far_mm: float = 350.0
    view_fov_degrees: float = 50.0

    # 探索率だけが上がってキューブを見落とさないよう、地図の各所へ
    # 実際に近づいて確認する二次探索（巡回確認）の設定。
    # 70%程度探索した時点で、まだexpected_cubesに届いていなければ
    # 180mm間隔程度の観測地点を巡回し、各地点で向きを変えて確認する。
    cube_search_patrol_start_ratio: float = 0.70
    inspection_waypoint_spacing_mm: float = 180.0
    inspection_reached_mm: float = 35.0
    inspection_scan_step_degrees: float = 90.0

    # 検出したキューブの真正面へ進まないための設定。
    # 未確定のキューブも次の走行区間では一時障害物として扱う。
    transient_cube_obstacle_enabled: bool = True

    # 検出距離の検証時と同じ角度に固定する。
    head_angle: float = -0.15

    # Cozmo の基準位置からカメラまでの前方オフセット [mm]。
    camera_forward_mm: float = 20.0

    # ------------------------------------------------------------------
    # 6. 走行制御
    # ------------------------------------------------------------------
    drive_speed_mm_s: float = 45.0

    # 実機では12 mm/s程度の短い旋回命令だと動き出しにくかったため、
    # 旋回は25 mm/sを基準にする。
    turn_speed_mm_s: float = 25.0

    # 1回の走行時間の上限。45mm/s × 0.8秒 = 最大36mmごとに停止観測。
    # 向き・目標地点に到達したら、この時間より早く停止する。
    motion_pulse_seconds: float = 0.8
    motion_step_mm: float = 40.0
    turn_tolerance_degrees: float = 5.0

    # 独立した停止画像を3枚観測し、チェス補正とキューブ確認を行う。
    observation_frames: int = 3
    observation_settle_seconds: float = 0.12

    # この実装の観測・動作条件を記録し、旧setupの初期値だけを移行する。
    # v8: 全体検出・停止確認・遮蔽を避けた6駒再観測。
    exploration_revision: int = 9

    # ------------------------------------------------------------------
    # 6-1. 仮想フェロモン / 面探索
    # ------------------------------------------------------------------
    # 走行地点に付ける仮想フェロモン量。濃い場所ほど再訪コストが高くなる。
    pheromone_deposit: float = 1.0

    # 停止観測1回ごとの蒸発率。0.04なら各観測で4%ずつ薄くなる。
    pheromone_evaporation_rate: float = 0.04

    # 中心セルだけでなく隣接セルにも弱くフェロモンを広げる範囲。
    pheromone_spread_cells: int = 1

    # 経路探索時にフェロモン濃度へ掛ける重み。
    pheromone_route_weight: float = 2.4

    # 既に他機/自機が通ったセルそのものへ追加する再訪ペナルティ。
    trail_route_weight: float = 1.4

    # 物理的に巡回する観測点の間隔。小さすぎると細かく止まりすぎる。
    exploration_waypoint_spacing_mm: float = 120.0

    # 通常探索では、可能ならこの距離以上離れた次観測点を選ぶ。
    # 同じ場所で方向だけ変え続ける挙動を抑える。
    exploration_min_move_mm: float = 90.0

    # 軌跡からこの距離以内の走行可能セルを「物理的に巡回済み」とみなす。
    trail_coverage_radius_mm: float = 85.0

    # 走行率は経路選択・研究評価用。v7では終了条件には使わない。
    # 6駒確認 + 視認率95%を終了条件とする。
    return_home_trail_ratio: float = 0.80

    # 進捗が増えない場合に別経路へ再計画する間隔。終了時間ではない。
    exploration_stall_timeout_seconds: float = 90.0

    # 探索を「十分に完了」とみなす探索率。
    # 0.95 = 地図セルの95%以上。キューブの発見数に関係なく帰還する。
    return_home_coverage_ratio: float = 0.95

    # 探索完了後、初期位置へ戻るために許可する最大時間 [秒]。
    return_home_timeout_seconds: float = 90.0

    # 初期位置へ戻ったと判定する位置・向きの許容誤差。
    return_home_position_tolerance_mm: float = 20.0
    return_home_angle_tolerance_degrees: float = 10.0

    # ------------------------------------------------------------------
    # 7. Cozmo 同士の遭遇 / 通信判定
    # ------------------------------------------------------------------
    encounter_cooldown_seconds: float = 8.0
    min_marker_pixels: float = 12.0
    marker_confirm_frames: int = 2

    # カメラ画像・状態情報を「古い」と判断するまでの時間 [秒]。
    frame_timeout_seconds: float = 1.5
    state_timeout_seconds: float = 2.0

    # ------------------------------------------------------------------
    # 8. バッテリー
    # ------------------------------------------------------------------
    # 電圧から求めた概算値。現在は運搬割り当てには使用していない。
    minimum_battery_percent: float = 8.0

    # ------------------------------------------------------------------
    # 9. 学習済みモデル / JSON ファイルの保存場所
    # ------------------------------------------------------------------
    chess_model_path: Path = (
        ROOT / "algorithm/mapping/learning_data/best_chess_4.0.pt"
    )
    robot_model_path: Path = (
        ROOT / "camera/learning_data/best_cozmo_distinguish.pt"
    )
    cube_model_path: Path = (
        ROOT / "algorithm/mapping/learning_data/best_powercube.pt"
    )

    map_path: Path = ROOT / "slam_data.json"
    setup_path: Path = ROOT / "setup.json"


# 通常はこの CONFIG を application/controller.py から読み込んで使用する。
CONFIG = ApplicationConfig()
