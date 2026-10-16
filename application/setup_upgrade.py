"""旧setup.jsonの標準値だけを移行する。元ファイルは書き換えない。"""


def upgrade_exploration_conditions(data, config):
    conditions = data['conditions']
    revision = conditions.get('exploration_revision', 1)
    if revision < 2:
        # ユーザーが実験用に変えた数値は維持する。旧版の標準値だけが対象。
        old_defaults = {
            'cube_map_near_mm': (60.0,),
            'cube_map_far_mm': (500.0,),
            'drive_speed_mm_s': (30.0,),
            'motion_pulse_seconds': (0.15, 0.45),
        }
        for name, values in old_defaults.items():
            if conditions.get(name) in values:
                old = conditions[name]
                conditions[name] = getattr(config, name)
                print(f'探索条件の更新: {name} {old} → {conditions[name]}')
    if revision < 4:
        # 誤検出を防ぐ新しい確認処理に必要な条件。キューブ個数は上限に使わない。
        conditions['cube_confirm_frames'] = config.cube_confirm_frames
        conditions['observation_frames'] = config.observation_frames
    if revision < 5:
        # v5では遠距離bboxを確定座標として使わない。旧標準1200mmだけを
        # 新しい安全値へ移行し、利用者が意図的に変更した値は維持する。
        if conditions.get('cube_map_far_mm') == 1200.0:
            conditions['cube_map_far_mm'] = config.cube_map_far_mm
        conditions.setdefault('cube_confirm_far_mm', config.cube_confirm_far_mm)
        conditions.setdefault('cube_candidate_projection_mm', config.cube_candidate_projection_mm)
        conditions.setdefault('cube_confirm_viewpoints', config.cube_confirm_viewpoints)
        conditions.setdefault('cube_viewpoint_baseline_mm', config.cube_viewpoint_baseline_mm)
        conditions.setdefault('cube_triangulation_min_angle_degrees', config.cube_triangulation_min_angle_degrees)
        conditions.setdefault('cube_viewpoint_preferred_angle_degrees', config.cube_viewpoint_preferred_angle_degrees)
        conditions.setdefault('cube_emergency_box_height_ratio', config.cube_emergency_box_height_ratio)
        conditions.setdefault('cube_face_vertical_overlap', config.cube_face_vertical_overlap)
        conditions.setdefault('cube_face_gap_ratio', config.cube_face_gap_ratio)
        conditions.setdefault('cube_face_size_ratio', config.cube_face_size_ratio)
    if revision < 6:
        # v6: 仮想フェロモンを実際の経路選択へ統合し、
        # 視認率だけでなく実走行カバー率も探索終了条件にする。
        for name in (
            'pheromone_deposit', 'pheromone_evaporation_rate', 'pheromone_spread_cells',
            'pheromone_route_weight', 'trail_route_weight',
            'exploration_waypoint_spacing_mm', 'exploration_min_move_mm',
            'trail_coverage_radius_mm', 'return_home_trail_ratio',
            'exploration_stall_timeout_seconds',
        ):
            conditions.setdefault(name, getattr(config, name))
    if revision < 7:
        # v7: 6駒を周回確認してから95%面探索で終了する。中央だけ全周旋回を許可し、
        # 遠距離キューブは方向追跡→短い接近→odometry測距へ変更する。
        for name in (
            'cube_lost_grace_seconds', 'cube_lost_forward_mm', 'cube_approach_step_mm',
            'cube_approach_stop_mm', 'cube_approach_stop_box_height_ratio',
            'cube_bearing_match_degrees', 'cube_bearing_match_origin_mm',
            'cube_odometry_min_approach_mm', 'cube_face_combined_width_ratio',
            'required_chess_landmarks', 'chess_seen_confirm_frames', 'chess_seen_match_mm',
            'center_scan_radius_mm', 'center_scan_step_degrees', 'edge_scan_guard_mm',
            'edge_inward_half_angle_degrees', 'landmark_vantage_distance_mm',
        ):
            conditions.setdefault(name, getattr(config, name))
    if revision < 8:
        # モデルの検出領域が変わったため、旧標準の確認条件を移行する。
        for name, old in (
            ('cube_review_confirm_confidence', 0.55),
            ('cube_review_confirm_frames', 3), ('cube_confirm_frames', 3),
            ('cube_confirm_viewpoints', 2), ('trial_limit_seconds', 300.0),
        ):
            if name not in conditions or conditions[name] == old:
                conditions[name] = getattr(config, name)
        for name in (
            'cube_detection_region', 'cube_duplicate_iou', 'cube_stopped_match_iou',
            'cube_stop_missing_frames', 'cube_hold_warning_seconds',
            'landmark_view_min_mm', 'landmark_view_max_mm',
            'landmark_vantage_separation_mm', 'landmark_occlusion_margin_mm',
        ):
            conditions.setdefault(name, getattr(config, name))
    if revision < 9:
        # v9: 旧地図の座標/試行IDはそのまま、平均化の共通条件だけを補う。
        for name in vars(config):
            if name.startswith('chess_refine_'):
                conditions.setdefault(name, getattr(config, name))
    conditions['exploration_revision'] = max(revision, config.exploration_revision)
