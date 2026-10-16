"""駒の安定補正、独立姿勢、実観測ループ、保存・表示・経路の回帰テスト。"""
import copy
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from application.config import ApplicationConfig
from application.controller import load_setup, validate_config
from research.exchange import EncounterExchange
from research.interface import MapWindow
from research.landmark_refinement import LandmarkRefiner
from research.landmark_search import LandmarkViewpoints
from research.localization import LandmarkLocalizer, transform
from research.planner import Explorer
from research.result_viewer import ResultEntry, _load_world
from research.session import ResearchSession
from research.world import World, fingerprint, save_json
from test_research import setup
from test_session import Clock, Camera, Window, Network, Perception


def observations(points):
    return [dict(local_x=x, local_y=y) for x, y in points]


class AveragingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.refiner = LandmarkRefiner({'0': (300, 100)}, self.cfg)

    def sample(self, points, start=0, current=(300, 100)):
        updates = []
        for i, point in enumerate(points):
            proposal, state = self.refiner.offer('0', point, current, start + i * .1, {'1', '2'})
            if proposal:
                updates.append(proposal)
                self.refiner.commit([proposal])
                current = proposal['after']
        return updates

    def test_six_samples_commit_mean_once_without_per_frame_drift(self):
        points = [(340 + dx, 120) for dx in (-2, 2, -1, 1, -3, 3)]
        self.assertFalse(self.sample(points[:5]))
        proposal, _ = self.refiner.offer('0', points[5], (300, 100), .5, {'1', '2'})
        self.assertEqual(proposal['after'], [340, 120])
        self.assertEqual(proposal['samples'], 6)

    def test_one_outlier_does_not_pull_the_mean(self):
        updates = self.sample([(390, 100)] + [(340, 120)] * 6)
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0]['after'], [340, 120])
        self.assertEqual(updates[0]['samples'], 6)

    def test_wide_scatter_is_never_committed(self):
        self.assertFalse(self.sample([(340 + (-15 if i % 2 else 15), 120) for i in range(30)]))

    def test_tiny_jitter_keeps_current_position(self):
        self.assertFalse(self.sample([(300 + (-2 if i % 2 else 2), 103) for i in range(30)]))

    def test_repeated_timestamp_cannot_supply_six_images(self):
        for _ in range(10):
            proposal, state = self.refiner.offer('0', (340, 120), (300, 100), 1, {'1', '2'})
        self.assertIsNone(proposal)
        self.assertEqual(state['samples'], 1)

    def test_old_samples_expire_and_cooldown_prevents_repeated_changes(self):
        self.sample([(340, 120)] * 5)
        self.assertFalse(self.sample([(340, 120)], start=10))
        updates = self.sample([(340, 120)] * 6, start=11)
        self.assertEqual(len(updates), 1)
        self.assertFalse(self.sample([(365, 120)] * 12, start=12, current=(340, 120)))

    def test_cumulative_shift_is_limited_against_original(self):
        self.assertFalse(self.sample([(430, 100)] * 10, current=(390, 100)))


class IndependentPoseTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.points = [(200, -200), (400, -100), (220, 220), (440, 180)]
        self.loc = LandmarkLocalizer({str(i): p for i, p in enumerate(self.points)}, self.cfg)

    def reference(self, ids=(0, 1)):
        for now in (0, .1):
            self.loc.update(observations([self.points[i] for i in ids]), (0, 0, 0), now, odometry=(0, 0, 0))

    def test_corrects_one_wrong_piece_with_three_unchanged_anchors(self):
        original = dict(self.loc.landmarks)
        points = self.points[:3] + [(480, 200)]
        for i in range(12):
            _, report = self.loc.update(observations(points), (0, 0, 0), i * .1)
            if i < 6:
                self.assertEqual(self.loc.landmarks, original)
        self.assertEqual(self.loc.landmarks['3'], (480, 200))
        self.assertEqual(len(self.loc.refiner.history), 1)
        self.assertEqual(self.loc.refiner.history[0]['anchor_ids'], ['0', '1', '2'])
        self.assertTrue(report['corrected'])

    def test_single_piece_can_refine_after_separate_other_anchor_view(self):
        self.reference()
        # 現在の補正済みposeに誤差があっても、独立raw基準から平均する。
        for i in range(6):
            _, report = self.loc.update(observations([(470, 200)]), (60, 0, 0), 1 + i * .1,
                                        travelled=10, odometry=(10, 0, 0))
        self.assertEqual(self.loc.landmarks['3'], (480, 200))
        self.assertFalse(report['corrected'])  # 1駒で新しい全姿勢を確定しない
        self.assertIn('3', self.loc.seen_ids)

    def test_recovery_matching_can_find_shift_beyond_old_100mm_gate(self):
        self.reference()
        for i in range(6):
            self.loc.update(observations([(550, 180)]), (0, 0, 0), 1 + i * .1)
        self.assertEqual(self.loc.landmarks['3'], (550, 180))

    def test_single_piece_without_reference_cannot_rewrite_map(self):
        for i in range(30):
            self.loc.update(observations([(480, 200)]), (0, 0, 0), i * .1)
        self.assertFalse(self.loc.refiner.history)
        self.assertEqual(self.loc.landmarks['3'], self.points[3])

    def test_target_cannot_use_a_reference_that_depends_on_itself(self):
        self.reference((0, 3))
        for i in range(10):
            self.loc.update(observations([(480, 200)]), (0, 0, 0), 1 + i * .1)
        self.assertFalse(self.loc.refiner.history)

    def test_leave_one_out_reference_survives_separate_view(self):
        self.reference((0, 1, 3))
        for i in range(6):
            self.loc.update(observations([(480, 200)]), (0, 0, 0), 1 + i * .1)
        self.assertEqual(self.loc.landmarks['3'], (480, 200))
        self.assertFalse(self.loc.trusted(2, 0))  # 旧3を使った最後の姿勢は無効

    def test_reference_expires_by_time_or_raw_travel(self):
        for expired_time, travelled in ((21, 0), (1, 101)):
            self.setUp()
            self.reference()
            for i in range(10):
                self.loc.update(observations([(480, 200)]), (0, 0, 0), expired_time + i * .1, travelled)
            self.assertFalse(self.loc.refiner.history)

    def test_rigid_robot_pose_error_does_not_move_all_landmarks(self):
        truth = (10, -20, .06)
        true_points = {str(i): transform(p, truth) for i, p in enumerate(self.points)}
        loc = LandmarkLocalizer(true_points, self.cfg)
        for i in range(12):
            pose, _ = loc.update(observations(self.points), (0, 0, 0), i * .1)
        self.assertFalse(loc.refiner.history)
        self.assertEqual(loc.landmarks, true_points)
        for a, b in zip(pose, truth):
            self.assertAlmostEqual(a, b)

    def test_ambiguous_duplicate_boxes_never_move_a_piece(self):
        self.reference()
        for i in range(12):
            self.loc.update(observations([(480, 200), (481, 200)]), (0, 0, 0), 1 + i * .1)
        self.assertFalse(self.loc.refiner.history)

    def test_pending_target_does_not_move_pose_before_its_mean_is_ready(self):
        self.reference()
        # 0/3の間隔差は旧許容内。別の基準(0/1)から3のずれを検出する。
        pose, report = self.loc.update(observations([self.points[0], (450, 190)]), (0, 0, 0), 1)
        pose, report = self.loc.update(observations([self.points[0], (450, 190)]), pose, 1.1)
        self.assertTrue(report['refinement_pending'])
        self.assertFalse(report['corrected'])
        self.assertEqual(pose, (0, 0, 0))


class IntegrationTests(unittest.TestCase):
    def test_stopped_session_gathers_six_then_updates_live_map_and_saved_result(self):
        data = setup()
        # この標本は機体・YOLOを模擬し、sessionの新画像/停止処理は実際に実行する。
        data['landmarks'] = {'0': (0, 0), '1': (250, 0), '2': (500, 0),
                             '3': (500, 500), '4': (250, 500), '5': (0, 500)}
        truth = (250, 250, 0)
        state = [truth]
        raw = SimpleNamespace(position=SimpleNamespace(x=0, y=0),
                              rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0)))
        commands = []
        cli = SimpleNamespace(pose=raw, latched_reason='', drive_wheels=lambda *args: commands.append(args))
        clock, frame = Clock(), np.zeros((240, 320, 3), dtype=np.uint8)
        camera = Camera(clock)
        sample = camera.get_frame_sample
        def new_frame():
            _, number, stamp = sample()
            return frame, number, stamp
        camera.get_frame_sample = new_frame
        points = [(0, 0), (500, 0), (0, 500), (460, 480)]
        detections = [dict(confidence=.9, box=(10, 10, 20, 30), point=p) for p in points]
        slam = SimpleNamespace(_get_estimated_pose=lambda *a: state[0],
            _set_estimated_pose=lambda *a: state.__setitem__(0, tuple(a[-3:])),
            detect_chess_pieces=lambda f: detections,
            _estimate_landmark_observation=lambda d, w, *p: dict(local_x=d['point'][0]-250, local_y=d['point'][1]-250))
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', sharing_mode='none')
            with patch('research.session.time', clock), patch('research.session.ResearchPerception', Perception), \
                 patch('research.session.MapWindow', Window), patch('research.session.BluetoothNetwork', Network):
                s = ResearchSession(cli, camera, slam, data, cfg, SimpleNamespace(percent=80))
                try:
                    s.started = clock.now
                    original_hash = fingerprint(data)
                    grid_before = dict(s.world.grid.cells)
                    s.observe_stopped()
                    self.assertGreaterEqual(camera.count, 7)
                    self.assertTrue(all(command == (0, 0) for command in commands))
                    self.assertEqual(s.world.landmarks['3'], (460, 480))
                    self.assertEqual(grid_before, s.world.grid.cells)
                    self.assertEqual(original_hash, fingerprint(data))
                    s.save()
                    save_json(s.output/'setup.json', data)
                    entry = ResultEntry(data['trial_id'], 'Cozmo-A', s.output,
                                        s.output/'exploration.json', s.output/'setup.json', 0)
                    _, saved, restored = _load_world(entry)
                    self.assertEqual(restored.landmarks, s.world.landmarks)
                    self.assertEqual(saved['current_pose'], list(truth))
                    self.assertEqual(len(saved['chess_position_updates']), 1)
                finally:
                    s.events.close()

    def test_changed_chess_clears_cube_sightline_and_is_a_motion_obstacle(self):
        world = World(setup(), 'Cozmo-A')
        planner = Explorer(world, ApplicationConfig())
        world.upsert_cube(500, 110, 0)
        start = (500, 400)
        self.assertFalse(LandmarkViewpoints.clear_view(start, '1', planner))
        world.landmarks['1'] = (610, 0)
        self.assertTrue(LandmarkViewpoints.clear_view(start, '1', planner))
        self.assertTrue(any((x, y) == (610, 0) for x, y, _ in planner.occluders()))
        self.assertTrue(any((x, y) == (610, 0) for x, y, _ in planner.obstacles()))
        planner.center_scan_done = True
        planner.landmark_command((*start, 0), {'1'})
        self.assertTrue(planner.landmark_view_clear)

    def test_live_refinement_keeps_shared_grid_and_setup_hash_compatible(self):
        data = setup()
        a, b = World(data, 'Cozmo-A'), World(copy.deepcopy(data), 'Cozmo-B')
        exchange = EncounterExchange(a, Network())
        a.landmarks['1'] = (560, 10)
        a.observe((250, 250, 0), {a.grid.key((250, 250))})
        b.merge(a.snapshot())
        self.assertEqual(a.covered, b.covered)
        self.assertEqual(exchange.setup_hash, fingerprint(b.setup))
        self.assertNotIn('landmarks', a.snapshot('full'))

    def test_start_robot_hidden_during_exploration_but_available_in_setup_editor(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        window = MapWindow(data['landmarks'], create_window=False)
        with patch.object(window, 'robot') as draw:
            window.draw(world, poses=data['robots'], current=(300, 300, 0))
            self.assertEqual([c.args[1] for c in draw.call_args_list], ['SELF'])
        with patch.object(window, 'robot') as draw:
            window.draw(world, poses=data['robots'])
            draw.assert_not_called()  # 古い結果も初期位置を現在地として表示しない
        with patch.object(window, 'robot') as draw:
            window.draw(poses=data['robots'])
            self.assertEqual(draw.call_count, 2)

    def test_v8_settings_upgrade_keeps_landmarks_and_refinement_custom_values(self):
        data = setup()
        data['conditions']['exploration_revision'] = 8
        data['conditions']['chess_refine_min_samples'] = 8
        original = copy.deepcopy(data)
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', boundary_margin_mm=60)
            save_json(cfg.setup_path, data)
            loaded = load_setup(cfg)
            self.assertEqual(cfg.chess_refine_min_samples, 8)
            self.assertEqual(cfg.exploration_revision, 9)
            for field in ('landmarks', 'robots', 'map_id', 'trial_id'):
                self.assertEqual(loaded[field], original[field])

    def test_invalid_average_settings_are_rejected(self):
        for kwargs in (dict(chess_refine_min_samples=1), dict(chess_refine_spread_mm=float('nan')),
                       dict(chess_refine_max_shift_mm=2), dict(chess_refine_min_span_seconds=50)):
            with self.assertRaises(ValueError):
                validate_config(ApplicationConfig(**kwargs))


if __name__ == '__main__':
    unittest.main()
