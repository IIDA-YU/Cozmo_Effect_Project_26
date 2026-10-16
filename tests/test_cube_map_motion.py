"""検知枠→地図→回避と、実測角度で停止する走行の回帰試験。

YOLOの出力のみを模擬する。座標変換・登録・描画・経路・走行処理は実装を使う。
実機カメラの認識率、摩擦や停止距離の保証をするテストではない。
"""
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from application.config import ApplicationConfig
from application.controller import load_setup
from application.controller import _make_experiment_setup
from research.cube_mapping import project_cube, classify_cube, cube_corners
from research.interface import MapWindow
from research.motion import run_motion
from research.perception import ResearchPerception, fuse_cube_face_detections
from research.planner import Explorer
from research.session import ResearchSession
from research.world import World, save_json, wrap
from test_research import setup
from test_session import Clock, Camera, Network


class SimulatedRobot:
    """45mmの車輪間隔で差動走行する模擬機体兼時計。"""
    def __init__(self, pose=(150, 150, 0)):
        self.position = list(pose)
        self.wheels = (0, 0)
        self.now = 100.0
        self.latched_reason = ''
        self.history = []

    def drive_wheels(self, left, right):
        self.wheels = left, right
        self.history.append(self.wheels)

    def monotonic(self):
        return self.now

    def pose(self):
        return tuple(self.position)

    def sleep(self, seconds):
        left, right = self.wheels
        speed, yaw = (left + right) / 2, (right - left) / 45
        theta = self.position[2] + yaw * seconds / 2
        self.position[0] += speed * math.cos(theta) * seconds
        self.position[1] += speed * math.sin(theta) * seconds
        self.position[2] = wrap(self.position[2] + yaw * seconds)
        self.now += seconds


def make_detector(config, boxes):
    detector = ResearchPerception.__new__(ResearchPerception)
    detector.config = config
    detector.class_ids = [0]
    detector.marker_counts = {}
    detections = [types.SimpleNamespace(xyxy=np.array([box]), conf=[0.9]) for box in boxes]
    detector.model = types.SimpleNamespace(predict=lambda *a, **kw: [types.SimpleNamespace(boxes=detections)])
    detector.robot_model = types.SimpleNamespace(predict=lambda *a, **kw: [types.SimpleNamespace(boxes=[])])
    detector.detector = types.SimpleNamespace(detectMarkers=lambda frame: ([], None, []))
    return detector


def make_session(config, detector):
    session = ResearchSession.__new__(ResearchSession)
    session.config = config
    session.world = World(setup(), config.robot_id)
    session.started = None
    session.perception = detector
    session.last_mapping_warning = None
    session.mapping_status = ''
    session.log = lambda *args, **kwargs: None
    return session


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()

    def test_height_avoids_oblique_width_distance_error(self):
        a = project_cube((130, 80, 190, 140), (240, 320), (150, 150, 0), self.cfg)
        b = project_cube((110, 80, 210, 140), (240, 320), (150, 150, 0), self.cfg)
        self.assertEqual(a['point'], b['point'])
        self.assertEqual(b['size_source'], 'whole-cube-height')

    def test_scaled_image_keeps_same_map_coordinate(self):
        box = (100, 80, 150, 130)
        a = project_cube(box, (240, 320), (150, 150, 0), self.cfg)
        b = project_cube(tuple(2 * x for x in box), (480, 640), (150, 150, 0), self.cfg)
        self.assertEqual(a['point'], b['point'])
        self.assertGreater(a['point'][1], 150)  # 画像の左は地図の+y。

    def test_rotated_and_negative_world_coordinates(self):
        obs = project_cube((135, 70, 185, 120), (240, 320), (-400, -400, math.pi / 2), self.cfg)
        self.assertAlmostEqual(obs['point'][0], -400)
        self.assertAlmostEqual(obs['point'][1], -145)

    def test_clipped_height_and_width_are_not_localized(self):
        valid = project_cube((120, 0, 200, 100), (240, 320), (150, 150, 0), self.cfg)
        clipped = project_cube((0, 0, 120, 100), (240, 320), (150, 150, 0), self.cfg)
        self.assertEqual(valid['status'], 'CLIPPED BOX')
        self.assertIsNone(valid['point'])
        self.assertEqual(clipped['status'], 'CLIPPED BOX')
        self.assertIsNone(clipped['point'])

    def test_nonfinite_and_tiny_boxes_have_explicit_reasons(self):
        invalid = project_cube((0, 0, math.nan, 100), (240, 320), (0, 0, 0), self.cfg)
        tiny = project_cube((150, 100, 153, 103), (240, 320), (0, 0, 0), self.cfg)
        self.assertEqual(invalid['status'], 'INVALID BOX')
        self.assertEqual(tiny['status'], 'SMALL BOX')

    def test_outside_map_is_explained_and_never_clamped(self):
        obs = project_cube((135, 70, 185, 120), (240, 320), (900, 150, 0), self.cfg)
        grid = World(setup(), 'Cozmo-A').grid
        self.assertEqual(classify_cube(obs, grid), 'OUTSIDE MAP')
        self.assertGreater(obs['point'][0], 1000)

    def test_legacy_default_range_and_motion_are_migrated(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder) / 'setup.json')
            data = setup()
            data['conditions'].pop('exploration_revision')
            data['conditions'].update(cube_map_near_mm=60, cube_map_far_mm=500,
                                      drive_speed_mm_s=30, motion_pulse_seconds=0.45)
            save_json(cfg.setup_path, data)
            loaded = load_setup(cfg)
            self.assertEqual(cfg.cube_map_far_mm, 700)
            self.assertEqual(cfg.drive_speed_mm_s, 45)
            self.assertEqual(loaded['conditions']['motion_pulse_seconds'], 0.8)
            self.assertEqual(loaded['trial_id'], data['trial_id'])

    def test_custom_speed_and_range_survive_upgrade(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder) / 'setup.json')
            data = setup()
            data['conditions'].pop('exploration_revision')
            data['conditions'].update(cube_map_far_mm=900, drive_speed_mm_s=22)
            save_json(cfg.setup_path, data)
            load_setup(cfg)
            self.assertEqual(cfg.drive_speed_mm_s, 22)
            self.assertEqual(cfg.cube_map_far_mm, 900)


class FaceFusionTests(unittest.TestCase):
    def test_two_adjacent_face_markers_become_one_cube_candidate(self):
        cfg = ApplicationConfig()
        detections = [
            {'box': (100, 80, 130, 120), 'confidence': .82},
            {'box': (132, 82, 162, 121), 'confidence': .90},
        ]
        fused = fuse_cube_face_detections(detections, cfg)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]['face_group_size'], 2)

    def test_separated_markers_remain_two_candidates(self):
        cfg = ApplicationConfig()
        detections = [
            {'box': (80, 80, 110, 120), 'confidence': .9},
            {'box': (160, 80, 190, 120), 'confidence': .9},
        ]
        fused = fuse_cube_face_detections(detections, cfg)
        self.assertEqual(len(fused), 2)


class DetectionToMapTests(unittest.TestCase):
    def test_real_stopped_observation_loop_registers_cube_and_updates_window(self):
        """sessionの実際の観測ループを通し、最終フレームまで地図登録が残る。"""
        clock = Clock()
        camera = Camera(clock)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original_sample = camera.get_frame_sample
        def sample():
            _, number, stamp = original_sample()
            return frame, number, stamp
        camera.get_frame_sample = sample
        pose = types.SimpleNamespace(position=types.SimpleNamespace(x=150, y=150),
                                     rotation=types.SimpleNamespace(angle_z=types.SimpleNamespace(radians=0)))
        cli = types.SimpleNamespace(pose=pose, latched_reason='', drive_wheels=lambda *args: None)
        slam = types.SimpleNamespace(_get_estimated_pose=lambda *args: args)
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder) / 'setup.json', cube_confirm_viewpoints=1)
            height = 50 * 235 / 250
            detector = make_detector(cfg, [(150, 80, 170, 80 + height)])
            with patch('research.session.time', clock), \
                 patch('research.session.ResearchPerception', lambda cfg: detector), \
                 patch('research.session.MapWindow', lambda *args: MapWindow(*args, create_window=False)), \
                 patch('research.session.BluetoothNetwork', Network):
                session = ResearchSession(cli, camera, slam, _make_experiment_setup(setup(), cfg, 1), cfg,
                                          types.SimpleNamespace(percent=80))
                try:
                    session.started = clock.now
                    session.observe_stopped()
                    self.assertEqual(len(session.world.cubes()), 1)
                    self.assertIn('MAPPED=1', session.mapping_status)
                    self.assertEqual(len(session.planner.obstacles()), 1)
                    rendered = session.window.draw(session.world)
                    x, y = session.window.pixel((420, 150))
                    self.assertEqual(tuple(rendered[y, x]), (100, 185, 255))
                finally:
                    session.events.close()
                    session.window.close()

    def test_700mm_camera_detection_stays_candidate_and_is_bounded_for_avoidance(self):
        cfg = ApplicationConfig()
        height = cfg.cube_width_mm * cfg.cube_focal_px / 690
        detector = make_detector(cfg, [(150, 80, 170, 80 + height)])
        session = make_session(cfg, detector)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        candidates, _, boxes, occluders = detector.detect(frame, (150, 150, 0))
        for _ in range(3):
            accepted = session._map_camera_detections(candidates, boxes)
            session._register_detected_cubes(accepted)
        self.assertFalse(session.world.cubes())
        self.assertIn('REVIEW RANGE', boxes[0]['label'])
        self.assertEqual(len(session.live_cube_candidates), 1)
        # 690mmという不確かな距離をそのまま使わず、300mm先の案内点へ丸める。
        self.assertAlmostEqual(session.live_cube_candidates[0][0], 470, delta=3)
        planner = Explorer(session.world, cfg)
        planner.set_transient_obstacles(occluders)
        self.assertFalse(session.world.grid.free_segment((350, 150), (520, 150), planner.obstacles()))

    def test_rejected_detection_is_visible_in_camera_and_status(self):
        cfg = ApplicationConfig()
        detector = make_detector(cfg, [(135, 70, 185, 120)])
        session = make_session(cfg, detector)
        candidates, _, boxes, _ = detector.detect(np.zeros((240, 320, 3)), (900, 150, 0))
        accepted = session._map_camera_detections(candidates, boxes)
        session._register_detected_cubes(accepted)
        self.assertFalse(session.world.cubes())
        self.assertIn('OUTSIDE MAP', boxes[0]['label'])
        self.assertIn('OUTSIDE MAP', session.mapping_status)

    def test_shared_cube_is_drawn_with_same_50mm_footprint(self):
        a = World(setup(), 'Cozmo-A')
        a.add_cube(400, 300, 1)
        b = World(a.setup, 'Cozmo-B')
        b.merge(a.snapshot())
        window = MapWindow(a.setup['landmarks'], create_window=False)
        np.testing.assert_array_equal(window.draw(a), window.draw(b))

    def test_clipped_close_cube_requests_yield_without_inventing_position(self):
        detector = make_detector(ApplicationConfig(), [(0, 0, 120, 100)])
        candidates, _, _, _ = detector.detect(np.zeros((240, 320, 3)), (150, 150, 0))
        self.assertTrue(detector.close_cube_unlocalized)
        self.assertFalse(candidates)


class MotionTests(unittest.TestCase):
    def run_burst(self, robot, command, **kwargs):
        return run_motion(robot, ApplicationConfig(), command, robot.pose,
                          kwargs.pop('process_events', lambda *args: None),
                          kwargs.pop('keep_running', lambda: True), clock=robot, **kwargs)

    def test_small_turn_stops_before_fixed_duration_overshoots(self):
        robot = SimulatedRobot()
        goal = math.radians(12)
        result = self.run_burst(robot, (-25, 25), heading=goal)
        self.assertLessEqual(abs(wrap(goal - robot.pose()[2])), math.radians(5))
        self.assertLess(result['duration'], 0.35)
        self.assertEqual(robot.wheels, (0, 0))

    def test_turn_across_pi_boundary(self):
        robot = SimulatedRobot((150, 150, math.radians(175)))
        goal = math.radians(-173)
        self.run_burst(robot, (-25, 25), heading=goal)
        self.assertLessEqual(abs(wrap(goal - robot.pose()[2])), math.radians(5))

    def test_forward_motion_is_visible_but_bounded(self):
        robot = SimulatedRobot()
        result = self.run_burst(robot, (45, 45), target=(500, 150))
        self.assertGreater(result['moved_mm'], 30)
        self.assertLessEqual(result['moved_mm'], 40)

    def test_close_target_stops_early(self):
        robot = SimulatedRobot()
        self.run_burst(robot, (45, 45), target=(170, 150))
        self.assertLessEqual(robot.pose()[0], 170)
        self.assertLessEqual(math.dist(robot.pose()[:2], (170, 150)), 10)

    def test_pause_and_exception_always_stop_wheels(self):
        for fail in (False, True):
            robot = SimulatedRobot()
            active = [True]
            def process(*args):
                active[0] = False
                if fail:
                    raise KeyboardInterrupt
            if fail:
                with self.assertRaises(KeyboardInterrupt):
                    self.run_burst(robot, (45, 45), process_events=process)
            else:
                self.run_burst(robot, (45, 45), process_events=process, keep_running=lambda: active[0])
            self.assertEqual(robot.wheels, (0, 0))

    def test_obstacle_is_checked_in_actual_heading(self):
        robot = SimulatedRobot()
        result = self.run_burst(robot, (45, 45), can_advance=lambda *args: False)
        self.assertEqual(result['stop_reason'], 'path_blocked')
        self.assertEqual(robot.pose(), (150, 150, 0))

    def test_observed_target_is_kept_until_robot_moves_near_it(self):
        world = World(setup(), 'Cozmo-A')
        planner = Explorer(world, ApplicationConfig())
        planner.goal = (650, 150)
        planner.path = [planner.goal]
        planner.predicted = {'10,3'}
        world.covered = {'10,3'}
        command = planner.command((150, 150, 0))
        self.assertGreater(command[0], 0)
        self.assertEqual(planner.goal, (650, 150))

    def test_inspection_directions_overlap_camera_field(self):
        planner = Explorer(World(setup(), 'Cozmo-A'), ApplicationConfig())
        planner.inspection_target_key = '4,4'
        planner._begin_or_continue_inspection_scan((225, 225, 0))
        headings = sorted([0] + [angle % math.tau for angle in planner.inspection_scan_queue])
        gaps = [(headings[(i + 1) % len(headings)] - a) % math.tau for i, a in enumerate(headings)]
        self.assertLess(max(gaps), math.radians(50))


if __name__ == '__main__':
    unittest.main()
