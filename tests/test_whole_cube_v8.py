"""全体検出、停止保持、遮蔽回避、終了条件の回帰試験。実機は使用しない。"""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from application.config import ApplicationConfig
from application.controller import load_setup, validate_config
from research.cube_calibration import signature, load_calibration
from research.cube_mapping import project_cube
from research.cube_review import CubeReviews, CubeVerifier
from research.cube_stop import CubeStopLock
from research.landmark_search import LandmarkViewpoints
from research.localization import LandmarkLocalizer
from research.motion import run_motion
from research.perception import filter_cube_detections
from research.planner import Explorer
from research.session import ResearchSession, TrialInterrupted
from research.world import World, save_json, wrap
from test_cube_map_motion import make_detector, SimulatedRobot
from test_research import setup
from test_session import Clock, Camera, Window, Network


def image_observation(confidence=.46, box=(130, 80, 190, 140)):
    return dict(box=box, confidence=confidence, point=(400, 300), distance=200,
                status='READY', range_reliable=True, pose_verified=True)


class WholeCubeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()

    def test_adjacent_whole_cubes_are_not_face_fused(self):
        boxes = [image_observation(box=(100, 80, 136, 120)),
                 image_observation(box=(138, 80, 174, 120))]
        self.assertEqual(len(filter_cube_detections(boxes, self.cfg)), 2)
        self.cfg.cube_detection_region = 'marker'
        self.assertEqual(len(filter_cube_detections(boxes, self.cfg)), 1)

    def test_duplicate_keeps_high_confidence_bbox_without_union(self):
        boxes = [image_observation(.5), image_observation(.8, (131, 81, 191, 141))]
        result = filter_cube_detections(boxes, self.cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['box'], boxes[1]['box'])

    def test_reliable_310mm_range_is_not_clamped_to_300mm(self):
        h = self.cfg.cube_width_mm * self.cfg.cube_focal_px / 310
        obs = project_cube((140, 80, 180, 80+h), (240, 320), (0, 0, 0), self.cfg)
        self.assertTrue(obs['range_reliable'])
        self.assertFalse(obs['point_is_bounded'])
        self.assertAlmostEqual(obs['point'][0], 330)

    def test_default_confirms_in_two_stopped_frames_from_one_view(self):
        world = World(setup(), 'Cozmo-A')
        reviews = CubeReviews(self.cfg)
        stop = CubeStopLock(self.cfg)
        for frame in range(2):
            obs = project_cube((130, 80, 190, 140), (240, 320), (150, 150, 0), self.cfg)
            obs.update(box=(130, 80, 190, 140), confidence=.46, pose_verified=True)
            stop.update([obs])
            accepted = reviews.update([obs], world.grid, frame)
            self.assertEqual(bool(accepted), frame == 1)
        self.assertEqual(len(reviews.items[0]['viewpoints']), 1)

    def test_full_cube_reliable_range_is_not_overridden_by_marker_approach(self):
        world = World(setup(), 'Cozmo-A')
        reviews = CubeReviews(self.cfg)
        for t, origin in enumerate(((100, 200), (180, 200))):
            obs = image_observation(.95)
            obs.update(point=(460, 200), camera_origin=origin, bearing_world=0,
                       distance=460-origin[0], range_reliable=True, box_height_ratio=.38)
            reviews.update([obs], world.grid, t)
        self.assertEqual(reviews.items[0]['point'], [460, 200])
        self.assertEqual(reviews.items[0]['refinement'], 'VISUAL RANGE')


class StopLockTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.lock = CubeStopLock(self.cfg)

    def test_weak_candidate_holds_without_timeout_then_two_good_frames_release(self):
        for _ in range(100):
            self.assertTrue(self.lock.update([image_observation(.4)]))
        self.assertTrue(self.lock.update([image_observation(.46)]))
        self.assertFalse(self.lock.update([image_observation(.46)]))

    def test_missed_frame_does_not_supply_consecutive_confirmation(self):
        self.lock.update([image_observation()])
        self.assertTrue(self.lock.update([]))
        self.assertTrue(self.lock.update([image_observation()]))
        self.assertFalse(self.lock.update([image_observation()]))

    def test_lost_candidate_needs_six_fresh_absent_frames(self):
        self.lock.update([image_observation(.4)])
        for _ in range(5):
            self.assertTrue(self.lock.update([]))
        self.assertFalse(self.lock.update([]))
        self.assertEqual(len(self.lock.lost), 1)

    def test_second_cube_cannot_release_first_weak_candidate(self):
        for _ in range(3):
            self.assertTrue(self.lock.update([image_observation(.4),
                image_observation(.9, (220, 80, 270, 140))]))
        self.assertEqual(self.lock.pending, 1)

    def test_visual_confirmation_does_not_forge_pose_or_range(self):
        for _ in range(2):
            obs = image_observation()
            obs.update(status='CLIPPED BOX', point=None, range_reliable=False, pose_verified=False)
            self.lock.update([obs])
        self.assertTrue(obs['visual_confirmed'])
        self.assertFalse(obs['range_reliable'])
        self.assertFalse(obs['pose_verified'])
        self.assertIsNone(obs['point'])


class StationarySessionTests(unittest.TestCase):
    def make_session(self, folder, confidence_sequence):
        clock = Clock()
        camera = Camera(clock)
        sample = camera.get_frame_sample
        def frame_sample():
            _, number, stamp = sample()
            return np.zeros((240, 320, 3), dtype=np.uint8), number, stamp
        camera.get_frame_sample = frame_sample
        cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json')
        detector = make_detector(cfg, [(130, 80, 190, 140)])
        predict = detector.model.predict
        calls = []
        def prediction(*args, **kwargs):
            index = min(len(calls), len(confidence_sequence)-1)
            confidence = confidence_sequence[index]
            calls.append(confidence)
            output = predict(*args, **kwargs)
            output[0].boxes[0].conf = [confidence]
            return output
        detector.model.predict = prediction
        raw = SimpleNamespace(position=SimpleNamespace(x=150, y=150),
                              rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0)))
        commands = []
        cli = SimpleNamespace(pose=raw, latched_reason='', drive_wheels=lambda *a: commands.append(a))
        slam = SimpleNamespace(_get_estimated_pose=lambda *args: args)
        patches = [patch('research.session.time', clock),
                   patch('research.session.ResearchPerception', lambda c: detector),
                   patch('research.session.MapWindow', Window),
                   patch('research.session.BluetoothNetwork', Network)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        session = ResearchSession(cli, camera, slam, setup(), cfg, SimpleNamespace(percent=80))
        self.addCleanup(session.events.close)
        session.started = clock.now
        return session, clock, calls, commands

    def test_real_loop_extends_past_three_frames_without_moving(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.4]*6 + [.46]*2)
            s.observe_stopped()
            self.assertEqual(len(calls), 8)
            self.assertTrue(all(c == (0, 0) for c in commands))
            self.assertFalse(s.cube_stop_active)
            self.assertEqual(len(s.world.cubes()), 1)
            self.assertEqual(s.cube_stop_count, 1)
            self.assertGreater(s.cube_stop_seconds, .5)

    def test_pose_unverified_releases_visual_lock_but_does_not_map(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.46])
            s._pose_is_verified = lambda: False
            s.observe_stopped()
            self.assertFalse(s.cube_stop_active)
            self.assertFalse(s.world.cubes())
            self.assertTrue(s.cube_reviews.unresolved(s.world)[0]['visual_confirmed'])
            self.assertIsNone(s.cube_verifier.command((150,150,0), s.planner, clock.now))

    def test_confirmed_cube_does_not_become_a_permanent_weak_detection_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.46]*3 + [.4]*10)
            s.observe_stopped()
            self.assertEqual(len(s.world.cubes()), 1)
            s.observe_stopped()
            self.assertEqual(len(calls), 6)
            self.assertFalse(s.cube_stop_active)
            self.assertEqual(len(s.world.cubes()), 1)

    def test_weak_candidate_stays_stopped_until_user_interrupts(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.4])
            key = s.key
            def interrupt(status):
                if len(calls) >= 12:
                    raise KeyboardInterrupt
                return key(status)
            s.key = interrupt
            with self.assertRaises(KeyboardInterrupt):
                s.observe_stopped()
            self.assertTrue(s.cube_stop_active)
            self.assertFalse(s.world.cubes())
            self.assertTrue(all(c == (0, 0) for c in commands))

    def test_low_battery_can_interrupt_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.4])
            key = s.key
            def drain(status):
                if len(calls) >= 4:
                    s.battery.percent = 1
                return key(status)
            s.key = drain
            with self.assertRaisesRegex(TrialInterrupted, 'low_battery'):
                s.observe_stopped()
            self.assertTrue(all(c == (0, 0) for c in commands))

    def test_motion_guard_blocks_nonzero_commands_during_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.4])
            s.cube_stop_active = True
            s._run_motion_burst((45,45))
            self.assertEqual(commands, [(0,0)])

    def test_duplicate_camera_frame_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            s, clock, calls, commands = self.make_session(folder, [.46])
            sample = s.camera.get_frame_sample
            def duplicates():
                frame, number, stamp = sample()
                return frame, number // 3, stamp
            s.camera.get_frame_sample = duplicates
            s.observe_stopped()
            self.assertEqual(len(calls), 3)
            self.assertGreaterEqual(s.camera.count, 6)
            self.assertEqual(len(s.world.cubes()), 1)


class LandmarkDetourTests(unittest.TestCase):
    def setUp(self):
        self.world = World(setup(), 'Cozmo-A')
        self.cfg = ApplicationConfig()
        self.planner = Explorer(self.world, self.cfg)
        self.planner.center_scan_done = True

    def test_occluded_chess_gets_a_clear_safe_viewpoint(self):
        self.world.upsert_cube(500, 160, 0)
        pose = (500, 400, -math.pi/2)
        self.assertFalse(LandmarkViewpoints.clear_view(pose[:2], '1', self.planner))
        self.planner.landmark_command(pose, {'1'})
        self.assertTrue(self.planner.landmark_view_clear)
        self.assertTrue(LandmarkViewpoints.clear_view(self.planner.goal, '1', self.planner))
        previous = pose[:2]
        for target in self.planner.path:
            self.assertTrue(self.world.grid.free_segment(previous, target, self.planner.obstacles()))
            previous = target

    def test_failed_observation_changes_viewpoint(self):
        self.planner.landmark_command((500, 400, 0), {'1'})
        first = self.planner.goal
        landmark = self.world.setup['landmarks']['1']
        heading = math.atan2(landmark[1]-first[1], landmark[0]-first[0])
        self.planner.path = []
        self.planner.landmark_command((*first, heading), {'1'})
        self.assertTrue(self.planner.landmark_waiting_observation)
        self.planner.landmark_command((*first, heading), {'1'})
        self.assertNotEqual(first, self.planner.goal)
        self.assertEqual(len(self.planner.landmark_views.attempts['1']), 1)

    def test_cube_candidate_remains_collision_obstacle_during_verification(self):
        self.planner.set_transient_obstacles([(300,300,25)])
        self.planner.mode = 'verify'
        self.planner.active_verify_point = (300,300)
        self.assertTrue(self.planner.obstacles())
        self.assertFalse(self.planner.can_advance((180,300,0), 40))

    def test_simulated_patrol_reobserves_all_six_with_cube_occlusion(self):
        self.world.upsert_cube(500, 160, 0)
        self.world.upsert_cube(500, 640, 0)
        robot = SimulatedRobot((500, 400, 0))
        localizer = LandmarkLocalizer(self.world.setup['landmarks'], self.cfg)
        for step in range(900):
            pose = robot.pose()
            visible = []
            for key, point in self.world.setup['landmarks'].items():
                distance = math.dist(pose[:2], point)
                angle = wrap(math.atan2(point[1]-pose[1], point[0]-pose[0]) - pose[2])
                if (80 <= distance <= 420 and abs(angle) <= math.radians(25)
                        and LandmarkViewpoints.clear_view(pose[:2], key, self.planner)):
                    dx, dy = point[0]-pose[0], point[1]-pose[1]
                    c, sn = math.cos(pose[2]), math.sin(pose[2])
                    visible.append(dict(local_x=c*dx+sn*dy, local_y=-sn*dx+c*dy))
            for frame in range(2):
                localizer.update(visible, pose, robot.now+frame*.01)
            unseen = set(self.world.setup['landmarks']) - localizer.seen_ids
            if not unseen:
                break
            command = self.planner.landmark_command(pose, unseen)
            self.assertIsNotNone(command)
            run_motion(robot, self.cfg, command, robot.pose, lambda *a: None, lambda: True,
                       target=self.planner.motion_target, heading=self.planner.motion_heading,
                       can_advance=self.planner.can_advance, clock=robot)
            robot.now += .2
            self.assertTrue(self.world.grid.inside(robot.pose()[:2], self.world.grid.margin))
            for cube in self.world.cubes():
                self.assertGreater(math.dist(robot.pose()[:2], (cube['x'],cube['y'])), self.planner._cube_safe_radius())
        self.assertEqual(localizer.seen_ids, set(self.world.setup['landmarks']))
        self.assertGreater(math.dist(robot.pose()[:2], (500,400)), 50)


class CompletionAndMigrationTests(unittest.TestCase):
    def test_default_runs_after_300_seconds_without_claiming_completion(self):
        s = ResearchSession.__new__(ResearchSession)
        s.config = ApplicationConfig()
        s.started = 0
        with patch('research.session.time.monotonic', return_value=900):
            self.assertFalse(s.trial_time_expired())

    def test_only_exact_six_ids_and_95_percent_can_complete(self):
        s = ResearchSession.__new__(ResearchSession)
        s.config = ApplicationConfig()
        s.setup = setup()
        s.world = World(s.setup, 'Cozmo-A')
        s.world.covered = set(s.world.grid.cells)
        s.localizer = SimpleNamespace(seen_ids={'0','1','2','3','4','other'})
        self.assertFalse(s.exploration_goal_reached())
        s.localizer.seen_ids = set(s.setup['landmarks'])
        self.assertTrue(s.exploration_goal_reached())
        s.cube_stop_active = True
        self.assertFalse(s.exploration_goal_reached())
        s.cube_stop_active = False
        s.setup['landmarks'].pop('5')
        self.assertFalse(s.exploration_goal_reached())

    def test_v7_defaults_migrate_without_changing_map_trial_or_custom_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', boundary_margin_mm=60)
            data = setup()
            original = copy.deepcopy(data)
            data['conditions'].update(exploration_revision=7, cube_confirm_frames=3,
                cube_review_confirm_frames=3, cube_review_confirm_confidence=.55,
                cube_confirm_viewpoints=2, trial_limit_seconds=300)
            save_json(cfg.setup_path, data)
            loaded = load_setup(cfg)
            self.assertEqual(cfg.cube_confirm_viewpoints, 1)
            self.assertEqual(cfg.cube_review_confirm_confidence, .45)
            self.assertEqual(cfg.trial_limit_seconds, 0)
            for key in ('trial_id','map_id','landmarks','robots'):
                self.assertEqual(loaded[key], original[key])
            data['conditions']['trial_limit_seconds'] = 120
            save_json(cfg.setup_path, data)
            load_setup(cfg)
            self.assertEqual(cfg.trial_limit_seconds, 120)

    def test_model_change_at_same_filename_invalidates_old_scale(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model = folder/'model.pt'
            model.write_bytes(b'old model bytes')
            cfg = ApplicationConfig(setup_path=folder/'setup.json', cube_model_path=model)
            save_json(folder/'cube_camera_calibration.json', dict(camera=signature(cfg), distance_scale=.5))
            model.write_bytes(b'new model bytes')
            self.assertFalse(load_calibration(cfg))
            self.assertEqual(cfg.cube_distance_scale, 1)

    def test_marker_calibration_cannot_be_used_for_whole_cube(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', cube_detection_region='marker')
            save_json(Path(folder)/'cube_camera_calibration.json', dict(camera=signature(cfg), distance_scale=.6))
            cfg.cube_detection_region = 'whole_cube'
            self.assertFalse(load_calibration(cfg))

    def test_config_rejects_changing_required_chess_to_five(self):
        with self.assertRaises(ValueError):
            validate_config(ApplicationConfig(required_chess_landmarks=5))


if __name__ == '__main__':
    unittest.main()
