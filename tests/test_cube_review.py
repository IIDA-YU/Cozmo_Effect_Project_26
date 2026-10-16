"""地図外の実ログ、弱い検知、確認走行、95%終了の回帰試験。"""
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from application.config import ApplicationConfig
from research.cube_review import CubeReviews, CubeVerifier
from research.interface import MapWindow
from research.planner import Explorer
from research.session import ResearchSession
from research.world import World
from test_research import setup
from test_cube_map_motion import SimulatedRobot
from research.motion import run_motion

ROOT = Path(__file__).resolve().parents[1]


def observation(point, confidence=0.4, status='READY'):
    return dict(point=point, confidence=confidence, status=status, distance=200)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.world = World(setup(), 'Cozmo-A')
        self.reviews = CubeReviews(self.cfg)

    def test_weak_detection_is_visible_then_confirms_with_stronger_frames(self):
        obs = observation((400, 300))
        self.assertFalse(self.reviews.update([obs], self.world.grid, 0))
        self.assertEqual(self.reviews.unresolved(self.world)[0]['point'], [400, 300])
        self.assertFalse(self.reviews.update([observation((400, 300), .46)], self.world.grid, 1))
        self.assertEqual(self.reviews.update([observation((400, 300), .46)], self.world.grid, 2), [(400, 300)])

    def test_missing_frame_and_multiple_boxes_do_not_fake_confirmation(self):
        self.reviews.update([observation((400, 300), .58)], self.world.grid, 1)
        self.reviews.update([], self.world.grid, 2)
        obs = [observation((400, 300), .58), observation((405, 300), .58)]
        self.assertFalse(self.reviews.update(obs, self.world.grid, 3))
        self.assertTrue(all(v['ready_hits'] == 1 for v in self.reviews.items))

    def test_outside_estimate_persists_without_entering_shared_records(self):
        for i in range(5):
            self.assertFalse(self.reviews.update([observation((1020, 300), .99)], self.world.grid, i))
        self.reviews.update([], self.world.grid, 6)
        self.assertEqual(len(self.reviews.unresolved(self.world)), 1)
        self.assertFalse(self.world.records)
        self.assertFalse(self.reviews.update([observation((980, 300), .9)], self.world.grid, 7))
        accepted = self.reviews.update([observation((980, 300), .9)], self.world.grid, 8)
        self.assertEqual(accepted, [(980, 300)])
        self.assertFalse(self.reviews.unresolved(self.world))

    def test_candidate_coordinate_is_visible_outside_polygon(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        point = (-50, 295)
        self.assertFalse(world.grid.inside(point))
        self.reviews.update([observation(point, .95)], world.grid, 0)
        window = MapWindow(data['landmarks'], create_window=False)
        image = window.draw(world, cube_candidates=[point])
        x, y = window.pixel(point)
        self.assertTrue(30 < x < window.WIDTH - 30)
        self.assertTrue(100 < y < 700)
        # 候補アイコンの辺に実際に色が付いている。
        px, py = window.pixel((point[0] - 25, point[1]))
        self.assertTrue((image[py-2:py+3, px-2:px+3] == [0,180,220]).all(axis=2).any())
        self.assertEqual(len(world.grid.cells), 320)
        self.assertFalse(world.cubes())

    def test_review_motion_approaches_target_with_safe_stops(self):
        self.reviews.update([observation((600, 400))], self.world.grid, 100)
        planner = Explorer(self.world, self.cfg)
        planner.set_transient_obstacles([(600, 400, 25)])
        verifier = CubeVerifier(self.reviews)
        robot = SimulatedRobot((150, 150, 0))
        initial_distance = math.dist(robot.pose()[:2], (600, 400))
        for _ in range(30):
            cmd = verifier.command(robot.pose(), planner, robot.now)
            if cmd is None:
                break
            run_motion(robot, self.cfg, cmd, robot.pose, lambda *a: None, lambda: True,
                       target=planner.motion_target, heading=planner.motion_heading,
                       can_advance=planner.can_advance, clock=robot)
            robot.now += .2
            self.assertTrue(self.world.grid.inside(robot.pose()[:2], self.world.grid.margin))
            self.assertGreater(math.dist(robot.pose()[:2], (600, 400)), planner._cube_safe_radius())
        self.assertLess(math.dist(robot.pose()[:2], (600, 400)), initial_distance - 80)
        self.assertEqual(robot.wheels, (0, 0))

    def test_outside_target_never_drives_outside_map(self):
        self.reviews.update([observation((1150, 400), .9)], self.world.grid, 100)
        planner = Explorer(self.world, self.cfg)
        verifier = CubeVerifier(self.reviews)
        robot = SimulatedRobot((750, 400, 0))
        for _ in range(30):
            cmd = verifier.command(robot.pose(), planner, robot.now)
            if cmd is None:
                break
            run_motion(robot, self.cfg, cmd, robot.pose, lambda *a: None, lambda: True,
                       target=planner.motion_target, heading=planner.motion_heading,
                       can_advance=planner.can_advance, clock=robot)
            robot.now += .2
            self.assertTrue(self.world.grid.inside(robot.pose()[:2], self.world.grid.margin))
        self.assertFalse(self.world.cubes())

    def test_confirmation_cancels_review_route(self):
        self.reviews.update([observation((600, 400))], self.world.grid, 0)
        planner = Explorer(self.world, self.cfg)
        verifier = CubeVerifier(self.reviews)
        verifier.command((150, 150, 0), planner, 0)
        self.world.add_cube(600, 400, 1)
        self.assertIsNone(verifier.command((150, 150, 0), planner, 1))
        self.assertEqual(planner.mode, 'coverage')
        self.assertFalse(planner.path)

    def test_unreachable_candidate_is_retained_and_retried(self):
        self.reviews.update([observation((200, 150))], self.world.grid, 0)
        planner = Explorer(self.world, self.cfg)
        planner.set_transient_obstacles([(200, 150, 25)])
        verifier = CubeVerifier(self.reviews)
        verifier.command((150, 150, 0), planner, 0)
        self.assertIsNone(verifier.command((150, 150, 0), planner, 1))
        self.assertTrue(self.reviews.unresolved(self.world))
        self.assertIsNone(verifier.command((150, 150, 0), planner, 2))
        # 再試行時も推定点が安全半径内なら、無理に前進せず候補を保持する。
        self.assertIsNone(verifier.command((150, 150, 0), planner, 7))
        self.assertTrue(self.reviews.unresolved(self.world))


class MultiViewDistanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig(cube_detection_region='marker', cube_confirm_viewpoints=2)
        self.world = World(setup(), 'Cozmo-A')
        self.reviews = CubeReviews(self.cfg)

    @staticmethod
    def geometric_observation(origin, target, estimated_distance):
        bearing = math.atan2(target[1] - origin[1], target[0] - origin[0])
        navigation = 300.0
        point = (
            origin[0] + navigation * math.cos(bearing),
            origin[1] + navigation * math.sin(bearing),
        )
        return dict(
            point=point,
            confidence=.95,
            status='REVIEW RANGE',
            distance=estimated_distance,
            sigma_mm=100,
            pose_verified=True,
            camera_origin=origin,
            bearing_world=bearing,
            range_reliable=False,
        )

    def test_marker_mode_does_not_confirm_far_range_from_same_viewpoint(self):
        target = (400, 300)
        origin = (100, 200)
        for t in range(6):
            accepted = self.reviews.update(
                [self.geometric_observation(origin, target, 630)], self.world.grid, t
            )
            self.assertFalse(accepted)
        self.assertEqual(len(self.reviews.items[0]['viewpoints']), 1)

    def test_two_view_rays_recover_position_even_when_bbox_distance_is_double(self):
        target = (400, 300)
        first = (100, 200)
        second = (100, 400)
        for t in range(3):
            self.reviews.update(
                [self.geometric_observation(first, target, 630)], self.world.grid, t
            )
        accepted = []
        for t in range(3, 6):
            accepted = self.reviews.update(
                [self.geometric_observation(second, target, 630)], self.world.grid, t
            )
        self.assertEqual(accepted, [target])
        self.assertEqual(self.reviews.items[0]['refinement'], 'TRIANGULATED')
        self.assertEqual(len(self.reviews.items[0]['viewpoints']), 2)


    def test_close_approach_uses_odometry_to_refine_far_candidate(self):
        # 初回はbbox距離が大きく外れていても、同じ方向へ160mm接近して
        # マーカーが画面を大きく占めた時点でodometryから位置を再構成する。
        first = self.geometric_observation((100, 200), (380, 200), 650)
        first['box_height_ratio'] = 0.10
        first['travelled_mm'] = 0.0
        self.reviews.update([first], self.world.grid, 0)

        close = self.geometric_observation((260, 200), (380, 200), 430)
        close['box_height_ratio'] = 0.35
        close['travelled_mm'] = 160.0
        self.reviews.update([close], self.world.grid, 1)

        item = self.reviews.items[0]
        self.assertEqual(item['refinement'], 'ODOMETRY APPROACH')
        self.assertAlmostEqual(item['point'][0], 380.0, delta=2.0)
        self.assertAlmostEqual(item['point'][1], 200.0, delta=2.0)


class CoverageTests(unittest.TestCase):
    def _session(self, seen):
        session = ResearchSession.__new__(ResearchSession)
        session.config = ApplicationConfig(expected_cubes=0)
        session.world = World(setup(), 'Cozmo-A')
        session.setup = session.world.setup
        session.localizer = SimpleNamespace(seen_ids=set(seen))
        return session

    def test_95_percent_visual_is_not_enough_until_all_six_chess_seen(self):
        session = self._session({'0','1','2','3','4'})
        cells = list(session.world.grid.cells)
        session.world.covered = set(cells[:math.ceil(len(cells) * 0.96)])
        self.assertFalse(session.exploration_goal_reached())

    def test_six_chess_and_95_percent_finish_even_when_route_is_low(self):
        session = self._session({'0','1','2','3','4','5'})
        cells = list(session.world.grid.cells)
        session.world.covered = set(cells[:math.ceil(len(cells) * 0.97)])
        self.assertLess(session.trail_coverage_ratio(), .80)
        self.assertTrue(session.exploration_goal_reached())

class SessionCompletionTests(unittest.TestCase):
    def test_run_returns_at_target_before_review_or_patrol_and_saves_candidates(self):
        import tempfile
        from unittest.mock import patch
        from application.controller import _make_experiment_setup
        from test_session import Clock, Camera, Window, Network, Perception
        clock = Clock()
        camera = Camera(clock)
        pose = SimpleNamespace(position=SimpleNamespace(x=150, y=150),
                               rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0)))
        cli = SimpleNamespace(pose=pose, latched_reason='', drive_wheels=lambda *a: None,
                              set_head_angle=lambda *a: None, set_lift_height=lambda *a: None,
                              acknowledge_placement=lambda: None)
        slam = SimpleNamespace(_get_estimated_pose=lambda *a: (150,150,0),
                               _set_estimated_pose=lambda *a: None)
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', trial_limit_seconds=3)
            with patch('research.session.time', clock), patch('research.session.ResearchPerception', Perception), \
                 patch('research.session.MapWindow', Window), patch('research.session.BluetoothNetwork', Network):
                session = ResearchSession(cli, camera, slam, _make_experiment_setup(setup(), cfg, 1),
                                          cfg, SimpleNamespace(percent=80))
                session.localizer = SimpleNamespace(
                    seen_ids={'0','1','2','3','4','5'},
                    report={'status':'POSE VERIFIED'},
                    trusted=lambda *a: True,
                )
                def observe():
                    session.world.covered = set(list(session.world.grid.cells)[:304])  # 304/320 =95%
                    session.cube_reviews.update([observation((1200,400), .9)], session.world.grid, clock.now)
                    return 150,150,0
                session.observe_stopped = observe
                def home():
                    session.reason = 'returned_home'
                    session.returned_home = True
                    return True
                session.return_home = home
                def must_not_run(*a):
                    raise AssertionError('6駒+95%到達後に探索命令が発行された')
                session.cube_verifier.command = must_not_run
                session.planner.command = must_not_run
                session.run()
                data = json.loads((session.output/'exploration.json').read_text())
                self.assertEqual(data['metrics']['completion_reason'], 'six_chess_and_95pct_view')
                self.assertTrue(data['metrics']['returned_home'])
                self.assertFalse(data['cubes'])
                self.assertEqual(data['cube_candidates'][0]['point'], [1200,400])
