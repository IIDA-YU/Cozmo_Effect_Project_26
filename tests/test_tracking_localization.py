"""重複キューブ、移動後の再観測、チェス照合、共有版管理の回帰テスト。"""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from application.config import ApplicationConfig
from application.controller import load_setup
from research.cube_calibration import calibrated_scale, load_calibration, signature
from research.cube_review import CubeReviews
from research.localization import LandmarkLocalizer, transform
from research.session import ResearchSession
from research.world import World, save_json
from test_research import setup
from test_session import Clock, Camera, Window, Network
from test_cube_map_motion import make_detector
from tools.import_measured_map import import_map, validate_landmarks


def obs(x, y=300, verified=True):
    return dict(point=(x, y), confidence=.95, status='READY', distance=300,
                sigma_mm=20, pose_verified=verified)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.tracks = CubeReviews(self.cfg)
        self.world = World(setup(), 'Cozmo-A')

    def frame(self, points, t):
        self.tracks.update(points, self.world.grid, t)
        for item in self.tracks.accepted_items:
            item['record_id'] = self.world.upsert_cube(*item['estimate'], t, item['record_id'])

    def test_gradual_90mm_shift_updates_one_id_instead_of_three(self):
        for t, x in enumerate([400]*3 + [420]*3 + [445]*3 + [470]*3 + [490]*7):
            self.frame([obs(x)], t)
        self.assertEqual(len(self.world.cubes()), 1)
        self.assertEqual(len(self.tracks.items), 1)
        self.assertEqual(next(iter(self.world.records)), 'Cozmo-A:0001')
        self.assertAlmostEqual(self.world.cubes()[0]['x'], 490)
        self.assertGreater(self.world.records['Cozmo-A:0001']['revision'], 1)

    def test_two_visible_adjacent_cubes_keep_two_ids(self):
        for t in range(3):
            self.frame([obs(400), obs(450)], t)
        self.assertEqual(len(self.world.cubes()), 2)
        for t in range(3, 10):
            self.frame([obs(410), obs(460)], t)
        self.assertEqual(len(self.world.cubes()), 2)
        self.assertEqual(len(self.tracks.items), 2)

    def test_simultaneous_tracks_are_not_merged_by_old_45mm_radius(self):
        self.world.upsert_cube(400, 300, 0)
        self.world.upsert_cube(440, 300, 0)
        self.assertEqual(len(self.world.cubes()), 2)

    def test_duplicate_boxes_do_not_supply_confirmation_or_extra_id(self):
        self.frame([obs(400), obs(402), obs(398)], 0)
        self.assertEqual(len(self.tracks.items), 1)
        self.assertFalse(self.world.records)
        self.assertEqual(self.tracks.items[0]['ready_hits'], 1)

    def test_unverified_pose_never_confirms_even_with_high_confidence(self):
        for t in range(6):
            self.frame([obs(400, verified=False)], t)
        self.assertFalse(self.world.cubes())
        self.assertEqual(len(self.tracks.unresolved(self.world)), 1)

    def test_missing_frame_resets_evidence_and_candidate_expires(self):
        self.frame([obs(400)], 0)
        self.frame([], 2)
        self.frame([obs(400)], 3)
        self.assertFalse(self.world.cubes())
        self.frame([], 30)
        self.assertFalse(self.tracks.unresolved(self.world))

    def test_pose_fix_preserves_id_then_reobserves_position(self):
        for t in range(3):
            self.frame([obs(400)], t)
        self.tracks.reset_after_pose_fix((150,150,0), (230,150,0))
        for t in range(3,6):
            self.frame([obs(480)], t)
        self.assertEqual(len(self.world.cubes()), 1)
        self.assertAlmostEqual(self.world.cubes()[0]['x'], 480)

    def test_ambiguous_reacquisition_does_not_create_third_cube(self):
        for t in range(3):
            self.frame([obs(400), obs(460)], t)
        self.frame([obs(430)], 3)
        self.assertEqual(len(self.tracks.items), 2)
        self.assertEqual(len(self.world.cubes()), 2)

    def test_old_shared_snapshot_cannot_undo_position_update(self):
        peer = World(setup(), 'Cozmo-B')
        key = self.world.upsert_cube(400,300,0)
        old = self.world.snapshot()
        self.world.upsert_cube(440,300,1,key)
        newest = self.world.snapshot()
        peer.merge(newest)
        peer.merge(old)
        self.assertEqual(peer.records[key]['x'], 440)
        peer.merge(newest)
        self.world.merge(peer.snapshot())
        self.assertEqual(self.world.records, peer.records)
        bad = copy.deepcopy(newest)
        bad['records'][key]['x'] = 441
        with self.assertRaises(ValueError):
            peer.merge(bad)


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ApplicationConfig()
        self.true_pose = (200, 250, .25)
        self.local = [(180,-130),(240,0),(180,130),(360,160)]
        self.landmarks = {str(i): transform(p,self.true_pose) for i,p in enumerate(self.local)}
        self.loc = LandmarkLocalizer(self.landmarks, self.cfg)
        self.predicted = (218,240,.29)

    def observations(self, points=None):
        return [dict(local_x=x, local_y=y) for x,y in (self.local if points is None else points)]

    def test_six_landmarks_accumulate_across_separate_camera_views(self):
        landmarks = {str(i): point for i, point in enumerate([
            (0,0), (500,0), (1000,0), (1000,800), (500,800), (0,800)
        ])}
        loc = LandmarkLocalizer(landmarks, self.cfg)
        pose = (0,0,0)
        now = 0
        for key, point in landmarks.items():
            obs = [dict(local_x=point[0], local_y=point[1])]
            loc.update(obs, pose, now); now += 1
            loc.update(obs, pose, now); now += 1
        self.assertEqual(loc.seen_ids, set(landmarks))
        self.assertEqual(loc.report['seen_count'], 6)

    def test_joint_translation_rotation_recovered_and_map_unchanged(self):
        original = copy.deepcopy(self.loc.landmarks)
        first, report = self.loc.update(self.observations(), self.predicted, 0)
        self.assertFalse(report['corrected'])
        pose, report = self.loc.update(self.observations(), self.predicted, 1)
        self.assertTrue(report['corrected'])
        for a,b in zip(pose, self.true_pose):
            self.assertAlmostEqual(a,b,places=6)
        self.assertEqual(original, self.loc.landmarks)
        self.assertLess(report['residual_mm'], 1e-6)
        self.assertTrue(all(abs(v['error_mm']) < 1e-6 for v in report['pairs']))

    def test_two_points_correct_only_when_distance_matches(self):
        values = self.observations(self.local[:2])
        self.loc.update(values, self.predicted, 0)
        _, report = self.loc.update(values, self.predicted, 1)
        self.assertTrue(report['corrected'])
        bad = self.observations([(180,-130),(320,0)])
        _, report = self.loc.update(bad, self.predicted, 2)
        self.assertFalse(report['corrected'])
        self.assertEqual(report['status'], 'GEOMETRY MISMATCH')

    def test_one_point_cannot_set_full_pose(self):
        for t in range(5):
            pose, report = self.loc.update(self.observations(self.local[:1]), self.predicted, t)
        self.assertFalse(report['corrected'])
        self.assertEqual(pose, self.predicted)
        self.assertFalse(self.loc.trusted(5,0))

    def test_moved_piece_is_excluded_when_three_other_pieces_agree(self):
        points = self.local[:3] + [(420,160)]
        for t in range(3):
            pose, report = self.loc.update(self.observations(points), self.predicted, t)
        self.assertTrue(report['corrected'])
        self.assertEqual(report['suspect_ids'], ['3'])
        self.assertAlmostEqual(pose[0], self.true_pose[0])
        self.assertEqual(report['inlier_ids'], ['0','1','2'])

    def test_two_of_three_cannot_decide_which_piece_moved(self):
        points = self.local[:2] + [(180,190)]
        for t in range(3):
            _, report = self.loc.update(self.observations(points), self.predicted, t)
        self.assertFalse(report['corrected'])

    def test_overlarge_correction_is_rejected(self):
        self.cfg.chess_max_shift_mm = 10
        for t in range(3):
            _, report = self.loc.update(self.observations(), self.predicted, t)
        self.assertFalse(report['corrected'])

    def test_geometry_failure_invalidates_recent_fix(self):
        for t in (0,1):
            self.loc.update(self.observations(), self.predicted, t)
        self.assertTrue(self.loc.trusted(2,0))
        self.loc.update(self.observations([(180,-130),(320,0)]),self.predicted,2)
        self.assertFalse(self.loc.trusted(2,0))
        self.loc.update([],self.predicted,3)
        self.assertFalse(self.loc.trusted(3,0))

    def test_fix_expires_by_time_or_odometry_distance(self):
        for t in (0,1):
            self.loc.update(self.observations(), self.predicted, t, 10)
        self.assertTrue(self.loc.trusted(5,100))
        self.assertFalse(self.loc.trusted(5,161))
        self.assertFalse(self.loc.trusted(32,10))

    def test_ambiguous_landmark_id_is_not_used(self):
        loc = LandmarkLocalizer({'0':(100,0),'1':(120,0)}, self.cfg)
        _, report = loc.update([dict(local_x=110,local_y=0)], (0,0,0),0)
        self.assertFalse(report['matches'])


class CalibrationTests(unittest.TestCase):
    def test_scale_is_absolute_and_load_does_not_multiply_twice(self):
        value = calibrated_scale([200,201,199,200,200], 250, 1)
        self.assertAlmostEqual(value,1.25)
        self.assertAlmostEqual(calibrated_scale([250]*5,250,value),1.25)
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json')
            save_json(Path(folder)/'cube_camera_calibration.json',dict(camera=signature(cfg),distance_scale=value))
            load_calibration(cfg)
            load_calibration(cfg)
            self.assertEqual(cfg.cube_distance_scale,1.25)
            cfg.head_angle = 0
            self.assertFalse(load_calibration(cfg))
            self.assertEqual(cfg.cube_distance_scale,1.0)

    def test_unstable_samples_are_rejected(self):
        with self.assertRaises(ValueError): calibrated_scale([150,180,200,230,250],250,1)

    def test_measured_map_requires_new_initial_pose_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'measured.json'
            save_json(source, dict(landmarks=setup()['landmarks']))
            save_json(root/'slam_data.json', dict(landmarks=setup()['landmarks']))
            save_json(root/'setup.json', setup())
            import_map(source,root)
            cfg = ApplicationConfig(setup_path=root/'setup.json')
            with self.assertRaisesRegex(ValueError, '初期姿勢'): load_setup(cfg)
            self.assertEqual(len(list((root/'map_backups').glob('*/setup.json'))),1)


class SessionGeometryTests(unittest.TestCase):
    def test_unverified_pose_does_not_lock_robot_into_repeated_turns(self):
        from research.planner import Explorer
        session = ResearchSession.__new__(ResearchSession)
        session.config = ApplicationConfig()
        session.localizer = LandmarkLocalizer(setup()['landmarks'],session.config)
        session.distance = 0
        session.localization_scan_since = None
        session.paused = False
        logs = []
        session.log = lambda *a,**k:logs.append((a,k))
        session.pose = lambda:(150,150,0)
        session.planner = Explorer(World(setup(),'Cozmo-A'),session.config)
        clock = Clock()
        with patch('research.session.time',clock):
            self.assertFalse(session._recheck_localization())
            clock.now += 46
            self.assertFalse(session._recheck_localization())
        self.assertTrue(any(a and a[0] == 'localization_roaming_recovery' for a,k in logs))

    def test_chess_fix_precedes_cube_projection_and_return_does_not_add_coverage(self):
        clock = Clock()
        camera = Camera(clock)
        frame = np.zeros((240,320,3),dtype=np.uint8)
        original = camera.get_frame_sample
        def sample():
            _, n, t = original()
            return frame,n,t
        camera.get_frame_sample = sample
        raw = SimpleNamespace(position=SimpleNamespace(x=0,y=0), rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0)))
        cli = SimpleNamespace(pose=raw,latched_reason='',drive_wheels=lambda *a:None)
        state = [(165,145,.01)]
        truth = (150,150,0)
        data = setup()
        # Two simulated visible chess detections; local distances are grounded in fixed map.
        landmarks = [('1',data['landmarks']['1']),('2',data['landmarks']['2'])]
        detections = [dict(box=(10,10,20,30),confidence=.9,point=p) for _,p in landmarks]
        slam = SimpleNamespace(_get_estimated_pose=lambda *a:state[0],
                               _set_estimated_pose=lambda *a:state.__setitem__(0,tuple(a[-3:])),
                               detect_chess_pieces=lambda f:detections,
                               _estimate_landmark_observation=lambda d,w,*p:dict(local_x=d['point'][0]-150,local_y=d['point'][1]-150))
        with tempfile.TemporaryDirectory() as folder:
            cfg = ApplicationConfig(setup_path=Path(folder)/'setup.json', cube_confirm_viewpoints=1)
            detector = make_detector(cfg,[(135,70,185,120)])
            with patch('research.session.time',clock), patch('research.session.ResearchPerception',lambda c:detector), \
                 patch('research.session.MapWindow',Window), patch('research.session.BluetoothNetwork',Network):
                session = ResearchSession(cli,camera,slam,data,cfg,SimpleNamespace(percent=80))
                session.started = clock.now
                try:
                    session.observe_stopped()
                    session.observe_stopped()
                    self.assertAlmostEqual(state[0][0],truth[0])
                    self.assertEqual(len(session.world.cubes()),1)
                    self.assertAlmostEqual(session.world.cubes()[0]['x'],405)
                    before = copy.deepcopy(session.world.export())
                    session.observe_stopped(record=False)
                    self.assertEqual(before,session.world.export())
                    self.assertTrue(any('chess_localization' in l for l in (session.output/'events.jsonl').read_text().splitlines()))
                finally:
                    session.events.close()
