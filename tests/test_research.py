"""実機不要：共有の整合性・探索経路・停止制御の回帰テスト。"""
import copy
import json
import math
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

from application.config import ApplicationConfig
from application.controller import (
    make_setup, load_setup, validate_config, validate_ports,
    validate_bluetooth_runtime, _make_experiment_setup,
)
from research.world import World,Grid,save_json,fingerprint
from research.interface import MapWindow,_position_is_valid
from research.session import prepare_output_for_run
from research.exchange import EncounterExchange
from research.planner import Explorer
from telecommunication.bluetooth_communication import LineDecoder


def setup():
    config = ApplicationConfig(boundary_margin_mm=60)
    slam = types.SimpleNamespace(true_landmarks={0:(0,0),1:(500,0),2:(1000,0),3:(1000,800),4:(500,800),5:(0,800)})
    poses = {'Cozmo-A':dict(x=150,y=150,theta=0,marker_id=1),
             'Cozmo-B':dict(x=850,y=650,theta=math.pi,marker_id=2)}
    return make_setup(slam,poses,config)


class Network:
    def __init__(self):
        self.messages = []
    def send(self,peer,msg):
        self.messages.append((peer,copy.deepcopy(msg)))
        return True
    def receive(self):
        return []


class ExperimentRobotCountTests(unittest.TestCase):
    def test_single_robot_mode_needs_no_bluetooth_port(self):
        data = setup()
        config = ApplicationConfig(
            robot_id='Cozmo-A',
            bluetooth_ports={'Cozmo-B': 'CHANGE_ME'},
            sharing_mode='full',
        )

        runtime = _make_experiment_setup(data, config, 1)

        self.assertEqual(set(runtime['robots']), {'Cozmo-A'})
        self.assertEqual(runtime['experiment_robot_count'], 1)
        # 1台探索では未設定COMポートでもBluetooth確認を行わない。
        validate_ports(config, runtime)
        validate_bluetooth_runtime(config, runtime)

    def test_two_robot_mode_keeps_both_robots(self):
        data = setup()
        config = ApplicationConfig(
            robot_id='Cozmo-A',
            bluetooth_ports={'Cozmo-B': 'COM5'},
            sharing_mode='full',
        )

        runtime = _make_experiment_setup(data, config, 2)

        self.assertEqual(set(runtime['robots']), {'Cozmo-A', 'Cozmo-B'})
        self.assertEqual(runtime['experiment_robot_count'], 2)
        validate_ports(config, runtime)


class LandmarkPatrolPlannerTests(unittest.TestCase):
    def test_center_allows_full_scan_before_patrol(self):
        data = setup()
        cfg = ApplicationConfig(boundary_margin_mm=60)
        planner = Explorer(World(data, 'Cozmo-A'), cfg)
        command = planner.landmark_command((500, 400, 0), {'0','1','2','3','4','5'})
        self.assertTrue(planner.center_scan_started)
        self.assertIsNone(planner.goal)
        self.assertLess(command[0] * command[1], 0)

    def test_edge_skips_full_scan_and_targets_landmark_vantage(self):
        data = setup()
        cfg = ApplicationConfig(boundary_margin_mm=60)
        planner = Explorer(World(data, 'Cozmo-A'), cfg)
        command = planner.landmark_command((150, 150, 0), {'0','1','2','3','4','5'})
        self.assertFalse(planner.center_scan_started)
        self.assertEqual(planner.mode, 'landmark_search')
        self.assertIsNotNone(planner.goal)
        self.assertIsNotNone(command)


class WorldTests(unittest.TestCase):
    def setUp(self):
        self.setup = setup()
        self.a = World(self.setup,'Cozmo-A')
        self.b = World(self.setup,'Cozmo-B')

    def test_identity_and_spatial_merge(self):
        self.a.add_cube(400,300,1)
        self.b.add_cube(410,305,2)
        sa,sb = self.a.snapshot(),self.b.snapshot()
        self.a.merge(sb); self.b.merge(sa)
        self.assertEqual(self.a.cubes(),self.b.cubes())
        self.assertEqual(len(self.a.cubes()),1)
        self.assertEqual(len(self.a.cubes()[0]['aliases']),2)

    def test_distinct_cubes_and_no_additive_duplicate(self):
        self.a.add_cube(400,300,1)
        self.a.add_cube(600,300,2)
        self.a.observe((200,200,0),{'4,4'})
        self.b.merge(self.a.snapshot())
        first = self.b.export()
        self.b.merge(self.a.snapshot())
        self.assertEqual(first,self.b.export())
        self.assertEqual(len(self.b.cubes()),2)
        self.assertEqual(self.b.local_covered,set())

    def test_invalid_data_is_atomic(self):
        self.a.add_cube(400,300,1)
        data = self.a.snapshot()
        data['covered'] = ['999,999']
        with self.assertRaises(ValueError): self.b.merge(data)
        self.assertEqual(self.b.records,{})

    def test_nan_rejected(self):
        self.a.add_cube(400,300,1)
        data = self.a.snapshot()
        next(iter(data['records'].values()))['x'] = float('nan')
        with self.assertRaises(ValueError): self.b.merge(data)

    def test_conflicting_record_rejected(self):
        self.a.add_cube(400,300,1)
        self.b.merge(self.a.snapshot())
        data = self.a.snapshot()
        next(iter(data['records'].values()))['x']=450
        with self.assertRaises(ValueError): self.b.merge(data)

    def test_trial_starts_empty(self):
        self.a.add_cube(400,300,1)
        self.a.observe((200,200,0),{'4,4'})
        new = World(self.setup,'Cozmo-A')
        self.assertFalse(new.covered or new.records or new.trails)

    def test_cubes_only(self):
        self.a.add_cube(400,300,1)
        self.a.observe((200,200,0),{'4,4'})
        self.b.merge(self.a.snapshot('cubes'))
        self.assertEqual(len(self.b.cubes()),1)
        self.assertFalse(self.b.covered or self.b.trails)

    def test_polygon_not_rectangle(self):
        g = Grid([(0,0),(600,0),(0,600)],50,50)
        self.assertFalse(g.inside((500,500)))
        self.assertTrue(g.inside((150,150),50))

    def test_initial_position_can_be_near_map_edge(self):
        g = Grid([(0,0),(1000,0),(1000,800),(0,800)],50,70)
        point = (10,400)
        self.assertTrue(g.inside(point))
        self.assertFalse(g.inside(point,g.margin))
        valid,_ = _position_is_valid(g,point,{})
        self.assertTrue(valid)

    def test_robot_display_uses_90_by_60_mm_footprint(self):
        window = MapWindow.__new__(MapWindow)
        window.robot_length_mm = 90
        window.robot_width_mm = 60
        corners = window._robot_corners((100,200,0))
        self.assertEqual(
            corners,
            [(145,230),(145,170),(55,170),(55,230)],
        )

    def test_route_can_enter_safe_area_from_edge_start(self):
        g = Grid([(0,0),(1000,0),(1000,800),(0,800)],50,70)
        previous,start = g.reachable((10,400),[])
        self.assertIsNotNone(start)
        self.assertTrue(g.inside(g.cells[start],g.margin))
        self.assertIn(start,previous)

    def test_planner_commands_motion_from_edge_start(self):
        setup_data = {
            'schema_version': 1,
            'map_id': 'edge-map',
            'trial_id': 'edge-trial',
            'landmarks': {
                1: (0, 0),
                2: (1000, 0),
                3: (1000, 800),
                4: (0, 800),
            },
            'robots': {
                'Cozmo-A': {'x': 10, 'y': 400, 'theta': 0, 'marker_id': 1},
            },
            'cell_mm': 50,
            'margin_mm': 70,
            'cube_merge_mm': 80,
        }
        world = World(setup_data, 'Cozmo-A')
        planner = Explorer(world, ApplicationConfig(boundary_margin_mm=70))

        # 以前は reachable() では進入経路を作れたのに、command() 側の
        # free_segment() が始点の70mm余白を要求し、(0, 0) になっていた。
        command = planner.command((10, 400, 0))

        self.assertIsNotNone(command)
        self.assertNotEqual(command, (0, 0))
        self.assertIsNotNone(planner.goal)

    def test_entry_segment_still_rejects_obstacle(self):
        g = Grid([(0,0),(1000,0),(1000,800),(0,800)],50,70)
        start = (10, 400)
        target = (75, 425)
        self.assertTrue(g.free_entry_segment(start, target, []))
        self.assertFalse(g.free_entry_segment(start, target, [(40, 410, 35)]))

    def test_fov_and_occlusion(self):
        g = self.a.grid
        pose = (125,125,0)
        visible = g.visible(pose,50,500,math.pi/3)
        occluded = g.visible(pose,50,500,math.pi/3,[(275,125,40)])
        self.assertIn('8,2',visible)
        self.assertNotIn('8,2',occluded)
        self.assertNotIn('1,2',visible)

    def test_path_goes_around_cube_and_respects_margin(self):
        g = self.a.grid
        obstacles = [(500,400,150)]
        prev,start = g.reachable((125,425),obstacles)
        path = g.route(prev,'17,8')
        for a,b in zip(path,path[1:]):
            self.assertTrue(g.free_segment(g.cells[a],g.cells[b],obstacles))
        self.assertFalse(g.free_segment((125,425),(875,425),obstacles))

    def test_covered_cells_are_still_passable(self):
        self.a.covered = set(self.a.grid.cells)
        prev,_ = self.a.grid.reachable((125,425),[])
        self.assertIn('17,8',prev)
        cfg = ApplicationConfig(boundary_margin_mm=60, expected_cubes=0)
        self.assertIsNotNone(Explorer(self.a,cfg).command((125,425,0)))

    def test_planner_targets_new_coverage(self):
        p = Explorer(self.a,ApplicationConfig(boundary_margin_mm=60))
        self.assertTrue(p.choose((125,425,0)))
        self.assertTrue(p.predicted-self.a.covered)
        self.assertTrue(all(self.a.grid.inside(pos,60) for pos in p.path))


class CubeSearchTests(unittest.TestCase):
    def test_two_of_three_frames_can_confirm_cube_position(self):
        from research.session import _confirm_cube_positions

        confirmed = _confirm_cube_positions(
            [
                [(400, 300)],
                [(430, 315)],
                [],
            ],
            minimum_frames=2,
            match_mm=70,
        )
        self.assertEqual(len(confirmed), 1)
        self.assertAlmostEqual(confirmed[0][0], 415)
        self.assertAlmostEqual(confirmed[0][1], 307.5)

    def test_adjacent_50mm_cubes_are_not_merged(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        self.assertLess(data['cube_merge_mm'], 50)
        world.add_cube(400, 300, 1)
        world.add_cube(450, 300, 2)
        self.assertEqual(len(world.cubes()), 2)

    def test_full_visual_coverage_still_roams_when_route_coverage_is_low(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        world.covered = set(world.grid.cells)
        cfg = ApplicationConfig(boundary_margin_mm=60, expected_cubes=3)
        planner = Explorer(world, cfg)
        command = planner.command((125, 425, 0))
        self.assertIsNotNone(command)
        self.assertEqual(planner.mode, 'coverage')
        self.assertIsNotNone(planner.goal)
        self.assertGreaterEqual(math.dist((125,425), planner.goal), cfg.exploration_min_move_mm)

    def test_unknown_cube_count_does_not_disable_physical_roaming(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        world.covered = set(world.grid.cells)
        cfg = ApplicationConfig(boundary_margin_mm=60, expected_cubes=0)
        command = Explorer(world, cfg).command((125, 425, 0))
        self.assertIsNotNone(command)

    def test_pheromone_deposit_evaporates_and_is_shareable(self):
        data = setup()
        a = World(data, 'Cozmo-A')
        b = World(data, 'Cozmo-B')
        pose = (225, 225, 0)
        a.observe(pose, set())
        a.deposit_pheromone(pose, 1.0, 1)
        key = a.grid.key(pose[:2])
        before = a.pheromones[key]
        a.evaporate_pheromones(0.1)
        self.assertLess(a.pheromones[key], before)
        b.merge(a.snapshot('full'))
        self.assertGreater(b.pheromone_at(key), 0)
        self.assertIn(key, b.trails['Cozmo-A'])

    def test_unconfirmed_camera_cube_is_used_as_safe_obstacle(self):
        data = setup()
        world = World(data, 'Cozmo-A')
        cfg = ApplicationConfig(boundary_margin_mm=60)
        planner = Explorer(world, cfg)
        planner.set_transient_obstacles([(300, 300, 25)])
        self.assertEqual(len(planner.transient_obstacles), 1)
        # Cozmo半対角 + Cube半対角 + 安全余白なので、単なる25mmより大きい。
        self.assertGreater(planner.transient_obstacles[0][2], 90)


class ResultFolderTests(unittest.TestCase):
    def test_prestart_failure_folder_is_archived(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = ApplicationConfig(setup_path=root/'setup.json', robot_id='Cozmo-A')
            data = setup()
            output = root/'results'/data['trial_id']/'Cozmo-A'
            output.mkdir(parents=True)
            (output/'events.jsonl').write_text(
                json.dumps({'event':'error','reason':'missing dependency'})+'\n',
                encoding='utf-8',
            )

            result = prepare_output_for_run(data, config)

            self.assertEqual(result, output)
            self.assertFalse(output.exists())
            archived = list((output.parent/'_failed_attempts').glob('Cozmo-A_*'))
            self.assertEqual(len(archived), 1)
            self.assertTrue((archived[0]/'events.jsonl').exists())

    def test_started_trial_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config = ApplicationConfig(setup_path=root/'setup.json', robot_id='Cozmo-A')
            data = setup()
            output = root/'results'/data['trial_id']/'Cozmo-A'
            output.mkdir(parents=True)
            (output/'events.jsonl').write_text(
                json.dumps({'event':'start','started_utc':'2026-09-16T00:00:00+00:00'})+'\n',
                encoding='utf-8',
            )

            with self.assertRaisesRegex(RuntimeError, '3:次の試行ID'):
                prepare_output_for_run(data, config)

            self.assertTrue(output.exists())
            self.assertFalse((output.parent/'_failed_attempts').exists())


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        s = setup()
        self.a,self.b = World(s,'Cozmo-A'),World(s,'Cozmo-B')
        self.na,self.nb = Network(),Network()
        self.la,self.lb = [],[]
        self.ea = EncounterExchange(self.a,self.na,log=lambda event,**kw:self.la.append(event))
        self.eb = EncounterExchange(self.b,self.nb,log=lambda event,**kw:self.lb.append(event))

    def test_one_sided_detection_exchanges_both_directions(self):
        self.a.add_cube(400,300,1); self.b.add_cube(700,400,1)
        self.ea.seen('Cozmo-B',2,0)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],1)
        self.ea.receive('Cozmo-B',self.nb.messages.pop()[1],2)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],3)
        self.assertEqual(self.a.cubes(),self.b.cubes())
        self.assertEqual(len(self.a.cubes()),2)
        self.assertIn('exchange_success',self.la)
        self.assertIn('exchange_success',self.lb)

    def test_no_send_without_marker(self):
        self.ea.tick(1)
        self.assertEqual(self.na.messages,[])

    def test_none_mode(self):
        self.ea.mode='none'
        self.ea.seen('Cozmo-B',2,1)
        self.assertFalse(self.na.messages)

    def test_wrong_trial_or_target_rejected(self):
        self.a.add_cube(400,300,1)
        self.ea.seen('Cozmo-B',2,0)
        msg = self.na.messages.pop()[1]
        msg['trial_id']='0'*32
        self.eb.receive('Cozmo-A',msg,1)
        self.assertFalse(self.b.records or self.nb.messages)
        self.assertIn('exchange_rejected',self.lb)

    def test_wrong_marker_rejected(self):
        self.ea.seen('Cozmo-B',99,0)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],1)
        self.assertFalse(self.nb.messages)

    def test_duplicate_request_and_ack_loss(self):
        self.a.add_cube(400,300,1)
        self.ea.seen('Cozmo-B',2,0)
        req = self.na.messages.pop()[1]
        self.eb.receive('Cozmo-A',req,1)
        response = self.nb.messages.pop()[1]
        self.ea.receive('Cozmo-B',response,2)
        self.na.messages.clear()  # ACKが失われた
        self.eb.receive('Cozmo-A',req,3)
        self.ea.receive('Cozmo-B',self.nb.messages.pop()[1],4)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],5)
        self.assertEqual(self.la.count('exchange_success'),1)
        self.assertEqual(self.lb.count('exchange_success'),1)
        self.assertEqual(len(self.b.cubes()),1)

    def test_receiver_retries_response_after_ack_lost(self):
        self.ea.seen('Cozmo-B',2,0)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],1)
        self.ea.receive('Cozmo-B',self.nb.messages.pop()[1],2)
        self.na.messages.clear()
        self.eb.tick(5)
        self.ea.receive('Cozmo-B',self.nb.messages.pop()[1],6)
        self.eb.receive('Cozmo-A',self.na.messages.pop()[1],7)
        self.assertIn('exchange_success',self.lb)

    def test_different_pose_setup_rejected(self):
        self.ea.seen('Cozmo-B',2,0)
        msg = self.na.messages.pop()[1]
        msg['setup_hash']='different'
        self.eb.receive('Cozmo-A',msg,1)
        self.assertIn('exchange_rejected',self.lb)

    def test_timeout_and_cooldown(self):
        self.ea.seen('Cozmo-B',2,0)
        self.ea.seen('Cozmo-B',2,1)
        self.assertEqual(len(self.na.messages),1)
        self.ea.tick(21)
        self.assertFalse(self.ea.pending)
        self.assertIn('exchange_timeout',self.la)

    def test_fragmented_frames_and_bad_json(self):
        decoder = LineDecoder()
        self.assertEqual(decoder.feed(b'{"a":'),[])
        self.assertEqual(decoder.feed(b'1}\nBAD\n{"b":2}\n'),[{'a':1},{'b':2}])
        self.assertEqual(decoder.feed(b'\n'),[])

    def test_oversize_recovery(self):
        decoder = LineDecoder()
        self.assertEqual(decoder.feed(b'x'*(1024*1024+1)),[])
        self.assertEqual(decoder.feed(b'\n{"ok":true}\n'),[{'ok':True}])


class SetupTests(unittest.TestCase):
    def test_json_roundtrip_and_condition_propagation(self):
        with tempfile.TemporaryDirectory() as d:
            config = ApplicationConfig(setup_path=Path(d)/'setup.json')
            save_json(config.setup_path,setup())
            data = load_setup(config)
            self.assertEqual(data['robots']['Cozmo-B']['theta'],math.pi)
            self.assertEqual(config.boundary_margin_mm,60)

    def test_setup_accepts_initial_pose_near_edge(self):
        with tempfile.TemporaryDirectory() as d:
            data = setup()
            data['robots']['Cozmo-A']['x'] = 10
            data['robots']['Cozmo-A']['y'] = 400
            config = ApplicationConfig(setup_path=Path(d)/'setup.json')
            save_json(config.setup_path,data)
            loaded = load_setup(config)
            self.assertEqual(loaded['robots']['Cozmo-A']['x'],10)

    def test_invalid_com(self):
        with self.assertRaises(ValueError): validate_ports(ApplicationConfig(),setup())

    def test_duplicate_markers(self):
        with self.assertRaises(ValueError):
            validate_config(ApplicationConfig(robot_markers={'Cozmo-A':1,'Cozmo-B':1}))

    def test_coordinate_rotation(self):
        # SLAMの実メソッドをASTで抽出し、重い機器依存なしで回転・平行移動を検証。
        import ast
        source = Path('algorithm/mapping/visual_slam.py').read_text()
        module = ast.parse(source)
        cls = next(n for n in module.body if isinstance(n,ast.ClassDef))
        methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in
                   ('_wrap_angle','_get_estimated_pose','_set_estimated_pose')]
        extracted = ast.Module(body=[ast.ClassDef(name='Transform',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        ns = {'math':math}
        exec(compile(ast.fix_missing_locations(extracted),'transform','exec'),ns)
        t = ns['Transform']()
        t._set_estimated_pose(10,20,0,300,400,math.pi/2)
        x,y,theta = t._get_estimated_pose(110,20,0)
        self.assertAlmostEqual(x,300); self.assertAlmostEqual(y,500)
        self.assertAlmostEqual(theta,math.pi/2)

    def test_safety_latches_and_no_restart(self):
        from research.safety import SafetyClient
        import threading
        raw = types.SimpleNamespace(calls=[])
        raw.drive_wheels = lambda *args,**kw:raw.calls.append(args)
        s = SafetyClient.__new__(SafetyClient)
        s.raw = raw; s.lock = threading.RLock(); s.latched_reason=''; s.moving=True
        s.on_state(raw,types.SimpleNamespace(status=0x4000,battery_voltage=4))
        self.assertEqual(raw.calls[-1],(0,0))
        with self.assertRaises(RuntimeError): s.drive_wheels(30,30)


if __name__=='__main__':
    unittest.main()


class ReturnHomeTests(unittest.TestCase):
    def test_cube_physical_size_is_50mm(self):
        cfg = ApplicationConfig()
        self.assertEqual(cfg.cube_width_mm, 50.0)

    def test_planner_can_return_to_edge_initial_pose(self):
        data = {
            'schema_version': 1,
            'map_id': 'return-map',
            'trial_id': 'return-trial',
            'landmarks': {
                1: (0, 0),
                2: (1000, 0),
                3: (1000, 800),
                4: (0, 800),
            },
            'robots': {
                'Cozmo-A': {'x': 10, 'y': 400, 'theta': math.pi / 2, 'marker_id': 1},
            },
            'cell_mm': 50,
            'margin_mm': 70,
            'cube_merge_mm': 60,
        }
        cfg = ApplicationConfig(boundary_margin_mm=70)
        world = World(data, 'Cozmo-A')
        planner = Explorer(world, cfg)

        self.assertTrue(planner.start_return((500, 400, 0), (10, 400, math.pi / 2)))
        self.assertEqual(planner.return_path[-1], (10.0, 400.0))

        # 帰還経路を通り終え、初期位置まで戻った状態を再現する。
        planner.return_path = [(10.0, 400.0)]
        command = planner.return_command((10, 400, 0))
        self.assertIsNotNone(command)
        self.assertLess(command[0] * command[1], 0)

        # 位置・向きがそろえば帰還完了。
        self.assertIsNone(planner.return_command((10, 400, math.pi / 2)))

    def test_return_final_segment_allows_edge_home_but_rejects_cube(self):
        grid = Grid([(0, 0), (1000, 0), (1000, 800), (0, 800)], 50, 70)
        safe = (75, 425)
        home = (10, 400)
        self.assertTrue(grid.free_map_segment(safe, home, []))
        self.assertFalse(grid.free_map_segment(safe, home, [(40, 410, 25)]))


class ResultViewerTests(unittest.TestCase):
    def test_find_and_restore_saved_result_without_robot_connection(self):
        from research.result_viewer import find_results, _load_world

        data = setup()
        world = World(data, 'Cozmo-A')
        world.add_cube(400, 300, 1.2)
        sample_cell = next(iter(world.grid.cells))
        world.observe((150, 150, 0), {sample_cell})
        exported = world.export()
        exported['metrics'] = {
            'elapsed_seconds': 12.3,
            'covered_cells': len(world.covered),
            'total_cells': len(world.grid.cells),
            'reason': 'test',
            'robot_count': 1,
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            directory = root / data['trial_id'] / 'Cozmo-A'
            directory.mkdir(parents=True)
            save_json(directory / 'setup.json', data)
            save_json(directory / 'exploration.json', exported)

            entries = find_results(root)
            self.assertEqual(len(entries), 1)
            restored_setup, restored_data, restored_world = _load_world(entries[0])

            self.assertEqual(restored_setup['trial_id'], data['trial_id'])
            self.assertEqual(restored_data['robot_id'], 'Cozmo-A')
            self.assertEqual(len(restored_world.cubes()), 1)
            self.assertEqual(restored_world.covered, world.covered)
            self.assertEqual(restored_world.trails, world.trails)

    def test_result_viewer_skips_result_without_saved_setup(self):
        from research.result_viewer import find_results

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'results' / 'trial' / 'Cozmo-A'
            directory.mkdir(parents=True)
            save_json(directory / 'exploration.json', {'schema_version': 1})
            self.assertEqual(find_results(Path(tmp) / 'results'), [])

class ImmediateCubeRegistrationTests(unittest.TestCase):
    def test_first_camera_detection_becomes_map_cube_and_obstacle(self):
        """1フレームの検知だけで50mmキューブを永続登録し、回避対象にする。"""
        data = setup()
        cfg = ApplicationConfig(boundary_margin_mm=60, cube_width_mm=50, cube_merge_mm=45)
        session = __import__('research.session', fromlist=['ResearchSession']).ResearchSession.__new__(
            __import__('research.session', fromlist=['ResearchSession']).ResearchSession
        )
        session.world = World(data, 'Cozmo-A')
        session.started = time.monotonic()
        session.events = types.SimpleNamespace(write=lambda *args: None, flush=lambda: None)
        session.log = lambda *args, **kwargs: None

        session._register_detected_cubes([(400, 300)])
        self.assertEqual(len(session.world.cubes()), 1)
        self.assertEqual(session.world.cubes()[0]['x'], 400)
        self.assertEqual(session.world.cubes()[0]['y'], 300)

        # 同じキューブを連続検知しても重複登録しない。
        session._register_detected_cubes([(410, 305)])
        self.assertEqual(len(session.world.cubes()), 1)

        planner = Explorer(session.world, cfg)
        obstacles = planner.obstacles()
        self.assertEqual(len(obstacles), 1)
        self.assertAlmostEqual(obstacles[0][0], 400)
        self.assertAlmostEqual(obstacles[0][1], 300)
        self.assertGreater(obstacles[0][2], 90)
