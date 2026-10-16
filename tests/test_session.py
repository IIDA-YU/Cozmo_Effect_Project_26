"""モック機体で、待機→姿勢確定→探索→記録→停止の接続を検証する。"""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from application.config import ApplicationConfig
from application.controller import _make_experiment_setup
from research.session import ResearchSession
from test_research import setup


class Clock:
    def __init__(self): self.now=100
    def monotonic(self): return self.now
    def sleep(self,seconds): self.now+=seconds


class Camera:
    def __init__(self,clock): self.clock=clock;self.count=0
    def get_frame_sample(self):
        self.count+=1
        return object(),self.count,self.clock.now
    def display_once(self): pass
    def set_detections(self,*args): pass
    def poll_key(self,delay): self.clock.now+=.1;return 13


class Window:
    def __init__(self,*args): self.closed=False
    def draw(self,*args,**kw): pass
    def close(self): self.closed=True


class Network:
    def __init__(self,*args): self.started=False;self.stopped=False;self.status={};self.links={}
    def start(self): self.started=True
    def stop(self): self.stopped=True
    def receive(self): return []
    def send(self,*args): return True


class Perception:
    def __init__(self,*args): self.calls=0;self.robot_in_view=False
    def detect(self,*args): self.calls+=1;return [],[],[],[]


class SessionTests(unittest.TestCase):
    def test_enter_starts_immediately_then_explore_then_cleanup(self):
        clock=Clock();camera=Camera(clock)
        raw_pose=types.SimpleNamespace(position=types.SimpleNamespace(x=0,y=0),
                                       rotation=types.SimpleNamespace(angle_z=types.SimpleNamespace(radians=0)))
        cli=types.SimpleNamespace(pose=raw_pose,latched_reason='',commands=[],acknowledgements=0)
        cli.drive_wheels=lambda *args:cli.commands.append(args)
        cli.set_head_angle=lambda *args:None
        cli.set_lift_height=lambda *args:None
        def ack(): cli.acknowledgements+=1
        cli.acknowledge_placement=ack
        slam=types.SimpleNamespace(offset=None)
        def set_pose(*args): slam.offset=args
        slam._set_estimated_pose=set_pose
        slam._get_estimated_pose=lambda *args:(150,150,0)
        with tempfile.TemporaryDirectory() as directory:
            cfg=ApplicationConfig(setup_path=Path(directory)/'setup.json',placement_wait_seconds=2,
                                  trial_limit_seconds=2,sharing_mode='none',boundary_margin_mm=60)
            with patch('research.session.time',clock),patch('research.session.ResearchPerception',Perception),\
                 patch('research.session.MapWindow',Window),patch('research.session.BluetoothNetwork',Network):
                s=ResearchSession(cli,camera,slam,setup(),cfg,types.SimpleNamespace(percent=80))
                s.run()
                # placement_wait_seconds=2 でも待たず、Enterですぐ開始する。
                self.assertLess(s.started,101)
                self.assertEqual(cli.acknowledgements,1)
                self.assertEqual(slam.offset[-3:],(150,150,0))
                self.assertGreater(s.perception.calls,0)
                self.assertTrue(s.network.started and s.network.stopped and s.window.closed)
                self.assertEqual(cli.commands[-1],(0,0))
                data=json.loads((s.output/'exploration.json').read_text())
                self.assertEqual(data['metrics']['reason'],'time_limit')
                self.assertFalse(data['cubes'])
                events=[json.loads(line) for line in (s.output/'events.jsonl').read_text().splitlines()]
                self.assertEqual(events[0]['event'],'start')
                self.assertEqual(events[-1]['event'],'end')

    def test_motion_burst_repeats_wheel_command(self):
        clock=Clock();camera=Camera(clock)
        raw_pose=types.SimpleNamespace(position=types.SimpleNamespace(x=0,y=0),
                                       rotation=types.SimpleNamespace(angle_z=types.SimpleNamespace(radians=0)))
        cli=types.SimpleNamespace(pose=raw_pose,latched_reason='',commands=[])
        cli.drive_wheels=lambda *args:cli.commands.append(args)
        cli.set_head_angle=lambda *args:None
        cli.set_lift_height=lambda *args:None
        cli.acknowledge_placement=lambda:None
        slam=types.SimpleNamespace()
        slam._set_estimated_pose=lambda *args:None
        slam._get_estimated_pose=lambda *args:(150,150,0)
        with tempfile.TemporaryDirectory() as directory:
            cfg=ApplicationConfig(
                setup_path=Path(directory)/'setup.json',
                motion_pulse_seconds=.15,
                sharing_mode='none',
                boundary_margin_mm=60,
            )
            with patch('research.session.time',clock),patch('research.session.ResearchPerception',Perception),\
                 patch('research.session.MapWindow',Window),patch('research.session.BluetoothNetwork',Network):
                session=ResearchSession(cli,camera,slam,_make_experiment_setup(setup(),cfg,1),cfg,
                                        types.SimpleNamespace(percent=80))
                session.started=clock.now
                session._run_motion_burst((-12,12))
                moving=[cmd for cmd in cli.commands if cmd != (0,0)]
                self.assertGreaterEqual(len(moving),2)
                self.assertTrue(all(abs(cmd[0])>=25 and abs(cmd[1])>=25 for cmd in moving))
                session.events.close()
                session.window.close()

    def test_single_robot_full_mode_starts_without_bluetooth_peer(self):
        clock=Clock();camera=Camera(clock)
        raw_pose=types.SimpleNamespace(position=types.SimpleNamespace(x=0,y=0),
                                       rotation=types.SimpleNamespace(angle_z=types.SimpleNamespace(radians=0)))
        cli=types.SimpleNamespace(pose=raw_pose,latched_reason='',commands=[],acknowledgements=0)
        cli.drive_wheels=lambda *args:cli.commands.append(args)
        cli.set_head_angle=lambda *args:None
        cli.set_lift_height=lambda *args:None
        cli.acknowledge_placement=lambda:None
        slam=types.SimpleNamespace(offset=None)
        slam._set_estimated_pose=lambda *args:setattr(slam,'offset',args)
        slam._get_estimated_pose=lambda *args:(150,150,0)
        with tempfile.TemporaryDirectory() as directory:
            cfg=ApplicationConfig(
                setup_path=Path(directory)/'setup.json',
                placement_wait_seconds=0,
                trial_limit_seconds=.5,
                sharing_mode='full',
                boundary_margin_mm=60,
                bluetooth_ports={'Cozmo-B':'CHANGE_ME'},
            )
            runtime_setup=_make_experiment_setup(setup(),cfg,1)
            with patch('research.session.time',clock),patch('research.session.ResearchPerception',Perception),\
                 patch('research.session.MapWindow',Window),patch('research.session.BluetoothNetwork',Network):
                session=ResearchSession(cli,camera,slam,runtime_setup,cfg,types.SimpleNamespace(percent=80))
                session.run()
                data=json.loads((session.output/'exploration.json').read_text())
                self.assertEqual(data['metrics']['robot_count'],1)
                self.assertFalse(data['metrics']['bluetooth_enabled'])
                self.assertEqual(data['metrics']['sharing_mode'],'full')

