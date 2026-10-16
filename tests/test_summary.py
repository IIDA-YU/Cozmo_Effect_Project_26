import json
import tempfile
import unittest
from pathlib import Path
from research.world import World,save_json
from tools.summarize_trial import summarize
from test_research import setup


class SummaryTests(unittest.TestCase):
    def test_team_discovery_and_all_robot_knowledge_are_distinct(self):
        s=setup()
        s['conditions']['expected_cubes']=2
        with tempfile.TemporaryDirectory() as directory:
            for rid,started,x in [('Cozmo-A','2026-09-16T00:00:00+00:00',300),
                                   ('Cozmo-B','2026-09-16T00:00:02+00:00',700)]:
                out=Path(directory)/rid
                world=World(s,rid)
                key=world.add_cube(x,300,5)
                world.observe((125,125,0),{'4,4'})
                data=world.export()
                data['metrics']=dict(elapsed_seconds=30,distance_mm=100,
                                     first_all_known_seconds=20,first_local_all_seconds=None,
                                     interventions=0,reason='time_limit',started_utc=started)
                save_json(out/'exploration.json',data)
                save_json(out/'setup.json',s)
                (out/'events.jsonl').write_text(json.dumps(dict(event='cube_discovered',elapsed=5,
                                                               cube_id=key,x=x,y=300))+'\n')
            rows,report=summarize(directory)
            self.assertEqual(report['first_team_found_all_seconds'],7)
            self.assertEqual(report['first_everyone_known_all_seconds'],22)
            self.assertEqual(report['total_distance_mm'],200)
            self.assertEqual(report['start_skew_seconds'],2)
            self.assertEqual(report['overlap_ratio'],0.5)
