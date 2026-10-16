"""実験後だけ使う集計。ロボットへ集計結果を渡さない（中央制御にしない）。
使い方: python -m tools.summarize_trial results/試行ID
全PCの同じ試行IDのフォルダを1か所へ集めて実行する。
"""
import argparse
import csv
import datetime
import json
from pathlib import Path
from research.world import World


def summarize(root):
    files = sorted(Path(root).glob('*/exploration.json'))
    if not files:
        raise ValueError('機体ごとのexploration.jsonがありません')
    rows,all_events,starts = [],[],{}
    common_setup = None
    team = None
    coverages = []
    for file in files:
        data = json.loads(file.read_text(encoding='utf-8'))
        setup = json.loads((file.parent/'setup.json').read_text(encoding='utf-8'))
        if common_setup is None:
            common_setup = setup
            team = World(setup,next(iter(setup['robots'])))
        elif setup!=common_setup:
            raise ValueError('異なる初期設定・試行の結果が含まれています')
        rid,m = data['robot_id'],data['metrics']
        if rid in starts:
            raise ValueError('機体の結果が重複しています')
        if not m['started_utc']:
            raise ValueError('開始前に終了した結果が含まれています')
        starts[rid] = datetime.datetime.fromisoformat(m['started_utc']).timestamp()
        coverage = set(data['local_covered']) | set(data['trails'].get(rid,[]))
        coverages.append(coverage)
        events = [json.loads(line) for line in (file.parent/'events.jsonl').read_text(encoding='utf-8').splitlines() if line]
        all_events.extend((starts[rid]+e['elapsed'],rid,e) for e in events)
        rows.append(dict(robot_id=rid,elapsed_seconds=m['elapsed_seconds'],distance_mm=m['distance_mm'],
                         first_all_known_seconds=m['first_all_known_seconds'],
                         first_local_all_seconds=m['first_local_all_seconds'],
                         final_known_cubes=len(data['cubes']),local_coverage_cells=len(coverage),
                         exchange_success=sum(e['event']=='exchange_success' for e in events),
                         exchange_timeout=sum(e['event']=='exchange_timeout' for e in events),
                         interventions=m['interventions'],reason=m['reason']))
    if set(starts)!=set(common_setup['robots']):
        raise ValueError('全参加機体の結果をそろえてください。予定='+str(sorted(common_setup['robots'])))
    t0 = min(starts.values())
    first_team = None
    expected = common_setup['conditions']['expected_cubes']
    for stamp,rid,event in sorted(all_events,key=lambda e:e[0]):
        if event['event']=='cube_discovered':
            key = event['cube_id']
            team.records[key] = dict(x=event['x'],y=event['y'],source=rid,elapsed=event['elapsed'])
            if expected>0 and len(team.cubes())>=expected and first_team is None:
                first_team = stamp-t0
    union = set().union(*coverages)
    total = sum(map(len,coverages))
    everyone = [starts[r['robot_id']]+r['first_all_known_seconds']-t0 for r in rows if r['first_all_known_seconds'] is not None]
    report = dict(trial_id=common_setup['trial_id'],mode=common_setup['conditions']['sharing_mode'],
                  robots=len(rows),start_skew_seconds=max(starts.values())-t0,
                  first_team_found_all_seconds=first_team,
                  first_everyone_known_all_seconds=max(everyone) if len(everyone)==len(rows) else None,
                  total_distance_mm=sum(r['distance_mm'] for r in rows),
                  team_observed_cells=len(union),total_cells=len(team.grid.cells),
                  overlap_ratio=(total-len(union))/total if total else 0,
                  final_team_cube_count=len(team.cubes()),
                  note='位置照合による推定集計。発見率・位置誤差は現物の正解配置と照合。時刻はPC時計の同期が前提。')
    return rows,report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trial_directory',type=Path)
    args = parser.parse_args()
    rows,report = summarize(args.trial_directory)
    for name,data in (('robot_metrics.csv',rows),('team_metrics.csv',[report])):
        with (args.trial_directory/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
