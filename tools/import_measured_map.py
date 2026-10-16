"""実測した6駒のmm座標を基準地図へ採用する（写真から座標を推測しない）。

実行例: python tools/import_measured_map.py measured_landmarks.json
入力: {"landmarks": {"0": [x_mm, y_mm], ... , "5": [x_mm, y_mm]}}
実行後は必ずmain.pyの1でCozmo初期位置を設定し直す。
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.world import Grid, save_json


def validate_landmarks(data):
    points = data.get('landmarks')
    if not isinstance(points, dict) or set(points) != set(map(str, range(6))):
        raise ValueError('landmarksに文字列ID 0〜5の6点が必要です')
    for point in points.values():
        if (not isinstance(point, list) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError('各座標は有限の数値[x_mm, y_mm]で指定してください')
    if len({tuple(p) for p in points.values()}) != 6:
        raise ValueError('同じ座標に複数の駒を置けません')
    return points


def import_map(source, root=ROOT):
    root = Path(root)
    points = validate_landmarks(json.loads(Path(source).read_text(encoding='utf-8')))
    from application.config import ApplicationConfig
    cfg = ApplicationConfig()
    Grid(points.values(), cfg.cell_mm, cfg.boundary_margin_mm)
    map_path, setup_path = root/'slam_data.json', root/'setup.json'
    old_map = json.loads(map_path.read_text(encoding='utf-8')) if map_path.exists() else {}
    old_setup = json.loads(setup_path.read_text(encoding='utf-8')) if setup_path.exists() else None
    backup = root/'map_backups'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True, exist_ok=False)
    for path in (map_path, setup_path):
        if path.exists():
            shutil.copy2(path, backup/path.name)
    # 先に探索を禁止する印を付ける。地図保存途中で中断しても旧初期姿勢で走らない。
    if old_setup is not None:
        old_setup['initial_poses_require_review'] = True
        save_json(setup_path, old_setup)
    old_map.update(landmarks=points, trajectory=[], coordinate_source='measured_mm')
    save_json(map_path, old_map)
    print('実測座標を反映しました。main.pyの1で初期姿勢を再設定してください。')
    print('以前の地図・設定の控え:', backup)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    import_map(args.input)


if __name__ == '__main__':
    main()
