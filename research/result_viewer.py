"""保存済み探索結果を、Cozmoへ接続せず地図上で確認するモジュール。

初期メニューの「5:検知結果・マップ確認」から使用する。
Wi-Fi切替、Cozmo接続、カメラ、Bluetoothは一切起動せず、results/ に
保存された setup.json と exploration.json だけを読み込んで表示する。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from research.interface import MapWindow
from research.world import World


@dataclass(frozen=True)
class ResultEntry:
    """1台分の保存済み探索結果。"""

    trial_id: str
    robot_id: str
    directory: Path
    exploration_path: Path
    setup_path: Path
    modified_time: float


def find_results(results_root: Path) -> list[ResultEntry]:
    """results/ 以下から表示可能な探索結果を新しい順に探す。"""
    if not results_root.exists():
        return []

    entries = []
    for exploration_path in results_root.glob("*/*/exploration.json"):
        directory = exploration_path.parent
        setup_path = directory / "setup.json"
        if not setup_path.exists():
            # 地図情報がなければ安全に再現できないため一覧には出さない。
            continue

        robot_id = directory.name
        trial_id = directory.parent.name
        try:
            modified = exploration_path.stat().st_mtime
        except OSError:
            continue

        entries.append(
            ResultEntry(
                trial_id=trial_id,
                robot_id=robot_id,
                directory=directory,
                exploration_path=exploration_path,
                setup_path=setup_path,
                modified_time=modified,
            )
        )

    return sorted(entries, key=lambda item: item.modified_time, reverse=True)


def _read_json(path: Path) -> dict:
    """JSONを辞書として読み込む。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSONの形式が不正です: {path}")
    return data


def _load_world(entry: ResultEntry):
    """保存済みJSONから、MapWindowで描画できるWorldを復元する。"""
    setup = _read_json(entry.setup_path)
    exploration = _read_json(entry.exploration_path)

    robot_id = exploration.get("robot_id", entry.robot_id)
    if robot_id not in setup.get("robots", {}):
        # 古い結果で robot_id が省略されている場合はフォルダ名を使う。
        robot_id = entry.robot_id
    if robot_id not in setup.get("robots", {}):
        raise ValueError("探索結果のrobot_idがsetup.jsonに存在しません")

    world = World(setup, robot_id)

    # 補正後のチェス位置も再現する。基準setup/共有セルは読み込み時に変更しない。
    corrected = exploration.get('landmarks', {})
    if isinstance(corrected, dict):
        for key, point in corrected.items():
            if (key in world.landmarks and isinstance(point, (list, tuple)) and len(point) == 2
                    and all(type(v) in (int, float) and math.isfinite(v) for v in point)):
                world.landmarks[key] = tuple(point)

    # World.merge() を使うことで、通常のBluetooth共有と同じ検証を通して
    # キューブ・探索済みセル・軌跡を復元する。
    world.merge(
        {
            "records": exploration.get("records", {}),
            "covered": exploration.get("covered", []),
            "trails": exploration.get("trails", {}),
        }
    )

    local_covered = exploration.get("local_covered", [])
    if isinstance(local_covered, list):
        world.local_covered = {
            key for key in local_covered
            if isinstance(key, str) and key in world.grid.cells
        }

    return setup, exploration, world


def _summary(exploration: dict, world: World) -> str:
    """画面上部に表示する短い結果概要を作る。"""
    metrics = exploration.get("metrics", {})
    total = len(world.grid.cells)
    covered = len(world.covered)
    percent = 0.0 if total == 0 else 100.0 * covered / total
    cubes = len(world.cubes())

    elapsed = metrics.get("exploration_finished_seconds")
    if elapsed is None:
        elapsed = metrics.get("elapsed_seconds")
    try:
        elapsed_text = f"{float(elapsed):.1f}s"
    except (TypeError, ValueError):
        elapsed_text = "-"

    reason = metrics.get("completion_reason") or metrics.get("reason") or "-"
    return (
        f"RESULT / covered={covered}/{total} ({percent:.1f}%) / "
        f"cubes={cubes} / time={elapsed_text} / reason={reason}"
    )


def choose_result(entries: list[ResultEntry]) -> ResultEntry | None:
    """保存済み結果を番号で選択する。Enterだけなら最新結果。"""
    if not entries:
        print("表示できる探索結果がありません。results フォルダを確認してください。")
        return None

    print("\n=== 保存済み探索結果 ===")
    for index, entry in enumerate(entries, start=1):
        try:
            data = _read_json(entry.exploration_path)
            metrics = data.get("metrics", {})
            cubes = len(data.get("cubes", []))
            covered = len(data.get("covered", []))
            total = metrics.get("total_cells", "?")
            robot_count = metrics.get("robot_count", "?")
            reason = metrics.get("completion_reason") or metrics.get("reason") or "-"
            info = (
                f"{entry.robot_id} / {robot_count}台 / "
                f"探索={covered}/{total} / キューブ={cubes} / {reason}"
            )
        except Exception:
            info = f"{entry.robot_id} / 詳細読込不可"

        short_trial = entry.trial_id[:10]
        latest = "  ← 最新" if index == 1 else ""
        print(f"{index}: {short_trial}... / {info}{latest}")

    print("0: 戻る")

    while True:
        choice = input("表示する結果番号（Enter=最新） > ").strip()
        if choice == "":
            return entries[0]
        if choice == "0":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            return entries[int(choice) - 1]
        print("一覧にある番号を入力してください。")


def show_result(entry: ResultEntry, config) -> None:
    """選択した探索結果をOpenCVのマップ画面に表示する。"""
    setup, exploration, world = _load_world(entry)

    window = MapWindow(
        setup["landmarks"],
        robot_length_mm=config.robot_length_mm,
        robot_width_mm=config.robot_width_mm,
        cube_size_mm=config.cube_width_mm,
    )

    status = _summary(exploration, world)
    print("\n検知結果を表示します。Cozmoには接続していません。")
    print(f"試行ID: {entry.trial_id}")
    print(f"機体: {entry.robot_id}")
    print(f"検知キューブ数: {len(world.cubes())}")
    print("Q または Esc で閉じます。")

    try:
        while True:
            window.draw(
                world=world,
                localization=exploration.get("localization"),
                current=exploration.get("current_pose"),
                status=status,
                cube_candidates=exploration.get('cube_candidates', []),
                help_text="RESULT VIEWER / Q or Esc: close",
            )
            key = window.cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        window.close()


def run_result_viewer(config) -> None:
    """結果一覧を表示し、選択した結果をオフラインで開く。"""
    results_root = config.setup_path.parent / "results"
    entry = choose_result(find_results(results_root))
    if entry is None:
        return
    show_result(entry, config)
