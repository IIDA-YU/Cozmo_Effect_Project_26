"""Cozmoアプリ全体の起動順序と各機能のライフサイクルを管理する。"""

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from algorithm.mapping.visual_slam import (
    DEFAULT_CHESS_MODEL_PATH,
    run_slam_mapping,
)
from camera.aruco_tracker import ArucoCoordinateTracker
from camera.camera_movie import CameraSettings, CameraViewer
from camera.cozmo_tracker import CozmoTracker
from camera.cube_tracker import CubeTracker
from connect.network_check import cozmo_session
from hardware.battery import BatteryMonitor
from hardware.collision_prevention import CollisionPrevention
from hardware.roaming_controller import RoamingController
from hardware.step_detection import StepDetector
from performance.facial_expression import FaceManager
from telecommunication.coordinate_sender import CoordinateSender
from telecommunication.pc_communication import CoordCommunication


@dataclass
class ApplicationConfig:
    cozmo_ssid: str
    return_ssid: str
    robot_id: str = "Cozmo-A"
    target_pc_ip: str = "127.0.0.1"
    listen_port: int = 50007
    chess_model_path: Path = DEFAULT_CHESS_MODEL_PATH
    camera_settings: CameraSettings = field(
        default_factory=lambda: CameraSettings(
            exposure_ms=5,
            gain=1.0,
            auto_exposure=False,
        )
    )
    target_ip_dict: dict = field(default_factory=dict)


def _start_module(module_obj):
    """モジュールごとの起動メソッド名の違いを吸収する。"""
    for method_name in ("start", "run", "begin", "activate"):
        method = getattr(module_obj, method_name, None)
        if callable(method):
            method()
            return
    raise AttributeError(
        f"{module_obj.__class__.__name__} に起動メソッド"
        "(start/run/begin/activate)がありません。"
    )


def _stop_module(module_obj):
    if module_obj is None:
        return
    stop = getattr(module_obj, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception as error:
            print(
                f"⚠️ {module_obj.__class__.__name__}の停止中にエラー: "
                f"{error}"
            )


def run_autonomous_mode(cli, config, battery, camera_viewer):
    """自律走行・各検出器・通信を起動し、カメラGUIを同じスレッドで更新する。"""
    print("\n🔧 自律走行モードを起動します...")
    started_modules = []
    receiver = None

    try:
        face_manager = FaceManager(cli)
        _start_module(face_manager)
        started_modules.append(face_manager)

        preventer = CollisionPrevention(
            cli,
            face_manager=face_manager,
            camera_viewer=camera_viewer,
        )
        _start_module(preventer)
        started_modules.append(preventer)

        step_detector = StepDetector(cli, preventer)
        _start_module(step_detector)
        started_modules.append(step_detector)

        cozmo_tracker = CozmoTracker(
            cli,
            collision_preventer=preventer,
            camera_viewer=camera_viewer,
        )
        _start_module(cozmo_tracker)
        started_modules.append(cozmo_tracker)

        cube_tracker = CubeTracker(
            cli,
            collision_preventer=preventer,
            camera_viewer=camera_viewer,
        )
        _start_module(cube_tracker)
        started_modules.append(cube_tracker)

        aruco_tracker = ArucoCoordinateTracker(
            cli,
            target_ip_dict=config.target_ip_dict,
            camera_viewer=camera_viewer,
        )
        _start_module(aruco_tracker)
        started_modules.append(aruco_tracker)

        # 後から起動した検出器にカメラ設定を変更されても、SLAM時の値へ戻す。
        camera_viewer.ensure_camera_settings(announce=True)

        receiver = CoordCommunication(port=config.listen_port)
        receiver_thread = threading.Thread(
            target=receiver.start_server,
            name="CoordinateReceiver",
            daemon=True,
        )
        receiver_thread.start()
        time.sleep(1.0)

        coord_sender = CoordinateSender(
            cli,
            target_ip=config.target_pc_ip,
            robot_id=config.robot_id,
            interval=0.2,
        )
        _start_module(coord_sender)
        started_modules.append(coord_sender)

        roamer = RoamingController(cli, preventer)
        _start_module(roamer)
        started_modules.append(roamer)

        print("\n🚀 自律走行・双方向通信・機体識別を開始しました！")
        print("💡 B:明るく / N:暗く / M:自動・手動 / 0:初期値 / Q:終了")

        while True:
            camera_viewer.display_once()
            key = camera_viewer.poll_key(delay_ms=10)
            if key in (ord("q"), 27):
                print("\n🚪 自律走行を終了します...")
                break
            time.sleep(0.01)
    finally:
        print("\n🛑 自律走行モジュールを停止中...")
        for module_obj in reversed(started_modules):
            _stop_module(module_obj)
        if receiver is not None and hasattr(receiver, "running"):
            receiver.running = False


def run_robot_mode(cli, config):
    """カメラを先に共有起動し、SLAMから自律走行へ引き継ぐ。"""
    cli.conn_lock = threading.Lock()
    battery = BatteryMonitor(cli)
    camera_viewer = CameraViewer(
        cli,
        battery_monitor=battery,
        enhance_image=True,
        camera_settings=config.camera_settings,
    )

    try:
        _start_module(battery)
        # SLAM側と同じメインスレッドで2画面を更新するため、表示スレッドは作らない。
        camera_viewer.start(background_display=False)
        run_slam_mapping(
            cli,
            model_path=str(config.chess_model_path),
            camera_settings=config.camera_settings,
            camera_viewer=camera_viewer,
        )
        run_autonomous_mode(
            cli,
            config=config,
            battery=battery,
            camera_viewer=camera_viewer,
        )
    finally:
        camera_viewer.stop()
        _stop_module(battery)


def run_receive_debug(config):
    receiver = CoordCommunication(port=config.listen_port)
    receiver.start_server()


def run_send_debug(config):
    communication = CoordCommunication()
    test_ip = input(
        "送信先PCのIPアドレスを入力 (自分のPCなら 127.0.0.1): "
    ).strip()
    print(f"\n{test_ip} へテスト送信を開始します...")
    for index in range(10):
        success = communication.send_data(
            test_ip,
            index * 10.0,
            index * 20.0,
            robot_id=config.robot_id,
        )
        result = "✅ 成功" if success else "❌ 失敗"
        print(f"送信 {index + 1}/10: {result}")
        time.sleep(1.0)


def run_application(config):
    """main.pyから呼び出すアプリケーションの単一エントリーポイント。"""
    try:
        choice = input(
            "\nモードの選択 (1:本番 / 2:受信デバッグ / 3:送信デバッグ): "
        ).strip()

        if choice == "1":
            with cozmo_session(config.cozmo_ssid, config.return_ssid) as cli:
                run_robot_mode(cli, config)
        elif choice == "2":
            run_receive_debug(config)
        elif choice == "3":
            run_send_debug(config)
        else:
            print("無効な選択です。プログラムを終了します。")
    except KeyboardInterrupt:
        print("\nユーザー操作によりシステムを終了しました。")
    except Exception as error:
        print(f"\nエラー：実行中に問題が発生しました。({error})")
        traceback.print_exc()
