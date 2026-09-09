"""各機能の起動順序と終了処理を管理する。

起動はプロジェクト直下のmain.pyから行う。
接続先などの設定はapplication/config.pyで変更する。
cliは、接続済みのCozmoを操作するためのオブジェクト。
画像認識・走行判断・通信の具体的な処理は、それぞれのモジュールに任せる。
"""

import threading
import time
import traceback

from application.config import CONFIG

# 既存の機能別フォルダから、使用する機能を読み込む。
from algorithm.mapping.visual_slam import run_slam_mapping
from camera.aruco_tracker import ArucoCoordinateTracker
from camera.camera_movie import CameraViewer
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


# ---------------------------------------------------------
# アプリの入口：モードを選択する
# ---------------------------------------------------------
def run_application():
    """main.pyから呼び出すアプリケーションの単一エントリーポイント。"""
    config = CONFIG

    try:
        choice = input(
            "\nモードの選択 (1:本番 / 2:受信デバッグ / 3:送信デバッグ): "
        ).strip()

        # 本番モードだけCozmoへ接続する。切断・Wi-Fi復帰は接続モジュールに任せる。
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


# ---------------------------------------------------------
# 本番モード：SLAMから自律走行へ進む
# ---------------------------------------------------------
def run_robot_mode(cli, config):
    """カメラを先に共有起動し、SLAMから自律走行へ引き継ぐ。"""
    # 各機能が機体との通信に共用できるロックを用意する。
    cli.conn_lock = threading.Lock()
    battery = None
    camera_viewer = None

    try:
        # カメラとバッテリー監視は、SLAMから自律走行まで共用する。
        battery = BatteryMonitor(cli)
        camera_viewer = CameraViewer(
            cli,
            battery_monitor=battery,
            enhance_image=True,
            camera_settings=config.camera_settings,
        )
        _start_module(battery)
        # SLAM側と同じメインスレッドで2画面を更新するため、表示スレッドは作らない。
        camera_viewer.start(background_display=False)
        # 1. SLAMを実行する。2画面の更新にはSLAM側の対応も必要。
        run_slam_mapping(
            cli,
            model_path=str(config.chess_model_path),
            camera_settings=config.camera_settings,
            camera_viewer=camera_viewer,
        )
        # 2. SLAMの関数が戻ったら、同じカメラで自律走行へ進む。
        #    この構成ではSLAMと自律走行を順番に実行する。
        run_autonomous_mode(
            cli,
            config=config,
            camera_viewer=camera_viewer,
        )
    finally:
        # カメラの停止に失敗しても、バッテリー監視の停止処理へ進む。
        _stop_module(camera_viewer)
        _stop_module(battery)


# ---------------------------------------------------------
# 自律走行：検知・通信・走行を組み合わせる
# ---------------------------------------------------------
def run_autonomous_mode(cli, config, camera_viewer):
    """自律走行・各検出器・通信を起動し、カメラGUIを同じスレッドで更新する。"""
    print("\n🔧 自律走行モードを起動します...")
    # 起動に成功した機能を記録し、終了時に逆順で停止する。
    started_modules = []
    receiver = None

    try:
        # 1. 表情と回避機能を準備する（後の検知機能が参照する）。
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

        # 2. 段差・機体・対象物・ArUcoマーカーの検知を起動する。
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

        # 3. 座標の受信を別スレッドで開始する。
        receiver = CoordCommunication(port=config.listen_port)
        receiver_thread = threading.Thread(
            target=receiver.start_server,
            name="CoordinateReceiver",
            daemon=True,
        )
        receiver_thread.start()
        time.sleep(1.0)  # 元コードと同じ起動待ち。受信準備完了の確認ではない。

        # 自分の座標を0.2秒間隔で送信する。
        coord_sender = CoordinateSender(
            cli,
            target_ip=config.target_pc_ip,
            robot_id=config.robot_id,
            interval=0.2,
        )
        _start_module(coord_sender)
        started_modules.append(coord_sender)

        # 4. 検知・回避・通信の起動後に、自律走行を開始する。
        roamer = RoamingController(cli, preventer)
        _start_module(roamer)
        started_modules.append(roamer)

        print("\n🚀 自律走行・双方向通信・機体識別を開始しました！")
        print("💡 B:明るく / N:暗く / M:自動・手動 / 0:初期値 / Q・Esc:終了")

        # 5. メインスレッドで画面を更新する。キー処理の詳細はカメラ側に任せる。
        while True:
            camera_viewer.display_once()
            key = camera_viewer.poll_key(delay_ms=10)
            if key in (ord("q"), 27):  # qキー、またはEscキー
                print("\n🚪 自律走行を終了します...")
                break
            time.sleep(0.01)
    finally:
        print("\n🛑 自律走行モジュールを停止中...")
        for module_obj in reversed(started_modules):
            _stop_module(module_obj)
        # 元コードの受信停止要求を維持する。実際の停止方法は通信側に依存する。
        if receiver is not None and hasattr(receiver, "running"):
            receiver.running = False


# ---------------------------------------------------------
# 通信テスト：Cozmoに接続せず送受信を確認する
# ---------------------------------------------------------
def run_receive_debug(config):
    """Cozmoに接続せず、他PCからの座標を受信する。"""
    receiver = CoordCommunication(port=config.listen_port)
    receiver.start_server()


def run_send_debug(config):
    """指定したPCへ、ダミー座標を1秒間隔で10回送信する。"""
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


# ---------------------------------------------------------
# 共通処理：各機能を起動・停止する
# ---------------------------------------------------------
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
    """停止メソッドがあれば呼ぶ。停止エラーが出ても残りの後処理を続ける。"""
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
