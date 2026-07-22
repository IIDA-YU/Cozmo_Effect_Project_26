import time, threading, traceback, pycozmo
from connect.network_check import cozmo_session
from hardware.battery import BatteryMonitor
from camera.camera_movie import CameraViewer
from performance.facial_expression import FaceManager
from hardware.collision_prevention import CollisionPrevention
from hardware.step_detection import StepDetector
from hardware.roaming_controller import RoamingController
from camera.cozmo_tracker import CozmoTracker
from camera.cube_tracker import CubeTracker
from algorithm.mapping.visual_slam import run_slam_mapping
from algorithm.virtual_pheromones.pheromone_navi_log import PheromoneTracker
from telecommunication.coordinate_sender import CoordinateSender
from telecommunication.pc_communication import CoordCommunication
from camera.aruco_tracker import ArucoCoordinateTracker
from camera.apriltag_tracker import ApriltagCoordinateTracker

# =========================================================
# 🚨 【環境設定】SSIDなどは全てここで変更してから実行 🚨
# アプリケーション全体で使う固定値（IPアドレスやポートなど）をまとめています。
# 5台のPCにこのコードを配る際、ここだけを書き換えれば動くように設計されています。
# =========================================================
COZMO_SSID = "Cozmo_42A832" # 接続するCozmoのSSIDを入力
RETURN_SSID = "R0347"       # プログラム終了後に戻る学校/自宅のWi-Fi

ROBOT_ID = "Cozmo-A"        # この機体の名前
TARGET_PC_IP = "127.0.0.1"  # 送信テスト用の宛先IP
LISTEN_PORT = 50007         # 座標データを受信するポート番号

# ArUcoマーカーのIDと、そのマーカーを付けている機体を操作するPCのIPアドレスの対応表
TARGET_IP_DICT = {
    1: "172.18.67.189",
    2: "172.18.67.198",
    3: "192.168.1.12",
    4: "192.168.1.13",
    5: "192.168.1.14"
}
# =========================================================

# =========================================================
# 📍 3. フェーズ2：自律走行とAI・通信モジュールの統合処理
# =========================================================
def run_autonomous_mode(cli):
    """
    Cozmoを自動で走らせながら、YOLO検知、マーカー検知、衝突回避、P2P通信を
    すべて同時に（マルチスレッドで）並行稼働させるメイン処理です。
    """
    print("\n🔧 自律走行モードを起動します...")
    cli.conn_lock = threading.Lock()

    def safe_start_module(module_obj):
        if hasattr(module_obj, "start"):
            module_obj.start()
        elif hasattr(module_obj, "run"):
            module_obj.run()
        elif hasattr(module_obj, "begin"):
            module_obj.begin()
        elif hasattr(module_obj, "activate"):
            module_obj.activate()
        else:
            print(f"⚠️ {module_obj.__class__.__name__} に起動メソッド(start/run/begin/activate)が見つかりません。")

    # --- ① 基礎システムの起動 〜 ③ AI画像認識システムの起動 ---
    started_modules = []
    init_steps = [
        ("BatteryMonitor",       lambda: BatteryMonitor(cli)),
        ("CameraViewer",         lambda: CameraViewer(cli, battery_monitor=battery)),
        ("FaceManager",          lambda: FaceManager(cli)),
        ("CollisionPrevention",  lambda: CollisionPrevention(cli, face_manager=face_manager, camera_viewer=camera_viewer)),
        ("StepDetector",         lambda: StepDetector(cli, preventer)),
        ("CozmoTracker",         lambda: CozmoTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)),
        ("CubeTracker",          lambda: CubeTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)),
        ("ArucoCoordinateTracker", lambda: ArucoCoordinateTracker(cli, target_ip_dict=TARGET_IP_DICT, camera_viewer=camera_viewer)),
        ("ApriltagCoordinateTracker", lambda: ApriltagCoordinateTracker(cli, target_ip_dict=TARGET_IP_DICT, camera_viewer=camera_viewer)),
    ]

    try:
        battery = init_steps[0][1]()
        safe_start_module(battery)
        started_modules.append(battery)

        camera_viewer = init_steps[1][1]()
        safe_start_module(camera_viewer)
        started_modules.append(camera_viewer)

        face_manager = init_steps[2][1]()
        safe_start_module(face_manager)
        started_modules.append(face_manager)

        preventer = init_steps[3][1]()
        safe_start_module(preventer)
        started_modules.append(preventer)

        step_detector = init_steps[4][1]()
        safe_start_module(step_detector)
        started_modules.append(step_detector)

        cozmo_tracker = init_steps[5][1]()
        safe_start_module(cozmo_tracker)
        started_modules.append(cozmo_tracker)

        cube_tracker = init_steps[6][1]()
        safe_start_module(cube_tracker)
        started_modules.append(cube_tracker)

        aruco_tracker = init_steps[7][1]()
        safe_start_module(aruco_tracker)
        started_modules.append(aruco_tracker)

        apriltag_tracker = init_steps[8][1]()
        safe_start_module(apriltag_tracker)
        started_modules.append(apriltag_tracker)

    except Exception as e:
        failed_name = init_steps[len(started_modules)][0]
        print(f"\n❌ 初期化中にエラー発生：直前に成功したのは "
              f"{started_modules[-1].__class__.__name__ if started_modules else '(なし)'}、"
              f"失敗したのはおそらく {failed_name} です。")
        print(f"エラー詳細: {e}")
        traceback.print_exc()
        # 起動済みのモジュールだけ、逆順で安全に停止しておく
        for m in reversed(started_modules):
            try:
                m.stop()
            except Exception:
                pass
        raise  # ここで再送出し、呼び出し元（cozmo_session）に切断処理を任せる

    # --- ④ P2P 双方向通信システムの起動 ---
    # 受信機：他PCから送られてくる座標を受け取るサーバーを裏側で立てる
    receiver = CoordCommunication(port=LISTEN_PORT)
    receiver_thread = threading.Thread(target=receiver.start_server, daemon=True)
    receiver_thread.start()
    time.sleep(1.0) # サーバーが立ち上がるのを少し待つ

    # 送信機：自分の現在座標を0.2秒ごとに指定した相手に送信し続ける
    coord_sender = CoordinateSender(cli, target_ip=TARGET_PC_IP, robot_id=ROBOT_ID, interval=0.2)
    safe_start_module(coord_sender)

    # --- ⑤ 自律走行コントローラーの起動 ---
    # 回避システム(preventer)の指示に従いながら、基本は前進し続ける処理
    roamer = RoamingController(cli, preventer)
    
    try:
        safe_start_module(roamer)
        print("\n🚀 自律走行および【双方向通信】【機体識別】を開始しました！")
        # メインスレッド（このプログラム自体）が終了しないように、ここで無限ループして待機
        while True:
            time.sleep(1)
    finally:
        # Ctrl+Cが押されたら、起動した全てのシステムを順番に安全に止める
        print("\n🛑 停止中...")
        try: roamer.stop() 
        except: pass
        try: coord_sender.stop()
        except: pass
        if hasattr(receiver, "running"):
            receiver.running = False
        try: aruco_tracker.stop()
        except: pass
        try: apriltag_tracker.stop()
        except: pass
        try: cube_tracker.stop()
        except: pass
        try: cozmo_tracker.stop()
        except: pass
        try: step_detector.stop()
        except: pass
        try: face_manager.stop()
        except: pass
        try: camera_viewer.stop()
        except: pass
        try: battery.stop()
        except: pass
    
if __name__ == "__main__":
# =========================================================
# ~~ まずここから実行されます（選択肢分岐は以下から） ~~
# 1: 🤖 ネット切り替え → マッピング → 自律走行 (本番用)
# 2: 💻 受信デバッグ (Cozmoに繋がず、相手からの座標を受信・表示するだけ)
# 3: 📡 送信デバッグ (Cozmoに繋がず、ダミーの座標を相手に送信テスト)
# =========================================================

    try:
        choice = input("\nモードの選択｛コメントアウト参照｝ (1 / 2 / 3): ").strip()
        
        if choice == '1':
            with cozmo_session(COZMO_SSID, RETURN_SSID) as cli:
                run_slam_mapping(cli)       # マッピング（SLAM）モジュールの呼び出し
                time.sleep(1.5)             # 次の処理へ行く前の安全なインターバル
                run_autonomous_mode(cli)    # 自律走行モードの呼び出し
            
        elif choice == '2':
            # 通信テスト用：受信サーバーのみ単独で起動
            receiver = CoordCommunication(port=LISTEN_PORT)
            receiver.start_server()
            
        elif choice == '3':
            # 通信テスト用：手動で入力したIP宛に、ダミーデータを10回投げる
            comm = CoordCommunication()
            test_ip = input("送信先PCのIPアドレスを入力 (自分のPCなら 127.0.0.1): ").strip()
            print(f"\n{test_ip} へテスト送信を開始します...")
            for i in range(10):
                success = comm.send_data(test_ip, i * 10.0, i * 20.0, robot_id=ROBOT_ID)
                print(f"送信 {i+1}/10: {'✅ 成功' if success else '❌ 失敗'}")
                time.sleep(1)
                
        else:
            print("無効な選択です。プログラムを終了します。")
            
    except KeyboardInterrupt:
        print("\nユーザー操作によりシステムを終了しました。")
    except Exception as e:
        print(f"\nエラー：実行中に問題が発生しました。({e})")
        traceback.print_exc()