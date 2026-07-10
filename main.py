import time, threading, pycozmo
from connect.network_check import cozmo_session, switch_wifi
from hardware.buttery import BatteryMonitor
from camera.camera_movie import CameraViewer
from performance.facial_expression import FaceManager
from hardware.collision_prevention import CollisionPrevention
from hardware.step_detection import StepDetector
from hardware.roaming_controller import RoamingController
from camera.cozmo_tracker import CozmoTracker
from camera.cube_tracker import CubeTracker
from algorithm.mapping.visual_slam import CozmoSlamController
from algorithm.virtual_pheromones.pheromone_navi_log import PheromoneTracker
from telecommunication.coordinate_sender import CoordinateSender
from telecommunication.pc_communication import CoordCommunication
from camera.aruco_tracker import ArucoCoordinateTracker

original_send = pycozmo.conn.Connection.send
_global_pycozmo_lock = threading.Lock()

def thread_safe_send(self, pkt):
    with _global_pycozmo_lock:
        original_send(self, pkt)

pycozmo.conn.Connection.send = thread_safe_send

# =========================================================
# 📌 ネットワーク・P2P通信設定
# =========================================================
COZMO_SSID = "Cozmo_42A832"
RETURN_SSID = "R0347"

ROBOT_ID = "Cozmo-A"
TARGET_PC_IP = "127.0.0.1"
LISTEN_PORT = 50007

TARGET_IP_DICT = {
    1: "172.18.67.189",
    2: "172.18.67.198",
    3: "192.168.1.12",
    4: "192.168.1.13",
    5: "192.168.1.14"
}
# =========================================================


def run_slam_mapping():
    print("\n=== SLAMマッピングを開始します ===")
    slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt", use_cached_map=False)
    try:
        slam_bot.run()
    except KeyboardInterrupt:
        print("\n強制終了が検知されました。")
    finally:
        slam_bot.save_map_data(json_filename="slam_data.json", img_filename="slam_map.png")


def run_autonomous_mode():
    with cozmo_session(COZMO_SSID, RETURN_SSID) as cli:
        print("\n🔧 自律走行モードを起動します...")
        cli.conn_lock = threading.Lock()

        battery = BatteryMonitor(cli)
        battery.start()

        camera_viewer = CameraViewer(cli, battery_monitor=battery)
        camera_viewer.start()

        face_manager = FaceManager(cli)

        preventer = CollisionPrevention(cli, face_manager=face_manager, camera_viewer=camera_viewer)
        step_detector = StepDetector(cli, preventer)
        step_detector.start()

        cozmo_tracker = CozmoTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)
        cozmo_tracker.start()
        cube_tracker = CubeTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)
        cube_tracker.start()

        aruco_tracker = ArucoCoordinateTracker(cli, target_ip_dict=TARGET_IP_DICT, camera_viewer=camera_viewer)
        aruco_tracker.start()

        receiver = CoordCommunication(port=LISTEN_PORT)
        receiver_thread = threading.Thread(target=receiver.start_server, daemon=True)
        receiver_thread.start()
        time.sleep(1.0)

        coord_sender = CoordinateSender(cli, target_ip=TARGET_PC_IP, robot_id=ROBOT_ID, interval=0.2)
        coord_sender.start()

        roamer = RoamingController(cli, preventer)
        try:
            roamer.start()
            print("\n🚀 自律走行および【双方向通信】【機体識別】を開始しました！")
            while True:
                time.sleep(1)
        finally:
            print("\n🛑 停止中...")
            roamer.stop()
            coord_sender.stop()
            if hasattr(receiver, "running"):
                receiver.running = False
            aruco_tracker.stop()
            cube_tracker.stop()
            cozmo_tracker.stop()
            step_detector.stop()
            camera_viewer.stop()
            battery.stop()


def run_network_map_and_roam():
    print("\n=== 現在のWi-Fiから Cozmo Wi-Fi へ切り替え中 ===")
    if not switch_wifi(COZMO_SSID):
        print("⚠️ Cozmo Wi-Fiへの切り替えに失敗しました。終了します。")
        return

    try:
        run_slam_mapping()
        run_autonomous_mode()
    finally:
        print("\n=== 可能であれば元のWi-Fiに戻します ===")
        switch_wifi(RETURN_SSID)


def main():
    print("\n【システム起動メニュー】")
    print("1: 🤖 ネット切り替え → マッピング → 自律走行")
    print("2: 💻 受信デバッグ (Cozmoに繋がず、相手からの座標を受信・表示するだけ)")
    print("3: 📡 送信デバッグ (Cozmoに繋がず、ダミーの座標を相手に送信テスト)")

    try:
        choice = input("\nモードを選択してください (1 / 2 / 3): ").strip()
        if choice == '1':
            run_network_map_and_roam()
        elif choice == '2':
            receiver = CoordCommunication(port=LISTEN_PORT)
            receiver.start_server()
        elif choice == '3':
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


if __name__ == "__main__":
    main()

 