import time, threading, pycozmo
from connect.network_check import cozmo_session
from hardware.buttery import BatteryMonitor
from camera.camera_movie import CameraViewer
from performance.facial_expression import FaceManager
from hardware.collision_prevention import CollisionPrevention
from hardware.step_detection import StepDetector
from hardware.roaming_controller import RoamingController
from camera.cozmo_tracker import CozmoTracker
from camera.cube_tracker import CubeTracker
from algorithm.mapping.visual_slam import CozmoSlamController

# algorithm フォルダの virtual_pheromones フォルダにあるファイルからインポート
from algorithm.virtual_pheromones.pheromone_navi_log import PheromoneTracker

# ★ 追加：座標通信の「送信（口）」と「受信（耳）」の両方をインポート
from telecommunication.coordinate_sender import CoordinateSender
from telecommunication.pc_communication import CoordCommunication

# 通信パッチ（通信クラッシュを防ぐ安全装置）
original_send = pycozmo.conn.Connection.send
_global_pycozmo_lock = threading.Lock()
def thread_safe_send(self, pkt):
    with _global_pycozmo_lock: original_send(self, pkt)
pycozmo.conn.Connection.send = thread_safe_send

# =========================================================
# 📌 ネットワーク・P2P通信設定（★ここを環境に合わせて書き換える）
# =========================================================
COZMO_SSID = "Cozmo_42A832"   
RETURN_SSID = "R0347"         

ROBOT_ID = "Cozmo-A"            # 自分の名前（もう1台のPCでは "Cozmo-B" にする）
TARGET_PC_IP = "127.0.0.1"    # 通信したい【相手のPC】のIPアドレス
LISTEN_PORT = 50007             # 受信用のポート番号（基本このままでOK）
# =========================================================

def main():
    with cozmo_session(COZMO_SSID, RETURN_SSID) as cli:
        print("\n🔧 システム起動中...")
        cli.conn_lock = threading.Lock()
        
        # 1. 各種モジュールの起動
        battery = BatteryMonitor(cli); battery.start()
        camera_viewer = CameraViewer(cli, battery_monitor=battery); camera_viewer.start()
        face_manager = FaceManager(cli)
        
        # 2. 回避システムの起動
        preventer = CollisionPrevention(cli, face_manager=face_manager, camera_viewer=camera_viewer)
        step_detector = StepDetector(cli, preventer); step_detector.start()
        
        # 3. AI検知システムの起動
        cozmo_tracker = CozmoTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)
        cozmo_tracker.start()
        cube_tracker = CubeTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)
        cube_tracker.start()

        # 📌 仮想フェロモントラッカーを初期化
        #tracker = PheromoneTracker(log_filename="forbidden_zones.json")

        # =========================================================
        # ★ 4. P2P 双方向通信システムの起動
        # =========================================================
        # ① 受信側（耳）を別スレッドで起動して待機させる
        receiver = CoordCommunication(port=LISTEN_PORT)
        receiver_thread = threading.Thread(target=receiver.start_server, daemon=True)
        receiver_thread.start()
        
        # 少し待って受信サーバーが立ち上がったのを確認
        time.sleep(1.0) 

        # ② 送信側（口）を起動して、相手のPCへ座標を送り始める
        coord_sender = CoordinateSender(cli, target_ip=TARGET_PC_IP, robot_id=ROBOT_ID, interval=0.2)
        coord_sender.start()
        # =========================================================
def main():
    # 📝 1台目（マップを作る側）の設定
    slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt", use_cached_map=False)
    
    # 📝 2台目（受信したマップを使う側）の場合は上を消して、こちらを有効にする
    # slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt", use_cached_map=True)
    
    try:
        slam_bot.run()
    except KeyboardInterrupt:
        print("\n強制終了が検知されました。")
    
    # 終了後にマッピングされた情報を保存
    slam_bot.save_map_data(json_filename="slam_data.json", img_filename="slam_map.png")

    if __name__ == "__main__":
      main()

        # 5. 自律走行の起動
    roamer = RoamingController(cli, preventer)
    try:
            roamer.start()
            print("\n🚀 自律走行および【双方向通信】を開始しました！")
            while True: time.sleep(1)
    finally:
            print("\n🛑 停止中...")
            roamer.stop()
            coord_sender.stop()     # 送信をストップ
            receiver.running = False # 受信サーバーも安全にストップ
            cube_tracker.stop()
            cozmo_tracker.stop()
            step_detector.stop()
            camera_viewer.stop()
            battery.stop()

if __name__ == "__main__":
    print("\n【システム起動メニュー】")
    print("1: 🤖 通常モード (Cozmoに接続し、自律走行とAIをフル稼働)")
    print("2: 💻 受信デバッグ (Cozmoに繋がず、相手からの座標を受信・表示するだけ)")
    print("3: 📡 送信デバッグ (Cozmoに繋がず、ダミーの座標を相手に送信テスト)")
    
    try:
        choice = input("\nモードを選択してください (1 / 2 / 3): ").strip()
        
        if choice == '1':
            main()  # いつもの本番稼働
            
        elif choice == '2':
            # デバッグ用の受信のみ行う
            from telecommunication.pc_communication import CoordCommunication
            receiver = CoordCommunication(port=LISTEN_PORT)
            receiver.start_server()
            
        elif choice == '3':
            # デバッグ用の送信のみ行う
            import time
            from telecommunication.pc_communication import CoordCommunication
            comm = CoordCommunication()
            test_ip = input("送信先PCのIPアドレスを入力 (自分のPCなら 127.0.0.1): ").strip()
            print(f"\n{test_ip} へテスト送信を開始します...")
            for i in range(10):
                success = comm.send_data(test_ip, i*10.0, i*20.0, robot_id=ROBOT_ID)
                print(f"送信 {i+1}/10: {'✅ 成功' if success else '❌ 失敗'}")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nユーザー操作によりシステムを終了しました。")
    except Exception as e:
        print(f"\nエラー：実行中に問題が発生しました。({e})")