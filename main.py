import time, threading, pycozmo
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

# =========================================================
# 🛠️ 1. マルチスレッド通信の安全対策（排他制御）
# 複数のAIや通信モジュールが「同時に」Cozmoへ命令を送ると通信がクラッシュします。
# これを防ぐため、pycozmoの根幹の送信関数を上書きし、Lock(鍵)を使って
# 「必ず順番に1つずつデータを送る」ように改造しています。
# =========================================================
original_send = pycozmo.conn.Connection.send
_global_pycozmo_lock = threading.Lock()

def thread_safe_send(self, pkt):
    with _global_pycozmo_lock:
        original_send(self, pkt)

pycozmo.conn.Connection.send = thread_safe_send

# =========================================================
# 📌 2. 環境設定（コンフィグレーション）
# アプリケーション全体で使う固定値（IPアドレスやポートなど）をまとめています。
# 5台のPCにこのコードを配る際、ここだけを書き換えれば動くように設計されています。
# =========================================================
COZMO_SSID = "Cozmo_42A832" # Cozmo本体が発しているWi-Fiの名前
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

    # --- ① 基礎システムの起動 ---
    # ※ := (セイウチ演算子) を使い、以下の2行の処理を1行にまとめています
    # 1. camera_viewer = CameraViewer(...) で作成・代入
    # 2. camera_viewer.start() で起動
    (battery := BatteryMonitor(cli)).start() # バッテリー残量の監視
    (camera_viewer := CameraViewer(cli, battery_monitor=battery)).start() # PC画面にカメラ映像を映すウィンドウを展開
    (face_manager := FaceManager(cli)).start() # Cozmoの顔（表情）の制御

    # --- ② 障害物と段差回避システムの起動 ---
    (preventer := CollisionPrevention(cli, face_manager=face_manager, camera_viewer=camera_viewer)).start() # 衝突回避システムの起動
    (step_detector := StepDetector(cli, preventer)).start() # 机から落ちないように段差を赤外線センサーで監視

    # --- ③ AI画像認識システムの起動 ---
    (cozmo_tracker := CozmoTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)).start() # 他のCozmoを見つける
    (cube_tracker := CubeTracker(cli, collision_preventer=preventer, camera_viewer=camera_viewer)).start() # YOLOを使ってキューブを見つける
    (aruco_tracker := ArucoCoordinateTracker(cli, target_ip_dict=TARGET_IP_DICT, camera_viewer=camera_viewer)).start() # ArUcoマーカー（ID）を見つけて、TARGET_IP_DICTから宛先を特定する

    # --- ④ P2P 双方向通信システムの起動 ---
    # 受信機：他PCから送られてくる座標を受け取るサーバーを裏側で立てる
    receiver = CoordCommunication(port=LISTEN_PORT)
    receiver_thread = threading.Thread(target=receiver.start_server, daemon=True)
    receiver_thread.start()
    time.sleep(1.0) # サーバーが立ち上がるのを少し待つ

    # 送信機：自分の現在座標を0.2秒ごとに指定した相手に送信し続ける
    coord_sender = CoordinateSender(cli, target_ip=TARGET_PC_IP, robot_id=ROBOT_ID, interval=0.2)
    coord_sender.start()

    # --- ⑤ 自律走行コントローラーの起動 ---
    # 回避システム(preventer)の指示に従いながら、基本は前進し続ける処理
    roamer = RoamingController(cli, preventer)
    
    try:
        roamer.start()
        print("\n🚀 自律走行および【双方向通信】【機体識別】を開始しました！")
        # メインスレッド（このプログラム自体）が終了しないように、ここで無限ループして待機
        while True:
            time.sleep(1)
    finally:
        # Ctrl+Cが押されたら、起動した全てのシステムを順番に安全に止める
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
                print("✅ Cozmoと接続しました。")
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