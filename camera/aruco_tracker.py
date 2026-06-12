import cv2
import numpy as np
import pycozmo
import time
import threading
from telecommunication.pc_communication import CoordCommunication

class ArucoCoordinateTracker:
    def __init__(self, cli, target_ip_dict):
        self.cli = cli
        self.target_ip_dict = target_ip_dict
        self.comm = CoordCommunication()
        
        # ArUcoマーカー検知の初期設定
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.last_sent_time = 0.0
        self.send_interval = 0.2  # 0.2秒に1回送信（ネットワーク安定化）
        self.running = True
        
        print(f"📡 [ArucoTracker] 初期化完了: 監視対象ID群 {list(self.target_ip_dict.keys())}")

    def on_camera_image(self, cli, image):
        # 終了フラグが立っていたら画像処理をしない
        if not self.running:
            return

        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(frame)

        if ids is not None:
            # 画面に見つけたマーカーの枠を描画
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            current_time = time.time()
            if current_time - self.last_sent_time >= self.send_interval:
                self.last_sent_time = current_time
                
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    
                    # 🟢 見つけたIDが、対応表(target_ip_dict)に登録されているか確認
                    if marker_id in self.target_ip_dict:
                        target_ip = self.target_ip_dict[marker_id]
                        print(f"🎯 ターゲットCozmo (ID: {marker_id}) をカメラ内に補足！")
                        
                        if self.cli.pose:
                            my_x = self.cli.pose.position.x
                            my_y = self.cli.pose.position.y
                            
                            # カメラの映像を止めないように別スレッドで送信
                            def send_task(ip):
                                result = self.comm.send_data(
                                    target_ip=ip, 
                                    x=my_x, 
                                    y=my_y, 
                                    robot_id="Cozmo-01"
                                )
                                if result:
                                    print(f" ┗ 座標送信成功: X={my_x:.2f}, Y={my_y:.2f} -> {ip}")
                                else:
                                    print(f" ┗ 座標送信失敗 (通信エラー) -> {ip}")
                            
                            threading.Thread(target=send_task, args=(target_ip,), daemon=True).start()

                        else:
                            print(" ┗ 自分の座標(Pose)を取得できませんでした。")

        # 映像を画面に表示
        cv2.imshow("Cozmo ArUco Target Tracking", frame)
        cv2.waitKey(1)