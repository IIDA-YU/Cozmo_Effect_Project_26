import cv2
import numpy as np
import pycozmo
import time
import threading
from telecommunication.pc_communication import CoordCommunication

class ArucoCoordinateTracker:
    def __init__(self, cli, target_ip_dict, camera_viewer=None):
        self.cli = cli
        self.target_ip_dict = target_ip_dict
        self.camera_viewer = camera_viewer  # ★追加：司令塔(CameraViewer)を受け取る
        self.comm = CoordCommunication()
        
        # ArUcoマーカー検知の初期設定
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.last_sent_time = 0.0
        self.send_interval = 0.2  
        self.running = False
        
        # ★追加：PCの負荷を抑えるための画像ロック機能（YOLOと同じ仕組み）
        self._latest_raw_image = None
        self._lock = threading.Lock()
        
        print(f"📡 [ArucoTracker] 初期化完了: 監視対象ID群 {list(self.target_ip_dict.keys())}")

    def on_camera_image(self, cli, image):
        # 映像を受け取るだけ（通信の邪魔をしない）
        with self._lock:
            self._latest_raw_image = image

    def _tracker_loop(self):
        while self.running:
            raw_image = None
            with self._lock:
                raw_image = self._latest_raw_image
            
            if raw_image is not None:
                try:
                    img_np = np.array(raw_image)
                    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                    corners, ids, rejectedImgPoints = self.detector.detectMarkers(frame)

                    boxes = []
                    if ids is not None:
                        current_time = time.time()
                        should_send = (current_time - self.last_sent_time >= self.send_interval)
                        if should_send:
                            self.last_sent_time = current_time

                        for i in range(len(ids)):
                            marker_id = ids[i][0]
                            
                            # ★追加：画面に描画するための枠(Bounding Box)の座標を計算
                            c = corners[i][0]
                            x1 = int(min(c[:, 0]))
                            y1 = int(min(c[:, 1]))
                            x2 = int(max(c[:, 0]))
                            y2 = int(max(c[:, 1]))
                            
                            boxes.append({
                                "box": (x1, y1, x2, y2),
                                "label": f"ID:{marker_id}"  # IDを表示
                            })
                            
                            # 🟢 見つけたIDが対応表(target_ip_dict)に登録されているか確認し、送信
                            if should_send and marker_id in self.target_ip_dict:
                                target_ip = self.target_ip_dict[marker_id]
                                print(f"🎯 ターゲットCozmo (ID: {marker_id}) を補足！")
                                
                                if self.cli.pose:
                                    my_x = self.cli.pose.position.x
                                    my_y = self.cli.pose.position.y
                                    
                                    def send_task(ip, mx, my):
                                        result = self.comm.send_data(
                                            target_ip=ip, 
                                            x=mx, 
                                            y=my, 
                                            robot_id="Cozmo-01" # 自分のID（必要に応じて変更）
                                        )
                                        if result:
                                            print(f" ┗ 座標送信成功: X={mx:.2f}, Y={my:.2f} -> {ip}")
                                        else:
                                            print(f" ┗ 座標送信失敗 (通信エラー) -> {ip}")
                                    
                                    threading.Thread(target=send_task, args=(target_ip, my_x, my_y), daemon=True).start()

                    # ★変更：自分ではcv2.imshowせず、司令塔(CameraViewer)に枠とIDを渡して描画してもらう
                    if self.camera_viewer:
                        self.camera_viewer.set_detections("aruco", boxes)
                        
                except Exception as e:
                    print(f"⚠️ ArUco検知エラー: {e}")
            
            # 負荷軽減
            time.sleep(0.05)

    def start(self):
        self.running = True
        threading.Thread(target=self._tracker_loop, daemon=True).start()
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def stop(self):
        self.running = False