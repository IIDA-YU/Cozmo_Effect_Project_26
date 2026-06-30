import threading, time, os, cv2, numpy as np
from ultralytics import YOLO
import pycozmo

KNOWN_CUBE_WIDTH_MM = 30.0  
FOCAL_LENGTH_PIX = 235.0    
AVOID_DISTANCE_THRESHOLD_MM = 120.0  

class CubeTracker:
    def __init__(self, cli, collision_preventer, camera_viewer=None):
        self.cli = cli
        self.preventer = collision_preventer
        self.camera_viewer = camera_viewer
        self.model = YOLO(os.path.join(os.path.dirname(__file__), "..", "hardware", "best.pt"))
        self._latest_raw_image = None
        self._lock = threading.Lock()
        self.is_running = False

    def on_camera_image(self, cli, image):
        # ★ 超軽量化ポイント：通信の邪魔をしないよう、ここでは画像を受け取るだけにする！
        with self._lock: 
            self._latest_raw_image = image

    def _ai_loop(self):
        while self.is_running:
            if self.preventer.is_avoiding_cliff or self.preventer.is_avoiding_collision:
                time.sleep(0.1)
                continue
                
            raw_image = None
            with self._lock: 
                raw_image = self._latest_raw_image
                
            if raw_image is not None:
                try:
                    # ★ 超軽量化ポイント：AIが考える瞬間（裏側）だけ画像を変換してPCの負担を激減させる
                    img_np = np.array(raw_image)
                    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

                    results = self.model(frame, conf=0.55, verbose=False)
                    boxes = []
                    
                    for r in results:
                        for box in r.boxes:
                            conf_val = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            w = x2 - x1; h = y2 - y1
                            if h == 0: continue
                            aspect_ratio = w / h
                            
                            if w < 60:
                                if aspect_ratio < 0.6 or aspect_ratio > 1.4: continue
                            elif w < 100:
                                if aspect_ratio < 0.4 or aspect_ratio > 2.0: continue
                                
                            distance_mm = (KNOWN_CUBE_WIDTH_MM * FOCAL_LENGTH_PIX) / w
                            
                            if distance_mm <= AVOID_DISTANCE_THRESHOLD_MM:
                                boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cube WARNING {int(distance_mm)}mm"})
                                self.preventer.trigger_obstacle_avoidance("Cube")
                            else:
                                boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cube FAR {int(distance_mm)}mm"})

                            if self.preventer:
                                self.preventer.trigger_obstacle_avoidance("Cube", distance_mm)
                                
                    if self.camera_viewer:
                        self.camera_viewer.set_detections("cube", boxes)
                except Exception: pass
            
            # ★ PCの処理落ちを防ぐため、YOLOの思考間隔を少しだけ休ませる（約6FPS）
            time.sleep(0.15)

    def start(self):
        self.is_running = True
        threading.Thread(target=self._ai_loop, daemon=True).start()
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def stop(self):
        self.is_running = False