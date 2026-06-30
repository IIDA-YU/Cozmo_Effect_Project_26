import threading
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pycozmo

# 📌 距離計算パラメータ（Cozmo用）
KNOWN_COZMO_WIDTH_MM = 60.0  # Cozmoの実際の正面幅(mm) ※実測に合わせて微調整可能
FOCAL_LENGTH_PIX = 235.0     # Cozmoカメラの焦点距離
AVOID_DISTANCE_THRESHOLD_MM = 90.0  # ★何mmまで近づいたら回避するか（ここを調整！）

class CozmoTracker:
    def __init__(self, cli, collision_preventer=None, camera_viewer=None):
        self.cli = cli
        self.preventer = collision_preventer
        self.camera_viewer = camera_viewer
        
        self.model = YOLO(os.path.join(os.path.dirname(__file__), "learning_data", "best_cozmo_distinguish.pt"))
        self._latest_frame = None
        self._lock = threading.Lock()
        self.is_running = False

    def on_camera_image(self, cli, image):
        img_np = np.array(image)
        frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        with self._lock: 
            self._latest_frame = frame

    def _ai_loop(self):
        last_processed_frame = None
        frame_skip_counter = 0
        FRAME_SKIP = 3  # ★ 3フレームに1回だけAI推論（カクつくなら4や5に増やせます）

        while self.is_running:
            if self.preventer and (self.preventer.is_avoiding_cliff or self.preventer.is_avoiding_collision):
                time.sleep(0.1)
                continue
                
            frame = None
            with self._lock: 
                frame = self._latest_frame
                
            if frame is not None:
                # 前回と同じ画像ならAI処理をしない（無駄な計算を省く）
                if last_processed_frame is None or not np.array_equal(frame, last_processed_frame):
                    last_processed_frame = frame.copy()
                    frame_skip_counter += 1
                    
                    # 指定フレームに1回だけYOLOを動かす
                    if frame_skip_counter % FRAME_SKIP == 0:
                        try:
                            results = self.model(frame, conf=0.65, verbose=False)
                            boxes = []
                            
                            for r in results:
                                for box in r.boxes:
                                    conf_val = float(box.conf[0])
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    w = x2 - x1
                                    h = y2 - y1
                                    
                                    # 距離計算
                                    distance_mm = (KNOWN_COZMO_WIDTH_MM * FOCAL_LENGTH_PIX) / w
                                    
                                    if distance_mm <= AVOID_DISTANCE_THRESHOLD_MM:
                                        boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cozmo WARNING {int(distance_mm)}mm"})
                                        if self.preventer:
                                            self.preventer.trigger_obstacle_avoidance("Cozmo")
                                    else:
                                        boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cozmo {int(distance_mm)}mm"})

                                    if self.preventer:
                                        self.preventer.trigger_obstacle_avoidance("Cozmo", distance_mm)
                                                                                
                            if self.camera_viewer:
                                self.camera_viewer.set_detections("cozmo", boxes)
                        except Exception: 
                            pass
            
            time.sleep(0.08) # CPU負荷を下げるための微小なスリープ

    def start(self):
        self.is_running = True
        threading.Thread(target=self._ai_loop, daemon=True).start()
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def stop(self):
        self.is_running = False