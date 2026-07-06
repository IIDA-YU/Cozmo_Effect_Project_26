import threading
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
import pycozmo

# 📌 距離計算パラメータ（Cozmo用）
KNOWN_COZMO_WIDTH_MM = 94.0  # Cozmoの実際の正面幅(mm) ※実測に合わせて微調整可能
FOCAL_LENGTH_PIX = 235.0     # Cozmoカメラの焦点距離
AVOID_DISTANCE_THRESHOLD_MM = 90.0  # ★何mmまで近づいたら回避するか（ここを調整！）
RESIZE_WIDTH = 416  # 推論時に縮小する幅（ピクセル）

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
                            # 推論負荷軽減: フレームを縮小してYOLOに渡し、検出座標を元サイズに戻す
                            orig_h, orig_w = frame.shape[:2]
                            resize_w = min(RESIZE_WIDTH, orig_w)
                            resize_h = max(1, int(orig_h * (resize_w / orig_w)))
                            small = cv2.resize(frame, (resize_w, resize_h))
                            scale_x = orig_w / resize_w
                            scale_y = orig_h / resize_h

                            results = self.model(small, conf=0.75, verbose=False)
                            boxes = []
                            
                            for r in results:
                                for box in r.boxes:
                                    conf_val = float(box.conf[0])
                                    x1_s, y1_s, x2_s, y2_s = box.xyxy[0].tolist()
                                    # 元画像サイズに戻す
                                    x1 = x1_s * scale_x
                                    x2 = x2_s * scale_x
                                    y1 = y1_s * scale_y
                                    y2 = y2_s * scale_y
                                    w = x2 - x1
                                    h = y2 - y1
                                    # 距離計算
                                    distance_mm = (KNOWN_COZMO_WIDTH_MM * FOCAL_LENGTH_PIX) / w

                                    # 追加: 画面上の位置情報とエッジ判定を行う（ただし検出自体は捨てない）
                                    frame_width = frame.shape[1]
                                    center_x = (x1 + x2) / 2
                                    if center_x < frame_width * 0.35:
                                        position = "left"
                                    elif center_x > frame_width * 0.65:
                                        position = "right"
                                    else:
                                        position = "center"

                                    # マージン（ピクセル）: 検出幅の半分を安全マージンとして使う
                                    margin_px = max(1, int(w / 2))
                                    is_edge = (x1 <= margin_px) or (x2 >= frame_width - margin_px)

                                    label_base = f"Cozmo {int(distance_mm)}mm"
                                    if distance_mm <= AVOID_DISTANCE_THRESHOLD_MM:
                                        label_base = f"Cozmo WARNING {int(distance_mm)}mm"
                                    if is_edge:
                                        label_base += " (EDGE)"

                                    boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": label_base})

                                    # 回避ロジックには位置も渡す（既存のpreventerインターフェイスに合わせる）
                                    if self.preventer:
                                        try:
                                            self.preventer.trigger_obstacle_avoidance("Cozmo", distance_mm, position)
                                        except TypeError:
                                            # 互換性確保: もし古いシグネチャなら位置を渡さずに呼ぶ
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