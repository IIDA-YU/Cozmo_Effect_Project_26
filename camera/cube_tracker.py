import threading, time, os, cv2, numpy as np
from ultralytics import YOLO
import pycozmo

KNOWN_CUBE_WIDTH_MM = 30.0  # ※実際のキューブの大きさに合わせて変更してください（標準は約44mmです）
FOCAL_LENGTH_PIX = 235.0    
AVOID_DISTANCE_THRESHOLD_MM = 90.0  
RESIZE_WIDTH = 416
FRAME_SKIP = 3

class CubeTracker:
    def __init__(self, cli, collision_preventer, camera_viewer=None):
        self.cli = cli
        self.preventer = collision_preventer
        self.camera_viewer = camera_viewer
        self.model = YOLO(os.path.join(os.path.dirname(__file__), "..", "algorithm", "mapping", "learning_data", "best_powercube.pt"))
        self._latest_raw_image = None
        self._lock = threading.Lock()
        self.is_running = False

    def on_camera_image(self, cli, image):
        # 超軽量化ポイント：通信の邪魔をしないよう、ここでは画像を受け取るだけにする
        with self._lock: 
            self._latest_raw_image = image

    def _ai_loop(self):
        frame_skip_counter = 0
        while self.is_running:
            if self.preventer.is_avoiding_cliff:
                time.sleep(0.1)
                continue
                
            raw_image = None
            with self._lock: 
                raw_image = self._latest_raw_image
                
            if raw_image is not None:
                try:
                    # 超軽量化ポイント：AIが考える瞬間（裏側）だけ画像を変換してPCの負担を激減させる
                    img_np = np.array(raw_image)
                    frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

                    # フレームスキップ
                    frame_skip_counter += 1
                    if frame_skip_counter % FRAME_SKIP != 0:
                        # 表示更新のみで推論をスキップ
                        if self.camera_viewer:
                            self.camera_viewer.set_detections("cube", [])
                        time.sleep(0.01)
                        continue

                    # 推論負荷軽減: 縮小して渡し、検出バウンディングを元サイズに戻す
                    orig_h, orig_w = frame.shape[:2]
                    resize_w = min(RESIZE_WIDTH, orig_w)
                    resize_h = max(1, int(orig_h * (resize_w / orig_w)))
                    small = cv2.resize(frame, (resize_w, resize_h))
                    scale_x = orig_w / resize_w
                    scale_y = orig_h / resize_h

                    # 誤検知対策: 確信度(conf)を緩める（より多く検出させる）
                    results = self.model(small, conf=0.64, verbose=False)
                    boxes = []
                    
                    for r in results:
                        for box in r.boxes:
                            # AIが判定した物体の名前を取得
                            class_name = self.model.names[int(box.cls[0])]
                            
                            # 名前がキューブ以外なら無視する（YOLOの学習ラベル名に合わせて調整してください）
                            if class_name.lower() not in ["cube", "powercube", "mark", "station"]: # ※Cubeやパワーキューブ、マーカー等を許可
                                continue

                            conf_val = float(box.conf[0])
                            x1_s, y1_s, x2_s, y2_s = box.xyxy[0].tolist()
                            x1 = x1_s * scale_x
                            x2 = x2_s * scale_x
                            y1 = y1_s * scale_y
                            y2 = y2_s * scale_y
                            w = x2 - x1; h = y2 - y1
                            if h == 0: continue
                            aspect_ratio = w / h
                            
                            frame_width = frame.shape[1]
                            
                            # 画面のどこにいるか判定 (右・左・中央)
                            center_x = (x1 + x2) / 2
                            if center_x < frame_width * 0.35:
                                position = "left"
                            elif center_x > frame_width * 0.65:
                                position = "right"
                            else:
                                position = "center"
                                
                            # 端っこで見切れているか判定 (端から15ピクセル以内)
                            is_edge = (x1 <= 15) or (x2 >= frame_width - 15)

                            # 誤検知対策: 端っこにいない時だけ、アスペクト比（縦横比）のチェックを行う
                            if not is_edge:
                                # より緩めに許可 (0.6〜1.8)。斜めや伸びた矩形を許容しやすくする
                                if aspect_ratio < 0.6 or aspect_ratio > 1.8:
                                    # ただし以下の緩和条件に合致すれば通す:
                                    # - 高信頼度 (>=0.75)
                                    # - 検出領域が十分に大きい（画面幅の15%以上）
                                    # - あるいは近距離（障害回避閾値の1.2倍以内）
                                    allow_rotated = (
                                        (conf_val >= 0.75)
                                        or (max(w, h) >= orig_w * 0.15)
                                        or ((KNOWN_CUBE_WIDTH_MM * FOCAL_LENGTH_PIX) / max(w, h) <= AVOID_DISTANCE_THRESHOLD_MM * 1.2)
                                    )
                                    if not allow_rotated:
                                        continue
                                
                            # 距離計算: 横が見切れていても「高さと幅の大きい方」を使って正確な距離を出す
                            effective_size = max(w, h)
                            distance_mm = (KNOWN_CUBE_WIDTH_MM * FOCAL_LENGTH_PIX) / effective_size
                            
                            if distance_mm <= AVOID_DISTANCE_THRESHOLD_MM:
                                boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cube WARNING {int(distance_mm)}mm"})
                            else:
                                boxes.append({"box": (int(x1), int(y1), int(x2), int(y2)), "label": f"Cube FAR {int(distance_mm)}mm"})

                            # 軽量ログ: 近距離または高信頼度の検出のみ出力してデバッグを助ける
                            if conf_val >= 0.9 or distance_mm <= AVOID_DISTANCE_THRESHOLD_MM:
                                try:
                                    print(f"[CubeDetect] cls={class_name} conf={conf_val:.2f} dist={distance_mm:.1f} pos={position} edge={is_edge} aspect={aspect_ratio:.2f}")
                                except Exception:
                                    pass

                            # 距離に関係なく、毎回距離と位置(position)を報告する
                            if self.preventer:
                                self.preventer.trigger_obstacle_avoidance("Cube", distance_mm, position)
                                
                    if self.camera_viewer:
                        self.camera_viewer.set_detections("cube", boxes)
                except Exception: pass
            
            # PCの処理落ちを防ぐため、YOLOの思考間隔を少しだけ休ませる（約6FPS）
            time.sleep(0.15)

    def start(self):
        self.is_running = True
        threading.Thread(target=self._ai_loop, daemon=True).start()
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def stop(self):
        self.is_running = False