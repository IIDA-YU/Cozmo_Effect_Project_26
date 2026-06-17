# 障害物を検知して回避するモード。AIモデル（Roboflow）を使ってキューブを認識し、距離推定も行います。
# 事前に download_model.py を実行してモデルをPC内にダウンロードしておいてください。
# AI推論はPC側で行い、Cozmoは回避動作だけを担当する設計です。

import os
import threading
import time

import cv2
import numpy as np
import pycozmo
from inference import get_model 

# 回避動作のパラメータ
FORWARD_SPEED = 40
BACKWARD_SPEED = -40
TURN_SPEED = 45
AVOID_DURATION_SEC = 0.9
AVOID_COOLDOWN_SEC = 1.5
# OpenCV距離推定のパラメータ
KNOWN_CUBE_WIDTH_MM = 32.0  # 実物キューブ幅(mm)。実測値に合わせて調整してください。
FOCAL_LENGTH_PIX = 207.0  # カメラの焦点距離(画素)。キャリブレーションで調整。
AVOID_DISTANCE_THRESHOLD_MM = 450.0  # この距離以内なら回避を開始
# AI判定のパラメータ
INFERENCE_INTERVAL_SEC = 0.01  # 0.12秒から短くして、推論頻度を増やす
CONFIDENCE_THRESHOLD = 0.50
MIN_AVOID_BOX_AREA = 8000
WINDOW_NAME = "AI Avoidance Camera"

# 📌 download_model.py と同じモデル・APIキー設定を使います
MODEL_ID = "powercube/1"
API_KEY = os.environ.get("ROBFLOW_API_KEY") or os.environ.get("ROBOFLOW_API_KEY")


def frame_from_raw_image(image):
    """pycozmo のカメラ画像を OpenCV 用のBGRフレームへ変換する"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


class CubeAvoidanceController:
    """AIモデル（Roboflow）を使ってキューブを検知・回避するクラス"""

    def __init__(self, cli):
        self.cli = cli
        self.is_running = True
        self.is_moving = False
        self.is_avoiding = False
        self.last_avoid_time = 0.0
        self.last_inference_time = 0.0

        self.latest_frame = None
        self.latest_predictions = []
        self.latest_detected = False
        self.frame_lock = threading.Lock()
        self.avoid_thread = None

        print("🧠 AIモデルを読み込んでいます（PC内のキャッシュを利用）...")
        self.model = get_model(model_id=MODEL_ID, api_key=API_KEY)
        print("✅ AIモデルの読み込み完了！")

        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def start_forward(self):
        """前進開始"""
        if self.is_running and not self.is_moving and not self.is_avoiding:
            self.cli.drive_wheels(FORWARD_SPEED, FORWARD_SPEED)
            self.is_moving = True

    def stop(self):
        """停止"""
        self.cli.drive_wheels(0, 0)
        self.is_moving = False

    def avoid_once(self):
        """障害物回避動作（少し下がって旋回）"""
        if self.is_avoiding or not self.is_running:
            return

        self.is_avoiding = True
        try:
            self.stop()
            print("⚠️ AIが近いキューブを検知しました！回避行動を開始します...")
            self.cli.drive_wheels(BACKWARD_SPEED, BACKWARD_SPEED)
            time.sleep(0.5)
            if not self.is_running:
                return

            self.cli.drive_wheels(-TURN_SPEED, TURN_SPEED)
            time.sleep(AVOID_DURATION_SEC)
            if not self.is_running:
                return

            self.stop()
            print("✅ 回避完了。前進を再開します。")
            self.last_avoid_time = time.time()
        finally:
            self.is_avoiding = False
            self.start_forward()

    def on_camera_image(self, cli, image):
        """Cozmoのカメラ映像を受け取り、最新フレームだけをAI推論スレッドへ渡す"""
        if not self.is_running:
            return

        frame = frame_from_raw_image(image)

        with self.frame_lock:
            self.latest_frame = frame.copy()
            predictions = list(self.latest_predictions)
            detected = self.latest_detected

        self._draw_predictions(frame, predictions)

        if self.is_avoiding:
            text = "AI: AVOIDING"
            color = (0, 255, 255)
        elif detected:
            text = "AI: OBSTACLE DETECTED!"
            color = (0, 0, 255)
        else:
            text = "AI: CLEAR"
            color = (0, 255, 0)
            self.start_forward()

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(1)

    def _inference_loop(self):
        """AI推論は別スレッドで最新フレームだけ処理し、古いフレームを溜めない"""
        while self.is_running:
            if self.is_avoiding:
                time.sleep(0.03)
                continue

            now = time.time()
            if now - self.last_inference_time < INFERENCE_INTERVAL_SEC:
                time.sleep(0.01)
                continue

            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is None:
                time.sleep(0.03)
                continue

            self.last_inference_time = time.time()

            try:
                results = self.model.infer(frame)
            except Exception as e:
                print(f"AI推論中にエラーが発生しました: {e}")
                time.sleep(0.3)
                continue

            predictions = self._parse_predictions(results)
            detected = any(
                pred["confidence"] >= CONFIDENCE_THRESHOLD
                and pred["area"] >= MIN_AVOID_BOX_AREA
                and pred["distance_mm"] is not None
                and pred["distance_mm"] <= AVOID_DISTANCE_THRESHOLD_MM
                for pred in predictions
            )

            with self.frame_lock:
                self.latest_predictions = predictions
                self.latest_detected = detected

            if detected and time.time() - self.last_avoid_time >= AVOID_COOLDOWN_SEC:
                self._start_avoidance()

    def _start_avoidance(self):
        if self.avoid_thread is not None and self.avoid_thread.is_alive():
            return

        self.avoid_thread = threading.Thread(target=self.avoid_once, daemon=True)
        self.avoid_thread.start()

    def _parse_predictions(self, results):
        if not results:
            return []

        predictions = getattr(results[0], "predictions", [])
        if not predictions and isinstance(results[0], dict):
            predictions = results[0].get("predictions", [])

        parsed = []
        for pred in predictions:
            conf = self._prediction_value(pred, "confidence", 0)
            x = self._prediction_value(pred, "x", 0)
            y = self._prediction_value(pred, "y", 0)
            w = self._prediction_value(pred, "width", 0)
            h = self._prediction_value(pred, "height", 0)
            area = w * h

            distance_mm = self._estimate_distance_mm(w)
            parsed.append({
                "confidence": conf,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "distance_mm": distance_mm,
            })

        return parsed

    def _prediction_value(self, prediction, name, default):
        if isinstance(prediction, dict):
            return prediction.get(name, default)
        return getattr(prediction, name, default)

    def _estimate_distance_mm(self, box_width_px):
        if box_width_px <= 0:
            return None
        return KNOWN_CUBE_WIDTH_MM * FOCAL_LENGTH_PIX / box_width_px

    def _draw_predictions(self, frame, predictions):
        for pred in predictions:
            conf = pred["confidence"]
            x = pred["x"]
            y = pred["y"]
            w = pred["width"]
            h = pred["height"]

            top_x = int(x - w / 2)
            top_y = int(y - h / 2)
            cv2.rectangle(frame, (top_x, top_y), (top_x + int(w), top_y + int(h)), (255, 0, 0), 2)
            label = f"Cube: {conf:.2f}"
            if pred["distance_mm"] is not None:
                label += f" {pred['distance_mm']:.0f}mm"
            cv2.putText(frame, label, (top_x, top_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def run_marker_avoidance_mode(cli):
    """Cozmoのカメラ映像をAI推論に渡し、キューブ検知時に回避する"""
    cli.enable_camera(enable=True, color=True)

    controller = CubeAvoidanceController(cli)
    controller.start_forward()
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, controller.on_camera_image)

    print("🤖 AIキューブ回避モードを開始しました。(Ctrl+Cで終了)")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nAIキューブ回避モードを終了します。")
    finally:
        controller.is_running = False
        try:
            cli.del_handler(pycozmo.event.EvtNewRawCameraImage, controller.on_camera_image)
        except Exception as e:
            print(f"カメラハンドラ解除中にエラーが発生しました: {e}")

        try:
            cli.enable_camera(enable=False)
        except Exception as e:
            print(f"カメラ停止中にエラーが発生しました: {e}")

        try:
            controller.stop()
        except Exception as e:
            print(f"停止命令中にエラーが発生しました: {e}")

        cv2.destroyAllWindows()
