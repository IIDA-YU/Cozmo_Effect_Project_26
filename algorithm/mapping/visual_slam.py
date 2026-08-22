# ----------------------------------------------------------------------
# visual_slam.py
# チェス駒をランドマークに使うCozmoのVisual SLAM
# ----------------------------------------------------------------------

import os
os.environ["YOLO_OFFLINE"] = "True"

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pycozmo
from ultralytics import YOLO

from camera.camera_movie import (
    CameraViewer,
    normalize_camera_settings,
)


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CHESS_MODEL_PATH = (
    MODULE_DIR
    / "learning_data"
    / "best_chess_4.0.pt"
)


class CozmoSlamController:
    MAP_WINDOW_NAME = "Cozmo SLAM Map"

    def __init__(
        self,
        model_path=DEFAULT_CHESS_MODEL_PATH,
        use_cached_map=False,
        confidence_threshold=0.35,
        inference_size=640,
        enhance_image=True,
        camera_settings=None,
        camera_viewer=None,
    ):
        model_path = self._resolve_model_path(model_path)
        print(f"チェス駒AIモデル({model_path})を準備中...")
        self.model = YOLO(str(model_path))
        self.use_cached_map = use_cached_map
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.INFERENCE_SIZE = inference_size
        self.ENHANCE_IMAGE = enhance_image
        self.camera_settings = normalize_camera_settings(camera_settings)
        self.camera_viewer = camera_viewer
        self.CHESS_CLASS_IDS = self._find_chess_class_ids()

        # --- 約20mmのチェス駒を使う教室デスク向けパラメータ ---
        # 駒の実寸が異なる場合は、この高さだけ実物に合わせて変更する。
        self.REAL_CHESS_PIECE_HEIGHT_MM = 20.0
        self.FOCAL_LENGTH_PX = 235.0
        self.FOV_H_RADIANS = math.radians(60)
        self.MAX_VIEW_DIST = 600.0
        self.MIN_DETECTION_DISTANCE_MM = 35.0
        self.EXPECTED_LANDMARKS = 6
        self.LANDMARK_MATCH_DISTANCE_MM = 150.0
        self.MIN_BOX_HEIGHT_PX = 6
        self.TARGET_HEAD_ANGLE = -0.15 
        self.MAP_SIZE = 450            
        self.MAP_SCALE = 0.5           
        
        # --- 状態変数 ---
        self.memory_map = np.full((self.MAP_SIZE, self.MAP_SIZE, 3), 127, dtype=np.uint8)
        self.true_landmarks = {}  
        self.persistent_mapped_pieces = []
        self.landmark_id_counter = 0
        self.offset_x, self.offset_y, self.offset_theta = 0.0, 0.0, 0.0
        self.current_raw_x, self.current_raw_y = 0.0, 0.0 
        self.target_x = None
        self.target_y = None
        self._last_detection_error_time = 0.0
        self.robot_trajectory = [] 
        self.distance_history = {}
        self.ALPHA = 0.2 

        # 🚀 追加: 自律走行用の変数
        self.autonomous_mode = False
        self.waypoints = []
        self.current_waypoint_index = 0

    @staticmethod
    def _resolve_model_path(model_path):
        """
        visual_slam.py基準でモデルを解決する。

        main.pyから絶対パスを受け取る標準動作に加え、テストなどで
        相対パスを渡した場合もmappingディレクトリを最優先にする。
        """
        requested = Path(model_path).expanduser()
        project_root = MODULE_DIR.parent.parent

        if requested.is_absolute():
            candidates = [requested]
        else:
            candidates = [
                MODULE_DIR / requested,
                project_root / requested,
                Path.cwd() / requested,
            ]

        # 「ファイル名だけ」を指定した場合も正式なlearning_dataを確認する。
        if not requested.is_absolute() and requested.parent == Path("."):
            candidates.insert(
                0,
                MODULE_DIR / "learning_data" / requested.name,
            )

        # 同じ候補を複数回検査しない。
        checked = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in checked:
                continue
            checked.append(resolved)
            if resolved.is_file():
                return resolved

        checked_paths = "\n  - ".join(str(path) for path in checked)
        raise FileNotFoundError(
            "best_chess_4.0.ptが見つかりません。確認した場所:\n"
            f"  - {checked_paths}"
        )

    def _find_chess_class_ids(self):
        """モデルが複数クラスでも、チェス駒クラスだけを推論対象にする。"""
        names = self.model.names
        if isinstance(names, dict):
            class_names = names.items()
        else:
            class_names = enumerate(names)

        class_names = list(class_names)
        if len(class_names) == 1:
            return [int(class_names[0][0])]

        keywords = ("chess", "piece", "pawn", "駒")
        class_ids = [
            int(class_id)
            for class_id, class_name in class_names
            if any(keyword in str(class_name).lower() for keyword in keywords)
        ]

        if not class_ids:
            print("⚠️ チェス駒クラス名を特定できないため、全クラスを検出対象にします。")
            return None
        return class_ids

    def _get_latest_image(self):
        if self.camera_viewer is None:
            return None
        return self.camera_viewer.get_latest_frame()

    def _report_detection_error(self, error):
        """同じ推論エラーでコンソールを埋めず、原因は確認できるようにする。"""
        now = time.time()
        if now - self._last_detection_error_time >= 5.0:
            print(f"⚠️ チェス駒の検出処理でエラー: {error}")
            self._last_detection_error_time = now

    def detect_chess_pieces(self, frame):
        """
        best_chess_4.0.ptでチェス駒を検出する。

        Power Cube向けだった正方形の縦横比制限は使用しない。
        小さな駒を拾えるよう信頼度を少し下げつつ、極端に小さい枠は除外する。
        """
        predict_kwargs = {
            "source": frame,
            "conf": self.CONFIDENCE_THRESHOLD,
            "iou": 0.45,
            "imgsz": self.INFERENCE_SIZE,
            "max_det": max(12, self.EXPECTED_LANDMARKS * 2),
            "verbose": False,
        }
        if self.CHESS_CLASS_IDS is not None:
            predict_kwargs["classes"] = self.CHESS_CLASS_IDS

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        detections = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            confidence = float(box.conf[0])

            if width <= 0 or height < self.MIN_BOX_HEIGHT_PX:
                continue

            class_id = int(box.cls[0])
            names = self.model.names
            if isinstance(names, dict):
                class_name = str(names.get(class_id, "chess-piece"))
            else:
                class_name = str(names[class_id])

            detections.append(
                {
                    "box": (
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                    "x": (x1 + x2) / 2.0,
                    "y": (y1 + y2) / 2.0,
                    "width": width,
                    "height": height,
                    "confidence": confidence,
                    "class_name": class_name,
                }
            )

        return detections

    @staticmethod
    def draw_chess_detections(frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            label = f"Chess {confidence:.0%}"
            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    def _select_center_detection(self, detections, frame_width):
        max_center_offset = frame_width * 0.375
        centered = [
            detection
            for detection in detections
            if abs(detection["x"] - frame_width / 2.0) <= max_center_offset
        ]
        if not centered:
            return None

        return min(
            centered,
            key=lambda detection: (
                abs(detection["x"] - frame_width / 2.0),
                -detection["confidence"],
            ),
        )

    def _match_landmark(self, observed_x, observed_y):
        for landmark_id, (landmark_x, landmark_y) in self.true_landmarks.items():
            if (
                math.hypot(observed_x - landmark_x, observed_y - landmark_y)
                < self.LANDMARK_MATCH_DISTANCE_MM
            ):
                return landmark_id
        return None

    def _estimate_landmark_observation(
        self,
        detection,
        frame_width,
        est_x,
        est_y,
        est_theta,
    ):
        """検出枠の高さから、チェス駒までの距離と地図座標を推定する。"""
        raw_distance = (
            self.REAL_CHESS_PIECE_HEIGHT_MM
            * self.FOCAL_LENGTH_PX
            / detection["height"]
        )
        if not self.MIN_DETECTION_DISTANCE_MM <= raw_distance <= self.MAX_VIEW_DIST:
            return None

        pixel_offset = detection["x"] - frame_width / 2.0
        angle_offset = (
            pixel_offset / (frame_width / 2.0)
        ) * (self.FOV_H_RADIANS / 2.0)
        observed_x = est_x + raw_distance * math.cos(est_theta - angle_offset)
        observed_y = est_y + raw_distance * math.sin(est_theta - angle_offset)
        matched_id = self._match_landmark(observed_x, observed_y)

        distance = raw_distance
        if matched_id is not None:
            previous = self.distance_history.get(matched_id, raw_distance)
            distance = self.ALPHA * raw_distance + (1.0 - self.ALPHA) * previous
            self.distance_history[matched_id] = distance
            observed_x = est_x + distance * math.cos(est_theta - angle_offset)
            observed_y = est_y + distance * math.sin(est_theta - angle_offset)

        return {
            "distance": distance,
            "angle_offset": angle_offset,
            "observed_x": observed_x,
            "observed_y": observed_y,
            "matched_id": matched_id,
        }

    def to_map_coords(self, x_mm, y_mm):
        offset = self.MAP_SIZE // 2
        mx = int(-y_mm * self.MAP_SCALE + offset)
        my = int(-x_mm * self.MAP_SCALE + offset)
        return (mx, my)

    def from_map_coords(self, mx, my):
        offset = self.MAP_SIZE // 2
        y_mm = -(mx - offset) / self.MAP_SCALE
        x_mm = -(my - offset) / self.MAP_SCALE
        return x_mm, y_mm

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # クリックされたら強制的にマニュアルモードにする
            self.autonomous_mode = False
            tx, ty = self.from_map_coords(x, y)
            self.target_x = tx
            self.target_y = ty
            print(f"🎯 [手動] 目標地点を設定しました: X:{self.target_x:.0f}mm, Y:{self.target_y:.0f}mm")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            rx, ry = self.from_map_coords(x, y)
            self.offset_x = rx - self.current_raw_x
            self.offset_y = ry - self.current_raw_y
            self.distance_history.clear() 
            print(f"🚁 誘拐リカバリ！現在地を強制リセットしました")

    # 🚀 追加: マップ形状から安全なパトロールルートを自動計算
    def generate_patrol_waypoints(self):
        if len(self.true_landmarks) < 3:
            return []
        
        # チェス駒ランドマークの座標から外枠（最小外接矩形）を求める
        pts = np.array(list(self.true_landmarks.values()), dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        
        # マップの中心を計算
        center_x = np.mean(box[:, 0])
        center_y = np.mean(box[:, 1])

        waypoints = []
        margin = 120.0  # 壁（枠線）から120mm内側をルートにする
        
        for pt in box:
            # 各角の座標を中心に向かって margin 分だけ縮める
            vec_x = center_x - pt[0]
            vec_y = center_y - pt[1]
            length = math.hypot(vec_x, vec_y)
            if length > 0:
                wp_x = pt[0] + (vec_x / length) * margin
                wp_y = pt[1] + (vec_y / length) * margin
                waypoints.append((wp_x, wp_y))
        
        print(f"📍 自動ウェイポイントを {len(waypoints)} 箇所生成しました。")
        return waypoints

    def draw_memory_map(self, est_x, est_y, est_theta):
        map_img = self.memory_map.copy()
        
        if len(self.robot_trajectory) > 1:
            for i in range(1, len(self.robot_trajectory)):
                pt1 = self.to_map_coords(self.robot_trajectory[i-1][0], self.robot_trajectory[i-1][1])
                pt2 = self.to_map_coords(self.robot_trajectory[i][0], self.robot_trajectory[i][1])
                cv2.line(map_img, pt1, pt2, (255, 200, 50), 1) 
                
        for (mx, my) in self.persistent_mapped_pieces:
            cv2.circle(
                map_img,
                self.to_map_coords(mx, my),
                max(4, int(8 * self.MAP_SCALE)),
                (0, 0, 255),
                -1,
            )
        
        if len(self.true_landmarks) >= 2:
            landmark_pts = []
            for l_id, (lx, ly) in self.true_landmarks.items():
                landmark_pts.append(self.to_map_coords(lx, ly))
            
            hull = cv2.convexHull(np.array(landmark_pts, dtype=np.int32))
            cv2.polylines(map_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # 🚀 ウェイポイントの描画（水色の円）
        for (wx, wy) in self.waypoints:
            wp_pt = self.to_map_coords(wx, wy)
            cv2.circle(map_img, wp_pt, int(10 * self.MAP_SCALE), (255, 255, 0), 2)
                
        if self.target_x is not None and self.target_y is not None:
            t_pt = self.to_map_coords(self.target_x, self.target_y)
            cv2.drawMarker(map_img, t_pt, (0, 165, 255), cv2.MARKER_CROSS, 20, 2)

        r_pt = self.to_map_coords(est_x, est_y)
        dx = int(math.sin(est_theta) * -20)
        dy = int(math.cos(est_theta) * -20)
        cv2.circle(map_img, r_pt, 8, (0, 255, 0), -1)
        cv2.line(map_img, r_pt, (r_pt[0] + dx, r_pt[1] + dy), (0, 0, 0), 2)
        
        return map_img

    def update_fov_map(self, est_x, est_y, est_theta):
        rob_pt = self.to_map_coords(est_x, est_y)
        left_angle = est_theta + (self.FOV_H_RADIANS / 2)
        right_angle = est_theta - (self.FOV_H_RADIANS / 2)
        offset = self.MAP_SIZE // 2
        l_pt = (int(-(est_y + self.MAX_VIEW_DIST*math.sin(left_angle)) * self.MAP_SCALE + offset), int(-(est_x + self.MAX_VIEW_DIST*math.cos(left_angle)) * self.MAP_SCALE + offset))
        r_pt = (int(-(est_y + self.MAX_VIEW_DIST*math.sin(right_angle)) * self.MAP_SCALE + offset), int(-(est_x + self.MAX_VIEW_DIST*math.cos(right_angle)) * self.MAP_SCALE + offset))
        cv2.fillPoly(self.memory_map, np.array([[rob_pt, l_pt, r_pt]], dtype=np.int32), (255, 255, 255))

    def get_map_data(self):
        return {
            "landmarks": self.true_landmarks,
            "trajectory": self.robot_trajectory,
            # 既存コードとの互換性を保つため、JSONキー名は変更しない。
            "mapped_cubes": self.persistent_mapped_pieces,
        }

    def save_map_data(self, json_filename="slam_data.json"):
        """
        JSONファイルのみを出力するように修正。
        """
        data = self.get_map_data()
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"💾 マップデータを保存しました: {json_filename}")

    def load_map_data(self, json_filename="slam_data.json"):
        if not os.path.exists(json_filename):
            print(f"⚠️ {json_filename} が見つかりません。新規マッピングを開始します。")
            return False
        
        try:
            with open(json_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.true_landmarks = {int(k): tuple(v) for k, v in data["landmarks"].items()}
            mapped_pieces = data.get(
                "mapped_chess_pieces",
                data.get("mapped_cubes", []),
            )
            self.persistent_mapped_pieces = [tuple(v) for v in mapped_pieces]
            self.landmark_id_counter = len(self.true_landmarks)
            
            print(f"📂 キャッシュされたマップデータを読み込みました！ (ランドマーク数: {self.landmark_id_counter})")
            return True
        except Exception as e:
            print(f"⚠️ マップ読み込みエラー: {e}")
            return False

    def run(self, cli=None):
        own_cli = False
        own_camera_viewer = False
        if cli is None:
            cli = pycozmo.Client()
            cli.start()
            cli.connect()
            cli.wait_for_robot()
            own_cli = True

        if self.camera_viewer is None:
            self.camera_viewer = CameraViewer(
                cli,
                enhance_image=self.ENHANCE_IMAGE,
                camera_settings=self.camera_settings,
            )
            own_camera_viewer = True

        # SLAMマップとカメラを同じGUIスレッドから更新する。
        self.camera_viewer.start(background_display=False)
        self.camera_viewer.open_window()
        cv2.namedWindow(self.MAP_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.MAP_WINDOW_NAME, self.mouse_callback)
        cv2.imshow(self.MAP_WINDOW_NAME, self.memory_map)
        cv2.waitKey(1)
        print("💡 カメラ: B=明るく / N=暗く / M=自動・手動 / 0=初期値")

        print("🤖 首のキャリブレーション（基準合わせ）を実行中...")
        time.sleep(0.5)
        cli.set_head_angle(-1.0)  
        time.sleep(1.5) 
        
        cli.set_head_angle(self.TARGET_HEAD_ANGLE)  
        cli.set_lift_height(0.0)  
        time.sleep(1.5)           
        
        try:
            last_head_cmd_time = time.time()
            loop_closed = False
            transition_requested = False
            
            map_loaded = False
            if self.use_cached_map:
                map_loaded = self.load_map_data()

            # === 1. 初期スキャンフェーズ ===
            if not map_loaded:
                print(
                    "🔄 初期スキャンを開始します"
                    f"（{self.EXPECTED_LANDMARKS}個のチェス駒を探します）"
                )
                scan_start_time = time.time()
                
                while (
                    time.time() - scan_start_time < 75.0
                    and not loop_closed
                    and not transition_requested
                ):
                    current_time = time.time()
                    if current_time - last_head_cmd_time > 1.5:
                        cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                        last_head_cmd_time = current_time
                    
                    cli.drive_wheels(-8.0, 8.0) 

                    frame = self._get_latest_image()
                    if frame is not None and cli.pose is not None:
                        self.current_raw_x = cli.pose.position.x
                        self.current_raw_y = cli.pose.position.y
                        raw_theta = cli.pose.rotation.angle_z.radians
                        
                        est_x = self.current_raw_x + self.offset_x
                        est_y = self.current_raw_y + self.offset_y
                        est_theta = raw_theta + self.offset_theta
                        
                        self.robot_trajectory.append((est_x, est_y))
                        if len(self.robot_trajectory) > 2000: self.robot_trajectory.pop(0)
                            
                        self.update_fov_map(est_x, est_y, est_theta)
                        
                        detections = []
                        try:
                            detections = self.detect_chess_pieces(frame)
                            best_detection = self._select_center_detection(
                                detections,
                                frame.shape[1],
                            )
                            observation = None
                            if best_detection is not None:
                                observation = self._estimate_landmark_observation(
                                    best_detection,
                                    frame.shape[1],
                                    est_x,
                                    est_y,
                                    est_theta,
                                )

                            if observation is not None:
                                distance = observation["distance"]
                                observed_x = observation["observed_x"]
                                observed_y = observation["observed_y"]
                                matched_id = observation["matched_id"]

                                if matched_id is None:
                                    if (
                                        len(self.true_landmarks)
                                        < self.EXPECTED_LANDMARKS
                                    ):
                                        landmark_id = self.landmark_id_counter
                                        self.true_landmarks[landmark_id] = (
                                            observed_x,
                                            observed_y,
                                        )
                                        self.persistent_mapped_pieces.append(
                                            (observed_x, observed_y)
                                        )
                                        self.distance_history[landmark_id] = distance
                                        print(
                                            "📌 チェス駒を初期スキャンで捕捉！ "
                                            f"[{landmark_id}]: 距離 {distance:.0f}mm"
                                        )
                                        self.landmark_id_counter += 1
                                elif (
                                    len(self.true_landmarks)
                                    == self.EXPECTED_LANDMARKS
                                    and matched_id == 0
                                ):
                                    drift_x = (
                                        observed_x - self.true_landmarks[0][0]
                                    )
                                    drift_y = (
                                        observed_y - self.true_landmarks[0][1]
                                    )
                                    print("🎉 ループクロージャ発動！ 空間補正します！")

                                    drift_x = max(min(drift_x, 100.0), -100.0)
                                    drift_y = max(min(drift_y, 100.0), -100.0)

                                    landmark_ids = sorted(self.true_landmarks)
                                    landmark_count = len(landmark_ids)
                                    for index, landmark_id in enumerate(
                                        landmark_ids[1:],
                                        start=1,
                                    ):
                                        factor = index / landmark_count
                                        old_x, old_y = self.true_landmarks[
                                            landmark_id
                                        ]
                                        self.true_landmarks[landmark_id] = (
                                            old_x - drift_x * factor,
                                            old_y - drift_y * factor,
                                        )

                                    self.persistent_mapped_pieces = list(
                                        self.true_landmarks.values()
                                    )
                                    loop_closed = True
                        except Exception as error:
                            self._report_detection_error(error)
                        
                        preview_img = frame.copy()
                        self.draw_chess_detections(preview_img, detections)
                        self.camera_viewer.display_once(
                            frame_override=preview_img,
                        )
                        map_img = self.draw_memory_map(est_x, est_y, est_theta)
                        cv2.imshow(self.MAP_WINDOW_NAME, map_img)
                        key = self.camera_viewer.poll_key(delay_ms=1)
                        if key in (ord("q"), 27):
                            print("➡️ SLAMを終了し、自律走行へ移行します...")
                            transition_requested = True
                    time.sleep(0.05)
                print("✅ スキャン完了！（クリック待ちです）")
            else:
                print("⏭️ スキャンをスキップし、既存のマップでナビゲーションを開始します。")

            cli.drive_wheels(0.0, 0.0) 

            if transition_requested:
                return

            # 🚀 マップ構築後、自律パトロール用のウェイポイントを生成
            self.waypoints = self.generate_patrol_waypoints()
            print("💡 キーボードの 'p' を押すと自律パトロールを開始/停止します")
            
            # === 2. ナビゲーションフェーズ ===
            while True:
                current_time = time.time()
                if current_time - last_head_cmd_time > 1.5:
                    cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                    last_head_cmd_time = current_time

                display_img = self._get_latest_image()
                if display_img is not None and cli.pose is not None:
                    self.current_raw_x = cli.pose.position.x
                    self.current_raw_y = cli.pose.position.y
                    raw_theta = cli.pose.rotation.angle_z.radians
                    
                    est_x = self.current_raw_x + self.offset_x
                    est_y = self.current_raw_y + self.offset_y
                    est_theta = raw_theta + self.offset_theta
                    
                    self.robot_trajectory.append((est_x, est_y))
                    if len(self.robot_trajectory) > 2000: self.robot_trajectory.pop(0)
                        
                    self.update_fov_map(est_x, est_y, est_theta)
                    correction_triggered = False
                    
                    try:
                        detections = self.detect_chess_pieces(display_img)
                        self.draw_chess_detections(display_img, detections)
                        best_detection = self._select_center_detection(
                            detections,
                            display_img.shape[1],
                        )
                        observation = None
                        if best_detection is not None:
                            observation = self._estimate_landmark_observation(
                                best_detection,
                                display_img.shape[1],
                                est_x,
                                est_y,
                                est_theta,
                            )

                        if observation is not None:
                            distance = observation["distance"]
                            angle_offset = observation["angle_offset"]
                            observed_x = observation["observed_x"]
                            observed_y = observation["observed_y"]
                            matched_id = observation["matched_id"]

                            if matched_id is None:
                                if (
                                    len(self.true_landmarks)
                                    < self.EXPECTED_LANDMARKS
                                ):
                                    landmark_id = self.landmark_id_counter
                                    self.true_landmarks[landmark_id] = (
                                        observed_x,
                                        observed_y,
                                    )
                                    self.persistent_mapped_pieces.append(
                                        (observed_x, observed_y)
                                    )
                                    self.distance_history[landmark_id] = distance
                                    print(
                                        "🌟 新しいチェス駒を移動中に発見！ "
                                        f"[{landmark_id}]"
                                    )
                                    self.landmark_id_counter += 1
                            else:
                                # チェス駒ランドマークによるSLAM補正
                                tx, ty = self.true_landmarks[matched_id]
                                angle_to_landmark = math.atan2(
                                    ty - est_y,
                                    tx - est_x,
                                )
                                heading_error = angle_to_landmark - (
                                    est_theta - angle_offset
                                )
                                heading_error = (
                                    heading_error + math.pi
                                ) % (2 * math.pi) - math.pi

                                if abs(heading_error) < 0.3:
                                    self.offset_theta += heading_error * 0.01
                                    new_est_theta = raw_theta + self.offset_theta
                                    correct_rob_x = tx - distance * math.cos(
                                        new_est_theta - angle_offset
                                    )
                                    correct_rob_y = ty - distance * math.sin(
                                        new_est_theta - angle_offset
                                    )

                                    err_x = correct_rob_x - est_x
                                    err_y = correct_rob_y - est_y

                                    if math.hypot(err_x, err_y) < 150.0:
                                        self.offset_x += max(
                                            -3.0,
                                            min(3.0, err_x * 0.05),
                                        )
                                        self.offset_y += max(
                                            -3.0,
                                            min(3.0, err_y * 0.05),
                                        )
                                        correction_triggered = True
                    except Exception as error:
                        self._report_detection_error(error)

                    out_of_bounds = False
                    if len(self.true_landmarks) >= 3:
                        pts = np.array(list(self.true_landmarks.values()), dtype=np.float32)
                        hull = cv2.convexHull(pts)
                        dist = cv2.pointPolygonTest(hull, (float(est_x), float(est_y)), True)
                        
                        if dist < 10.0: 
                            out_of_bounds = True

                    if out_of_bounds:
                        cv2.putText(display_img, "VIRTUAL FENCE! ESCAPE", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-10.0, -10.0)
                        time.sleep(0.4)
                        self.target_x, self.target_y = None, None 
                    
                    elif any(math.hypot(est_x - tx, est_y - ty) < 60.0 for tx, ty in self.true_landmarks.values()):
                        cv2.putText(display_img, "WALL BLOCK! STOP", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-15.0, -15.0)
                        time.sleep(0.4)
                        self.target_x, self.target_y = None, None 
                    
                    elif self.target_x is not None and self.target_y is not None:
                        dist_to_target = math.hypot(self.target_x - est_x, self.target_y - est_y)
                        angle_to_target = math.atan2(self.target_y - est_y, self.target_x - est_x)
                        angle_diff = angle_to_target - est_theta
                        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                        
                        if dist_to_target < 15.0:
                            print("🎯 目標地点に到着しました！")
                            self.target_x, self.target_y = None, None
                            cli.drive_wheels(0.0, 0.0)
                            
                            # 🚀 追加: 自律モードなら数秒待って次の目的地へ
                            if self.autonomous_mode:
                                time.sleep(1.0) # 到着したら1秒停止
                        
                        elif abs(angle_diff) > 0.3:
                            cv2.putText(display_img, "ADJUSTING HEADING", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            if angle_diff > 0: cli.drive_wheels(-12.0, 12.0)
                            else: cli.drive_wheels(12.0, -12.0)
                        else:
                            cv2.putText(display_img, "NAVIGATING TO TARGET", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cli.drive_wheels(22.0, 22.0)
                            
                    # 🚀 追加: 自律パトロールモードで目標地点が空の時、自動設定する
                    elif self.autonomous_mode and len(self.waypoints) > 0:
                        self.target_x, self.target_y = self.waypoints[self.current_waypoint_index]
                        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                        print(f"🤖 [自律パトロール] 次の経由地({self.target_x:.0f}, {self.target_y:.0f})を設定しました")
                        
                    else:
                        cv2.putText(display_img, "WAITING FOR CLICK...", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cli.drive_wheels(0.0, 0.0)

                    # --- UI表示の更新 ---
                    if correction_triggered:
                        cv2.putText(display_img, "SLAM FIXED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
                    mode_text = "AUTO PATROL MODE" if self.autonomous_mode else "MANUAL (CLICK) MODE"
                    cv2.putText(display_img, mode_text, (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if self.autonomous_mode else (255, 255, 255), 2)

                    self.camera_viewer.display_once(frame_override=display_img)
                    map_img = self.draw_memory_map(est_x, est_y, est_theta)
                    cv2.imshow(self.MAP_WINDOW_NAME, map_img)
                    
                    key = self.camera_viewer.poll_key(delay_ms=1)
                    if key in (ord('q'), 27):
                        print("➡️ SLAMを終了し、自律走行へ移行します...")
                        break
                    elif key == ord('a'): 
                        self.offset_theta -= 0.2
                        print("🔄 コンパスを左に修正しました")
                    elif key == ord('d'): 
                        self.offset_theta += 0.2
                        print("🔄 コンパスを右に修正しました")
                    elif key == ord('p'):  # 🚀 追加: パトロールのオン・オフ切り替え
                        self.autonomous_mode = not self.autonomous_mode
                        if self.autonomous_mode:
                            print("🚀 自律パトロールモードを開始します！")
                        else:
                            print("🛑 自律パトロールモードを停止しました。マウス操作に戻ります。")
                            self.target_x, self.target_y = None, None

                time.sleep(0.05)
                
        finally:
            cli.drive_wheels(0.0, 0.0)
            try:
                cv2.destroyWindow(self.MAP_WINDOW_NAME)
            except cv2.error:
                pass
            if own_camera_viewer:
                self.camera_viewer.stop()
            if own_cli:
                cli.disconnect()
                cli.stop()


#----------------------------------------------------------------------
# main.py などの呼び出し側から1〜2行で使えるようにするための
# エントリーポイント関数。CozmoSlamControllerの生成・実行・保存の
# 一連の処理をこの関数の中にまとめている。
#----------------------------------------------------------------------
def run_slam_mapping(
    cli,
    model_path=DEFAULT_CHESS_MODEL_PATH,
    use_cached_map=False,
    json_filename="slam_data.json",
    camera_settings=None,
    camera_viewer=None,
):
    """
    SLAMマッピングを開始から保存まで一括で実行する。

    引数:
        cli: 接続済みのCozmoクライアント(pycozmo.Client)
        model_path: YOLOモデルファイルのパス
        use_cached_map: Trueの場合、既存のマップデータを読み込んで再利用する
        json_filename: マップデータの保存先ファイル名
        camera_settings: Cozmoカメラへ適用する露出・ゲイン設定
        camera_viewer: SLAMと自律走行で共有するCameraViewer
    """
    print("\n=== SLAMマッピングを開始します ===")
    slam_bot = CozmoSlamController(
        model_path=model_path,
        use_cached_map=use_cached_map,
        camera_settings=camera_settings,
        camera_viewer=camera_viewer,
    )
    try:
        slam_bot.run(cli=cli)
    except KeyboardInterrupt:
        print("\n\n🛑 マッピングを終了し、自律走行へ移行します...")
    finally:
        slam_bot.save_map_data(json_filename=json_filename)
