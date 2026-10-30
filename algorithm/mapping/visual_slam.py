# ----------------------------------------------------------------------
# visual_slam.py
# チェス駒をランドマークに使うCozmoのVisual SLAM
# ----------------------------------------------------------------------

import os
os.environ["YOLO_OFFLINE"] = "True"

import json
import math
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycozmo
import torch
torch.set_num_threads(4)
from ultralytics import YOLO

from camera.camera_movie import (
    CameraViewer,
    normalize_camera_settings,
)


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent  # algorithm/mapping/ の2階層上（main.pyの場所）
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
        use_cached_map=True,
        confidence_threshold=0.35,
        inference_size=320,
        enhance_image=True,
        camera_settings=None,
        camera_viewer=None,
        piece_height_mm=20.0,
        focal_length_px=235.0,
        calibration_width_px=320,
        json_filename="slam_data.json",
        max_detection_distance_mm=1000.0,
        force_remap=False,
    ):
        model_path = self._resolve_model_path(model_path)
        print(f"チェス駒AIモデル({model_path})を準備中...")
        self.model = YOLO(str(model_path))
        self.use_cached_map = use_cached_map
        # use_cached_mapは旧呼び出しとの互換用。新仕様ではファイルの有無を優先する。
        self.force_remap = bool(force_remap)
        self._map_loaded_from_json = False
        self._map_created_this_run = False
        self._map_observation_correction_enabled = True
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.INFERENCE_SIZE = inference_size
        self.ENHANCE_IMAGE = enhance_image
        self.camera_settings = normalize_camera_settings(camera_settings)
        self.camera_viewer = camera_viewer
        self._camera_adjustment_completed = False
        self.CHESS_CLASS_IDS = self._find_chess_class_ids()

        # --- 約20mmのチェス駒を使う教室デスク向けパラメータ ---
        # 駒の実寸が異なる場合は、この高さだけ実物に合わせて変更する。
        self.REAL_CHESS_PIECE_HEIGHT_MM = float(piece_height_mm)
        self.FOCAL_LENGTH_PX = float(focal_length_px)
        # 焦点距離235pxは幅320pxの画像に対する仮値。実測で校正すること。
        self.CALIBRATION_WIDTH_PX = float(calibration_width_px)
        if min(self.REAL_CHESS_PIECE_HEIGHT_MM, self.FOCAL_LENGTH_PX,
               self.CALIBRATION_WIDTH_PX) <= 0:
            raise ValueError("駒の高さ・焦点距離・校正画像幅は正の値にしてください")
        self.json_filename = str(self._resolve_map_path(json_filename))
        self.TARGET_HEAD_ANGLE = -0.15
        self.distance_scales = {}  # 検出クラスごとの実測距離補正。焦点距離とは独立させる。
        self.calibration_filename = str(Path(self.json_filename).with_suffix(".camera_calibration.json"))
        self._load_distance_scales()
        self.FOV_H_RADIANS = math.radians(60)
        self.MAX_VIEW_DIST = float(max_detection_distance_mm)
        if not math.isfinite(self.MAX_VIEW_DIST) or not 100 <= self.MAX_VIEW_DIST <= 2000:
            raise ValueError("検出上限距離は100〜2000mmにしてください")
        self.MIN_DETECTION_DISTANCE_MM = 35.0
        self.EXPECTED_LANDMARKS = 6
        self.LANDMARK_MATCH_DISTANCE_MM = 130.0
        self.MATCH_AMBIGUITY_MM = 15.0
        self.CANDIDATE_MATCH_DISTANCE_MM = 45.0
        self.EXISTING_LANDMARK_MIN_DISTANCE_MM = 120.0
        self.MIN_CONFIRMATIONS = 3
        self.MAX_HEADING_ERROR = math.radians(40)
        self.candidate_landmarks = []
        self.last_observations = []
        self.last_correction = "Waiting for landmarks"
        self.map_center_x = self.map_center_y = 0.0
        self.MIN_BOX_HEIGHT_PX = 6
        self.MAP_SIZE = 450            
        self.MAP_SCALE = 0.5           
        
        # --- 状態変数 ---
        self.memory_map = np.full((self.MAP_SIZE, self.MAP_SIZE, 3), 127, dtype=np.uint8)
        # 確定した駒の座標はこの辞書だけで管理する（ID → (X, Y)）。
        self.true_landmarks = {}
        self.landmark_id_counter = 0
        self.offset_x, self.offset_y, self.offset_theta = 0.0, 0.0, 0.0
        self.current_raw_theta = 0.0
        self.current_raw_x, self.current_raw_y = 0.0, 0.0 
        self.target_x = None
        self.target_y = None
        self._last_detection_error_time = 0.0
        self.robot_trajectory = [] 
        self.distance_history = {}
        # 距離を過去フレームと平均すると走行中に遅れるため、補正には最新値を使う。
        self.ALPHA = 0.2 

        # 🚀 追加: 自律走行用の変数
        self.autonomous_mode = False
        self.show_camera_window = True
        self.waypoints = []
        self.current_waypoint_index = 0

    @staticmethod
    def _resolve_map_path(filename):
        """起動したカレントディレクトリではなく、プロジェクト直下を基準にする。"""
        path = Path(filename).expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

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

        keywords = ("chess", "piece", "pawn", "rook", "knight", "bishop", "queen", "king", "駒")
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
                    "clipped": x1 <= 1 or y1 <= 1 or x2 >= frame.shape[1]-1 or y2 >= frame.shape[0]-1,
                }
            )

        return detections

    @staticmethod
    def draw_chess_detections(frame, detections):
        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            confidence = detection["confidence"]
            distance = detection.get("distance")
            suffix = f" ~{distance:.0f}mm" if distance is not None else ""
            label = f"{detection.get('map_status', 'Chess')}{suffix} {confidence:.0%}"
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

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _get_estimated_pose(self, raw_x, raw_y, raw_theta):
        """生の走行座標を、回転＋平行移動で地図座標へ変換する。"""
        c, sn = math.cos(self.offset_theta), math.sin(self.offset_theta)
        return (c * raw_x - sn * raw_y + self.offset_x,
                sn * raw_x + c * raw_y + self.offset_y,
                self._wrap_angle(raw_theta + self.offset_theta))

    def _set_estimated_pose(self, raw_x, raw_y, raw_theta, x, y, theta):
        # 向きの補正後も、現在位置を基準に今後の移動方向が整合するようにする。
        self.offset_theta = self._wrap_angle(theta - raw_theta)
        c, sn = math.cos(self.offset_theta), math.sin(self.offset_theta)
        self.offset_x = x - (c * raw_x - sn * raw_y)
        self.offset_y = y - (sn * raw_x + c * raw_y)

    def _estimate_landmark_observation(self, detection, frame_width,
                                       est_x, est_y, est_theta, check_range=True):
        """単眼の概算距離。駒の実寸・焦点距離・首角度に誤差があると距離もずれる。

        枠の高さから得るのは光軸方向の奥行き。横方向成分も含めて距離にする。
        小さい首角度を前提とする近似で、正確な床面距離には実機校正が必要。
        """
        detection.pop("distance", None)
        if check_range and detection["class_name"] not in self.distance_scales:
            detection["map_status"] = "Uncalibrated / press C"
            return None
        if detection.get("clipped", False):
            detection["map_status"] = "Clipped box / retry"
            return None
        height = detection["height"]
        if height < self.MIN_BOX_HEIGHT_PX:
            detection["map_status"] = "Box too small"
            return None
        focal = self.FOCAL_LENGTH_PX * frame_width / self.CALIBRATION_WIDTH_PX
        scale = self.distance_scales.get(detection["class_name"], 1.0)
        depth = self.REAL_CHESS_PIECE_HEIGHT_MM * focal / height * scale
        lateral = -(detection["x"] - frame_width / 2.0) * depth / focal
        distance = math.hypot(depth, lateral)
        if not math.isfinite(distance):
            detection["map_status"] = "Invalid distance"
            return None
        detection["distance"] = distance
        if check_range and not self.MIN_DETECTION_DISTANCE_MM <= distance <= self.MAX_VIEW_DIST:
            detection["map_status"] = "Out of range"
            return None
        bearing = math.atan2(lateral, depth)
        return {"distance": distance, "angle_offset": -bearing,
                "bearing": bearing, "local_x": depth, "local_y": lateral,
                "observed_x": est_x + distance * math.cos(est_theta + bearing),
                "observed_y": est_y + distance * math.sin(est_theta + bearing),
                "matched_id": None, "detection": detection}

    def _calibration_signature(self):
        return {"piece_height_mm": self.REAL_CHESS_PIECE_HEIGHT_MM,
                "focal_length_px": self.FOCAL_LENGTH_PX,
                "calibration_width_px": self.CALIBRATION_WIDTH_PX,
                "head_angle": self.TARGET_HEAD_ANGLE}

    def _load_distance_scales(self):
        path = Path(self.calibration_filename)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("camera") != self._calibration_signature():
                print("⚠️ カメラ条件が変わったため、以前の距離校正は使用しません。Cキーで再校正してください。")
                return
            scales = data["class_scales"]
            if not isinstance(scales, dict) or not all(
                    isinstance(k, str) and isinstance(v, (int, float))
                    and math.isfinite(v) and 0.1 <= v <= 10.0 for k, v in scales.items()):
                raise ValueError("校正倍率が不正です")
            self.distance_scales = dict(scales)
            print(f"📏 距離校正を読み込みました: {self.distance_scales}")
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            print(f"⚠️ 距離校正を読み込めません: {error}")

    def _apply_distance_calibration(self, samples, measured_mm):
        """停止中に同じ中央の駒を5回測り、実測距離で倍率を決める。"""
        if not math.isfinite(measured_mm) or not 50 <= measured_mm <= 1200:
            raise ValueError("実測距離は50〜1200mmで入力してください")
        if len(samples) < 5:
            raise ValueError("中央に駒を1個置き、5回以上安定して検出されるまで待ってください")
        names = {name for name, distance in samples}
        values = np.array([distance for name, distance in samples], dtype=float)
        median = float(np.median(values))
        if len(names) != 1 or not np.all(np.isfinite(values)) or median <= 0:
            raise ValueError("同じクラスの駒を連続して検出してください")
        if np.ptp(values) / median > 0.15:
            raise ValueError("検出枠が不安定です。駒を固定し、明るさを調整してください")
        name = next(iter(names))
        factor = self.distance_scales.get(name, 1.0) * measured_mm / median
        if not 0.1 <= factor <= 10.0:
            raise ValueError("補正倍率が範囲外です。実測距離と対象の駒を確認してください")
        scales = dict(self.distance_scales)
        scales[name] = factor
        path = Path(self.calibration_filename).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
                temporary = f.name
                json.dump({"camera": self._calibration_signature(), "class_scales": scales},
                          f, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        self.distance_scales = scales
        # 旧倍率で作った地図を同じ座標として扱わない。
        self.use_cached_map = False
        self.true_landmarks.clear()
        self.candidate_landmarks.clear()
        self.distance_history.clear()
        self.landmark_id_counter = 0
        self.robot_trajectory.clear()
        self.last_observations = []
        self.memory_map.fill(127)
        print(f"📏 {name}: 推定{median:.0f}mm → 実測{measured_mm:.0f}mm、倍率{factor:.3f}で校正しました。")
        return name, factor

    def _associate_observations(self, observations, pose):
        """全検出を一対一で対応付ける。候補が曖昧な駒は補正にも新規登録にも使わない。"""
        x, y, theta = pose
        proposals = []
        for index, obs in enumerate(observations):
            choices = []
            for landmark_id, (lx, ly) in self.true_landmarks.items():
                error = math.hypot(obs["observed_x"] - lx, obs["observed_y"] - ly)
                if error <= self.LANDMARK_MATCH_DISTANCE_MM:
                    choices.append((error, landmark_id))
            choices.sort()
            obs["near_existing"] = bool(choices)
            if not choices:
                continue
            if len(choices) > 1 and choices[1][0] - choices[0][0] < self.MATCH_AMBIGUITY_MM:
                obs["detection"]["map_status"] = "Ambiguous"
                continue
            error, landmark_id = choices[0]
            lx, ly = self.true_landmarks[landmark_id]
            heading_error = self._wrap_angle(math.atan2(ly-y, lx-x) - theta - obs["bearing"])
            if (abs(heading_error) <= self.MAX_HEADING_ERROR and
                    abs(math.hypot(lx-x, ly-y) - obs["distance"]) <= self.LANDMARK_MATCH_DISTANCE_MM):
                proposals.append((error, index, landmark_id))
            else:
                obs["detection"]["map_status"] = "Match rejected"

        # 同一IDに複数の枠が競合する場合も、明確な最小誤差だけを採用する。
        used = set()
        for error, index, landmark_id in sorted(proposals):
            competitors = [e for e, i, lid in proposals if lid == landmark_id and i != index]
            if landmark_id in used or (competitors and min(competitors) - error < self.MATCH_AMBIGUITY_MM):
                observations[index]["detection"]["map_status"] = "Ambiguous"
                continue
            observations[index]["matched_id"] = landmark_id
            used.add(landmark_id)

    def _correct_pose(self, observations, raw_pose, pose):
        matched = [o for o in observations if o["matched_id"] is not None]
        self.last_correction = "No reliable match"
        if not matched:
            return pose, False
        x, y, theta = pose
        if len(matched) >= 2:
            # 2個以上の既知の駒: ロボット基準点群を地図点群へ剛体変換する。
            local = np.array([(o["local_x"], o["local_y"]) for o in matched])
            world = np.array([self.true_landmarks[o["matched_id"]] for o in matched])
            a, b = local - local.mean(axis=0), world - world.mean(axis=0)
            baseline = max(np.linalg.norm(p-q) for p in world for q in world)
            local_baseline = max(np.linalg.norm(p-q) for p in local for q in local)
            if baseline < 80.0 or local_baseline < 80.0:
                self.last_correction = "Landmarks too close for pose fix"
                return pose, False
            angle = math.atan2(float(np.sum(a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0])),
                               float(np.sum(a*b)))
            c, sn = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -sn], [sn, c]])
            translation = world.mean(axis=0) - rotation @ local.mean(axis=0)
            residual = np.sqrt(np.mean(np.sum((local @ rotation.T + translation - world)**2, axis=1)))
            ex, ey = float(translation[0] - x), float(translation[1] - y)
            heading_error = self._wrap_angle(angle - theta)
            if residual > 30.0 or math.hypot(ex, ey) > 100.0 or abs(heading_error) > self.MAX_HEADING_ERROR:
                self.last_correction = "Pose fix rejected (inconsistent geometry)"
                return pose, False
            gain, max_shift, max_turn = 0.35, 12.0, math.radians(3)
            self.last_correction = f"Pose fix: {len(matched)} landmarks"
        else:
            # 1個だけでは(x,y,向き)の3自由度は一意に決まらない。
            # 走行推定を基準に、距離方向の位置と向きだけを小さく補正する。
            obs = matched[0]
            lx, ly = self.true_landmarks[obs["matched_id"]]
            predicted_distance = math.hypot(lx-x, ly-y)
            if predicted_distance < 1.0:
                return pose, False
            radial_error = predicted_distance - obs["distance"]
            ex = radial_error * (lx-x) / predicted_distance
            ey = radial_error * (ly-y) / predicted_distance
            heading_error = self._wrap_angle(math.atan2(ly-y, lx-x) - theta - obs["bearing"])
            gain, max_shift, max_turn = 0.2, 6.0, math.radians(2)
            self.last_correction = "Partial fix: 1 landmark + odometry"
        shift = math.hypot(ex, ey)
        scale = min(gain, max_shift / max(shift, 1e-9))
        turn = max(-max_turn, min(max_turn, gain * heading_error))
        corrected = (x + ex * scale, y + ey * scale, self._wrap_angle(theta + turn))
        self._set_estimated_pose(*raw_pose, *corrected)
        return corrected, shift > 0.1 or abs(turn) > math.radians(0.01)

    def process_detections(self, detections, frame_width, raw_x, raw_y, raw_theta,
                           allow_pose_correction=True, allow_new_landmarks=True):
        """スキャン・移動中で共用する、距離推定→対応付け→補正→新規登録。"""
        raw_pose = (raw_x, raw_y, raw_theta)
        pose = self._get_estimated_pose(*raw_pose)
        observations = []
        for detection in detections:
            detection["map_status"] = "Chess"
            obs = self._estimate_landmark_observation(detection, frame_width, *pose)
            if obs is not None:
                observations.append(obs)
        self._associate_observations(observations, pose)
        if allow_pose_correction:
            # 単眼の1点だけでは距離誤差と位置・向きの誤差を分離できない。
            # 日常の補正は2点以上を確実に対応付けられたときだけ行う。
            matched_count = sum(o["matched_id"] is not None for o in observations)
            if matched_count >= 2:
                pose, corrected = self._correct_pose(observations, raw_pose, pose)
            else:
                corrected = False
                self.last_correction = "Need 2 landmarks for pose fix"
        else:
            corrected = False
            self.last_correction = "Mapping: fixed odometry frame"
        now = time.monotonic()
        # 3回続けて観測できた候補のみを赤印に確定する。
        self.candidate_landmarks = [p for p in self.candidate_landmarks if now-p["time"] < 3.0]
        previous_candidates = list(self.candidate_landmarks)
        used_candidates = set()
        for obs in observations:
            detection = obs["detection"]
            landmark_id = obs["matched_id"]
            if landmark_id is not None:
                detection["map_status"] = f"ID {landmark_id+1}"
                self.distance_history[landmark_id] = obs["distance"]
                continue
            if obs["near_existing"]:
                if detection["map_status"] == "Chess":
                    detection["map_status"] = "Ambiguous"
                continue
            if not allow_new_landmarks:
                detection["map_status"] = "Not in saved map"
                continue
            if len(self.true_landmarks) >= self.EXPECTED_LANDMARKS:
                detection["map_status"] = "Map full / unmatched"
                continue
            x = pose[0] + obs["distance"] * math.cos(pose[2] + obs["bearing"])
            y = pose[1] + obs["distance"] * math.sin(pose[2] + obs["bearing"])
            # 補正で地図座標が変わった場合にも既存IDの重複を作らない。
            if any(math.hypot(x-lx, y-ly) < self.EXISTING_LANDMARK_MIN_DISTANCE_MM
                   for lx, ly in self.true_landmarks.values()):
                detection["map_status"] = "Near existing / retry"
                continue
            choices = sorted((math.hypot(x-p["x"], y-p["y"]), i)
                             for i, p in enumerate(previous_candidates)
                             if i not in used_candidates)
            if choices and choices[0][0] <= self.CANDIDATE_MATCH_DISTANCE_MM:
                i = choices[0][1]
                used_candidates.add(i)
                candidate = previous_candidates[i]
                count = candidate["count"] + 1
                candidate["x"] += (x-candidate["x"]) / count
                candidate["y"] += (y-candidate["y"]) / count
                candidate.update(count=count, time=now, seen=True)
            else:
                candidate = dict(x=x, y=y, count=1, time=now, seen=True)
                self.candidate_landmarks.append(candidate)
            detection["map_status"] = f"Pending {candidate['count']}/{self.MIN_CONFIRMATIONS}"
            if candidate["count"] >= self.MIN_CONFIRMATIONS:
                landmark_id = self.landmark_id_counter
                self.landmark_id_counter += 1
                self.true_landmarks[landmark_id] = (candidate["x"], candidate["y"])
                self.distance_history[landmark_id] = obs["distance"]
                obs["matched_id"] = landmark_id
                detection["map_status"] = f"ID {landmark_id+1}"
                self.candidate_landmarks.remove(candidate)
                print(f"📌 駒 {landmark_id+1} を記録 ({len(self.true_landmarks)}/"
                      f"{self.EXPECTED_LANDMARKS}) 推定距離 {obs['distance']:.0f}mm")
        # 途中で見失った候補は確認回数をリセットする。
        self.candidate_landmarks = [p for p in self.candidate_landmarks if p.pop("seen", False)]
        self.last_observations = observations
        return (*pose, corrected)

    def _fit_map_view(self, est_x, est_y):
        """ロボットと全ランドマークが見える縮尺に合わせる（検出上限600mmも表示）。"""
        points = [(0.0, 0.0), (est_x, est_y)] + list(self.true_landmarks.values())
        if self.target_x is not None and self.target_y is not None:
            points.append((self.target_x, self.target_y))
        xs, ys = zip(*points)
        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        scale = (self.MAP_SIZE-80) / max(700.0, max(xs)-min(xs), max(ys)-min(ys))
        if (abs(cx-self.map_center_x) > 1 or abs(cy-self.map_center_y) > 1
                or abs(scale-self.MAP_SCALE) > 0.001):
            self.map_center_x, self.map_center_y, self.MAP_SCALE = cx, cy, scale
            # 旧縮尺のFOV画像を使うと座標がずれるため、背景だけ再構築する。
            self.memory_map.fill(127)

    def to_map_coords(self, x_mm, y_mm):
        offset = self.MAP_SIZE // 2
        return (int(-(y_mm-self.map_center_y)*self.MAP_SCALE + offset),
                int(-(x_mm-self.map_center_x)*self.MAP_SCALE + offset))

    def from_map_coords(self, mx, my):
        offset = self.MAP_SIZE // 2
        return (-(my-offset)/self.MAP_SCALE + self.map_center_x,
                -(mx-offset)/self.MAP_SCALE + self.map_center_y)

    def _capture_stopped_frame(self, cli):
        # 推論中の回転と画像・poseの時間差を減らすため、一旦停止して観測する。
        # CameraViewerに撮影時刻APIがないため、完全な時刻同期ではない。
        cli.drive_wheels(0.0, 0.0)
        time.sleep(0.15)
        return self._capture_frame(cli)

    def _capture_frame(self, cli):
        """画像と推論前のposeを取得する。回転・停止の命令はここでは送らない。"""
        frame = self._get_latest_image()
        if frame is None or cli.pose is None:
            return None
        pose = cli.pose
        self.current_raw_x = pose.position.x
        self.current_raw_y = pose.position.y
        self.current_raw_theta = pose.rotation.angle_z.radians
        return frame.copy()

    def _calibrate_distance_interactively(self, cli):
        """OpenCV画面内で距離を入力する。入力中の0を露出リセットに渡さない。"""
        print("📏 距離校正: 駒を1個だけ画面中央に置き、Cozmo車体中心から駒の底面中心まで")
        print("   机に沿って測った距離をmmで入力してください。例: 300 → Enter。")
        print("   姿勢・種類・高さをそろえた駒を使用してください。Escで調整画面へ戻ります。")
        typed = ""
        samples = []
        try:
            while True:
                cli.drive_wheels(0.0, 0.0)
                frame = self._get_latest_image()
                if frame is not None:
                    frame = frame.copy()
                    try:
                        detections = self.detect_chess_pieces(frame)
                        centered = [d for d in detections if abs(d["x"]-frame.shape[1]/2) < frame.shape[1]*0.10
                                    and not d.get("clipped", False)]
                        if len(centered) == 1:
                            detection = centered[0]
                            obs = self._estimate_landmark_observation(detection, frame.shape[1], 0, 0, 0,
                                                                      check_range=False)
                            if obs is not None:
                                if samples and samples[-1][0] != detection["class_name"]:
                                    samples.clear()
                                samples.append((detection["class_name"], obs["distance"]))
                                samples = samples[-9:]
                                detection["map_status"] = f"CAL {detection['class_name']} n={len(samples)}"
                            else:
                                samples.clear()
                        else:
                            samples.clear()
                        self.draw_chess_detections(frame, detections)
                    except Exception as error:
                        samples.clear()
                        self._report_detection_error(error)
                    cv2.putText(frame, f"Measured mm: {typed}_", (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(frame, "1 piece at center. ENTER: apply", (8, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                    self.camera_viewer.display_once(frame_override=frame)
                else:
                    samples.clear()
                canvas = self.memory_map.copy()
                cv2.putText(canvas, f"Actual distance: {typed}_ mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.putText(canvas, f"Samples: {len(samples)} (need 5)   ESC: back", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                cv2.imshow(self.MAP_WINDOW_NAME, canvas)
                # 数値入力専用。CameraViewer.poll_key()を使うと0が明るさ初期化になる。
                key = cv2.waitKey(20) & 0xff
                if key in (27, ord("q")):
                    return False
                if ord("0") <= key <= ord("9") and len(typed) < 4:
                    typed += chr(key)
                elif key in (8, 127):
                    typed = typed[:-1]
                elif key in (10, 13):
                    try:
                        self._apply_distance_calibration(samples, float(typed))
                        return True
                    except (ValueError, OSError) as error:
                        print(f"⚠️ 校正を適用できません: {error}")
                time.sleep(0.01)
        finally:
            cli.drive_wheels(0.0, 0.0)

    def _wait_for_camera_adjustment(self, cli):
        """停止したまま明るさを調整し、Enterを押したらマッピングへ進む。

        B/N/M/0の処理は既存CameraViewer.poll_key()に任せる。
        同じViewerを使い続けるので、調整後に露出設定を初期化しない。
        """
        print("💡 マッピング前の明るさ調整です。Cozmoは停止して待機します。")
        print("   カメラ画面をクリック → B:明るく / N:暗く / M:自動・手動 / 0:初期値")
        print("   C:実測距離による校正。初回は必須です。異なる検出クラスはそれぞれ校正してください。")
        print("   調整が終わったら Enter で開始します。Q / EscでSLAMを手動終了します。")
        try:
            while True:
                cli.drive_wheels(0.0, 0.0)
                frame = self._get_latest_image()
                ready = frame is not None and cli.pose is not None
                if frame is not None:
                    preview = frame.copy()
                    cv2.putText(preview, "Adjust brightness: B / N", (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(preview, "C: distance calibration / ENTER: start", (8, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    self.camera_viewer.display_once(frame_override=preview)

                # マップ画面も更新するが、この段階では駒の検出・登録をしない。
                map_img = self.memory_map.copy()
                status = ("C: calibrate distance / ENTER: start" if self.distance_scales else "Press C to calibrate distance first") if ready else "Waiting for camera / robot pose..."
                cv2.putText(map_img, status, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                cv2.putText(map_img, "B: brighter   N: darker", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.imshow(self.MAP_WINDOW_NAME, map_img)
                key = self.camera_viewer.poll_key(delay_ms=20)
                if key in (ord("q"), 27):
                    print("⏹️ 明るさ調整を中断し、SLAMを終了します。")
                    return False
                if key in (ord("c"), ord("C")) and ready:
                    self._calibrate_distance_interactively(cli)
                    continue
                if key in (10, 13):  # Enter（環境によってLFまたはCR）
                    if ready and self.distance_scales:
                        print("✅ 明るさ調整完了。マッピングを開始します。")
                        return True
                    print("⚠️ 映像・位置情報を確認し、Cキーで距離校正してからEnterを押してください。")
                time.sleep(0.01)
        finally:
            cli.drive_wheels(0.0, 0.0)

    def _scan_landmarks(self, cli):
        """探索は連続回転、新しい駒の測距は停止中に行う。6個で次フェーズへ。"""
        print(f"🔄 回転探索を開始します ({len(self.true_landmarks)}/"
              f"{self.EXPECTED_LANDMARKS})。新しい駒を見つけたら停止して測距します。")
        last_head_time = time.monotonic()
        last_report_time = last_head_time
        observing = False
        settle_until = 0.0
        # 前回の0.25秒パルスの問題を繰り返さないよう、探索は最低1秒連続で回す。
        search_until = last_head_time + 1.0
        observation_count = 0
        try:
            while len(self.true_landmarks) < self.EXPECTED_LANDMARKS:
                now = time.monotonic()
                if now - last_head_time > 1.5:
                    cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                    last_head_time = now

                frame = self._capture_frame(cli)
                if frame is None:
                    cli.drive_wheels(0.0, 0.0)
                    observing = False
                    self.candidate_landmarks.clear()
                    search_until = now + 1.0
                else:
                    cli.drive_wheels(0.0, 0.0) if observing else cli.drive_wheels(-5.0, 5.0)
                    est_x, est_y, est_theta = self._get_estimated_pose(
                        self.current_raw_x, self.current_raw_y, self.current_raw_theta)
                    detections = []
                    try:
                        if not observing or now >= settle_until:
                            detections = self.detect_chess_pieces(frame)
                        if observing and now >= settle_until:
                            # 回転中の画像を登録せず、停止後の3回の観測だけで記録する。
                            est_x, est_y, est_theta, _ = self.process_detections(
                                detections, frame.shape[1], self.current_raw_x,
                                self.current_raw_y, self.current_raw_theta,
                                allow_pose_correction=False)
                            observation_count += 1
                            if observation_count >= 3:
                                observing = False
                                search_until = time.monotonic() + 1.0
                                self.candidate_landmarks.clear()
                        elif not observing:
                            proposals = []
                            for detection in detections:
                                obs = self._estimate_landmark_observation(
                                    detection, frame.shape[1], est_x, est_y, est_theta)
                                if obs is not None:
                                    proposals.append(obs)
                            self._associate_observations(proposals, (est_x, est_y, est_theta))
                            self.last_observations = []
                            self.last_correction = "Scanning / C: recalibrate"
                            if now >= search_until and any(not o["near_existing"] for o in proposals):
                                cli.drive_wheels(0.0, 0.0)
                                observing = True
                                settle_until = time.monotonic() + 0.5
                                observation_count = 0
                                self.candidate_landmarks.clear()
                                self.last_correction = "Stopped: measuring new landmark"
                    except Exception as error:
                        cli.drive_wheels(0.0, 0.0)
                        self._report_detection_error(error)
                        observing = False
                        search_until = time.monotonic() + 1.0

                    # 6個そろったら、表示処理を待たず直ちに回転を止める。
                    if len(self.true_landmarks) == self.EXPECTED_LANDMARKS:
                        cli.drive_wheels(0.0, 0.0)
                    self.robot_trajectory.append((est_x, est_y))
                    self.robot_trajectory = self.robot_trajectory[-2000:]
                    self.draw_chess_detections(frame, detections)
                    self.camera_viewer.display_once(frame_override=frame)
                    cv2.imshow(self.MAP_WINDOW_NAME,
                               self.draw_memory_map(est_x, est_y, est_theta))

                key = self.camera_viewer.poll_key(delay_ms=1)
                if key in (ord("q"), 27):
                    print(f"⏹️ 初期スキャンを手動終了しました "
                          f"({len(self.true_landmarks)}/{self.EXPECTED_LANDMARKS})")
                    return False
                if key in (ord("c"), ord("C")):
                    cli.drive_wheels(0.0, 0.0)
                    if not self._wait_for_camera_adjustment(cli):
                        return False
                    observing = False
                    search_until = time.monotonic() + 1.0
                    self.candidate_landmarks.clear()
                if now - last_report_time >= 30.0:
                    print(f"🔎 探索継続中: {len(self.true_landmarks)}/"
                          f"{self.EXPECTED_LANDMARKS}個。Uncalibrated表示はCで校正、Out of rangeは距離を確認してください。")
                    last_report_time = now
                if len(self.true_landmarks) < self.EXPECTED_LANDMARKS:
                    time.sleep(0.01)
            print("✅ チェス駒6個の登録が完了しました。次のフェーズへ進みます。")
            return True
        finally:
            # 正常完了・手動終了・例外のいずれでも回転を止める。
            cli.drive_wheels(0.0, 0.0)

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
            _, _, theta = self._get_estimated_pose(
                self.current_raw_x, self.current_raw_y, self.current_raw_theta)
            self._set_estimated_pose(self.current_raw_x, self.current_raw_y,
                                     self.current_raw_theta, rx, ry, theta)
            self.candidate_landmarks.clear()
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
                actual_margin = min(margin, length * 0.45)
                wp_x = pt[0] + (vec_x / length) * actual_margin
                wp_y = pt[1] + (vec_y / length) * actual_margin
                waypoints.append((float(wp_x), float(wp_y)))
        
        print(f"📍 自動ウェイポイントを {len(waypoints)} 箇所生成しました。")
        return waypoints

    def draw_memory_map(self, est_x, est_y, est_theta):
        self._fit_map_view(est_x, est_y)
        self.update_fov_map(est_x, est_y, est_theta)
        map_img = self.memory_map.copy()
        
        if len(self.robot_trajectory) > 1:
            step = max(1, len(self.robot_trajectory) // 100)
            pts = [self.to_map_coords(x, y) for x, y in self.robot_trajectory[::step]]
            cv2.polylines(map_img, [np.array(pts, dtype=np.int32)], isClosed=False, color=(255, 200, 50), thickness=1) 
                
        if len(self.true_landmarks) >= 2:
            landmark_pts = []
            for l_id, (lx, ly) in self.true_landmarks.items():
                landmark_pts.append(self.to_map_coords(lx, ly))
            
            hull = cv2.convexHull(np.array(landmark_pts, dtype=np.int32))
            cv2.polylines(map_img, [hull], isClosed=True, color=(0, 255, 0), thickness=2)
        
        for landmark_id, (lx, ly) in self.true_landmarks.items():
            pt = self.to_map_coords(lx, ly)
            cv2.circle(map_img, pt, 5, (0, 0, 255), -1)
            cv2.putText(map_img, str(landmark_id+1), (pt[0]+7, pt[1]-7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1)
        for obs in self.last_observations:
            landmark_id = obs["matched_id"]
            if landmark_id in self.true_landmarks:
                pt = self.to_map_coords(*self.true_landmarks[landmark_id])
                cv2.putText(map_img, f"~{obs['distance']:.0f}mm", (pt[0]+7, pt[1]+13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 140), 1)

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
        
        cv2.putText(map_img, f"Chess: {len(self.true_landmarks)}/{self.EXPECTED_LANDMARKS}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 2)
        cam_str = "ON" if getattr(self, "show_camera_window", True) else "OFF [V]"
        mode_str = "AUTO" if self.autonomous_mode else "MANUAL"
        cv2.putText(map_img, f"Mode: {mode_str} | Cam: {cam_str}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if self.autonomous_mode else (180, 180, 180), 1)
        cv2.putText(map_img, self.last_correction, (10, self.MAP_SIZE-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1)
        # 縮尺を変えても距離感を確認できる実寸の物差し（ロボットの丸印は縮尺外）。
        ruler_length = max(1, int(100*self.MAP_SCALE))
        cv2.line(map_img, (10, self.MAP_SIZE-40), (10+ruler_length, self.MAP_SIZE-40), (0, 0, 0), 2)
        cv2.putText(map_img, "100 mm", (10, self.MAP_SIZE-48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return map_img

    def update_fov_map(self, est_x, est_y, est_theta):
        rob_pt = self.to_map_coords(est_x, est_y)
        left_angle = est_theta + (self.FOV_H_RADIANS / 2)
        right_angle = est_theta - (self.FOV_H_RADIANS / 2)
        l_pt = self.to_map_coords(est_x + self.MAX_VIEW_DIST*math.cos(left_angle),
                                  est_y + self.MAX_VIEW_DIST*math.sin(left_angle))
        r_pt = self.to_map_coords(est_x + self.MAX_VIEW_DIST*math.cos(right_angle),
                                  est_y + self.MAX_VIEW_DIST*math.sin(right_angle))
        cv2.fillPoly(self.memory_map, np.array([[rob_pt, l_pt, r_pt]], dtype=np.int32), (255, 255, 255))

    def get_map_data(self):
        """確定した駒の座標はlandmarksだけに保存する（形式バージョン3）。"""
        return {
            "landmarks": dict(self.true_landmarks),
            "trajectory": self.robot_trajectory,
            "schema_version": 3,
            "expected_landmarks": self.EXPECTED_LANDMARKS,
            "distance_calibration": {
                "piece_height_mm": self.REAL_CHESS_PIECE_HEIGHT_MM,
                "focal_length_px": self.FOCAL_LENGTH_PX,
                "calibration_width_px": self.CALIBRATION_WIDTH_PX,
                "class_scales": dict(self.distance_scales),
                "camera": self._calibration_signature(),
            },
        }

    def save_map_data(self, json_filename=None):
        """同期側が書きかけのJSONを読まないよう、完成後に置き換える。"""
        path = self._resolve_map_path(json_filename or self.json_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=path.parent, delete=False) as f:
                temporary = f.name
                json.dump(self.get_map_data(), f, indent=2, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        print(f"💾 マップデータを保存しました: {path}")

    def load_map_data(self, json_filename=None):
        try:
            with self._resolve_map_path(json_filename or self.json_filename).open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("JSONの先頭はオブジェクトである必要があります")
            calibration = data.get("distance_calibration", {})
            landmarks = {}
            # 旧形式も読み込めるが、内部管理と次回の保存はlandmarksに統一する。
            raw_landmarks = data.get("landmarks")
            if raw_landmarks is None:
                points = data.get("mapped_chess_pieces", data.get("mapped_cubes"))
                if not isinstance(points, list):
                    raise ValueError("駒の座標データがありません")
                raw_landmarks = dict(enumerate(points))
            if not isinstance(raw_landmarks, dict):
                raise ValueError("landmarksはIDと座標の辞書である必要があります")
            for key, value in raw_landmarks.items():
                landmark_id = int(key)
                if (landmark_id < 0 or landmark_id in landmarks
                        or not isinstance(value, (list, tuple)) or len(value) != 2
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
                    raise ValueError("ランドマークのIDまたは座標が不正です")
                point = tuple(float(v) for v in value)
                if not all(math.isfinite(v) for v in point):
                    raise ValueError("座標に有限でない数値が含まれています")
                landmarks[landmark_id] = point
            if not 1 <= len(landmarks) <= self.EXPECTED_LANDMARKS:
                raise ValueError("マップは1〜6個のランドマークが必要です")
            if len(set(landmarks.values())) != len(landmarks):
                raise ValueError("異なるIDに同じ座標が重複しています")

            # 地図JSONの校正値を使う。別ファイルがなくても地図だけで再利用できる。
            # 古いJSONに校正値がない場合、座標による走行は可能だが画像による補正は無効。
            restored_camera = None
            restored_scales = {}
            if isinstance(calibration, dict):
                camera = calibration.get("camera")
                scales = calibration.get("class_scales")
                if isinstance(camera, dict) and isinstance(scales, dict) and scales:
                    try:
                        values = [float(camera[k]) for k in ("piece_height_mm", "focal_length_px", "calibration_width_px", "head_angle")]
                        if (not all(math.isfinite(v) for v in values) or min(values[:3]) <= 0
                                or not -0.5 <= values[3] <= 0.8):
                            raise ValueError("カメラ校正値が範囲外です")
                        if not all(isinstance(k, str) and isinstance(v, (int, float))
                                   and not isinstance(v, bool) and math.isfinite(v) and 0.1 <= v <= 10 for k,v in scales.items()):
                            raise ValueError("距離補正倍率が範囲外です")
                        restored_camera, restored_scales = values, dict(scales)
                    except (KeyError, ValueError, TypeError) as error:
                        print(f"⚠️ JSONの距離校正は使用せず、地図座標で走行します: {error}")
            self.true_landmarks = landmarks
            self.landmark_id_counter = max(landmarks) + 1
            self.candidate_landmarks.clear()
            self.distance_history.clear()
            self.last_observations = []
            self._map_observation_correction_enabled = restored_camera is not None
            self.distance_scales = restored_scales
            if restored_camera is not None:
                (self.REAL_CHESS_PIECE_HEIGHT_MM, self.FOCAL_LENGTH_PX,
                 self.CALIBRATION_WIDTH_PX, self.TARGET_HEAD_ANGLE) = restored_camera
            else:
                print("ℹ️ 校正値がないため、画像による自己位置補正は無効です。走行座標と保存地図を使用します。")
            print(f"📂 {len(landmarks)}個の駒を読み込みました。保存時と同じ開始位置・向きで使用してください。")
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"⚠️ マップを読み込めません: {error}")
            return False

    def _select_startup_map(self):
        """有効なJSONがあるなら読み込む。既存の不正なJSONで勝手に再探索しない。"""
        if self.force_remap:
            return False
        path = Path(self.json_filename)
        if not path.exists():
            return False
        if not self.load_map_data(path):
            raise RuntimeError(f"地図JSONを確認してください: {path}。再作成する場合はforce_remap=Trueを指定します。")
        if len(self.true_landmarks) != self.EXPECTED_LANDMARKS:
            raise RuntimeError(f"保存地図は{len(self.true_landmarks)}/6個で未完成です。自律走行は開始しません。再作成にはforce_remap=Trueを指定してください。")
        self._map_loaded_from_json = True
        return True

    def _activate_autonomous_navigation(self):
        """地図内の駒座標から巡回ルートを作り、キー操作を待たず自律走行へ移る。"""
        points = np.asarray(list(self.true_landmarks.values()), dtype=float)
        if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise RuntimeError("駒が一直線上にあり、走行領域を作れません。地図座標を確認してください。")
        self.waypoints = self.generate_patrol_waypoints()
        if not self.waypoints:
            raise RuntimeError("この地図から巡回ルートを生成できません。駒の配置を確認してください。")
        self.current_waypoint_index = 1 % len(self.waypoints)
        self.target_x, self.target_y = self.waypoints[0]
        self.autonomous_mode = True
        print("🤖 地図を参照して自律パトロールを開始します。Pで停止・再開、QでSLAM終了。")

    def run(self, cli=None):
        self._camera_adjustment_completed = False
        self._map_loaded_from_json = False
        self._map_created_this_run = False
        self.autonomous_mode = False
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

        try:
            cli.drive_wheels(0.0, 0.0)
            # 明るさ調整より先にJSONを判定する。読み込んだ校正値は首の設定にも反映。
            map_loaded = self._select_startup_map()
            print("🤖 首のキャリブレーション（基準合わせ）を実行中...")
            time.sleep(0.5)
            cli.set_head_angle(-1.0)
            time.sleep(1.5)
            cli.set_head_angle(self.TARGET_HEAD_ANGLE)
            cli.set_lift_height(0.0)
            time.sleep(1.5)

            last_head_cmd_time = time.time()
            if not map_loaded:
                # JSONがないときだけ、明るさ・距離校正→回転マッピングを行う。
                self.true_landmarks.clear()
                self.candidate_landmarks.clear()
                self.distance_history.clear()
                self.robot_trajectory.clear()
                self.last_observations = []
                self.landmark_id_counter = 0
                self.memory_map.fill(127)
                if not self._wait_for_camera_adjustment(cli):
                    return
                self._camera_adjustment_completed = True
                self._map_observation_correction_enabled = True
                if not self._scan_landmarks(cli):
                    return
            else:
                cli.drive_wheels(0.0, 0.0)
                print(f"⏭️ {self.json_filename} を使用し、マッピングをスキップします。")

            self._activate_autonomous_navigation()
            if not map_loaded:
                self._map_created_this_run = True
                # 6個そろった段階で保存。走行終了まで待たず、次回から使用可能にする。
                self.save_map_data()
            
            # マップ完成後は常時OFFにして軽量化（Vキーでいつでも表示切替可能）
            self.show_camera_window = False
            self.camera_viewer.close_window()
            print("💡 マップ準備完了: 動作軽量化のためカメラ映像を非表示にしました（Vキーで表示切替可能）")

            # === 2. ナビゲーションフェーズ ===
            last_inference_time = 0.0
            cached_detections = []
            while True:
                current_time = time.time()
                if current_time - last_head_cmd_time > 1.5:
                    cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                    last_head_cmd_time = current_time

                display_img = self._capture_frame(cli)
                if display_img is not None and cli.pose is not None:
                    raw_theta = self.current_raw_theta
                    est_x, est_y, est_theta = self._get_estimated_pose(
                        self.current_raw_x, self.current_raw_y, raw_theta)
                    correction_triggered = False
                    # 💡 高速化: YOLO推論を最大10Hz(0.1秒ごと)に制限し、CPU負荷を大幅削減
                    if current_time - last_inference_time >= 0.10:
                        last_inference_time = current_time
                        try:
                            cached_detections = self.detect_chess_pieces(display_img)
                            old_count = len(self.true_landmarks)
                            est_x, est_y, est_theta, correction_triggered = self.process_detections(
                                cached_detections, display_img.shape[1], self.current_raw_x,
                                self.current_raw_y, raw_theta,
                                allow_pose_correction=self._map_observation_correction_enabled,
                                allow_new_landmarks=not self._map_loaded_from_json)
                            if self._map_loaded_from_json and not self._map_observation_correction_enabled:
                                self.last_correction = "Saved map / odometry only"
                                for detection in cached_detections:
                                    detection["map_status"] = "Chess (saved map)"
                            if len(self.true_landmarks) != old_count:
                                self.waypoints = self.generate_patrol_waypoints()
                                self.current_waypoint_index = 0
                        except Exception as error:
                            self._report_detection_error(error)

                    if self.show_camera_window:
                        self.draw_chess_detections(display_img, cached_detections)
                    self.robot_trajectory.append((est_x, est_y))
                    self.robot_trajectory = self.robot_trajectory[-2000:]

                    out_of_bounds = False
                    # 未完成の地図の凸包を壁と誤認し、勝手に後退することを防ぐ。
                    if len(self.true_landmarks) == self.EXPECTED_LANDMARKS:
                        pts = np.array(list(self.true_landmarks.values()), dtype=np.float32)
                        hull = cv2.convexHull(pts)
                        # dist > 0: 内部, dist < 0: 外部
                        dist = cv2.pointPolygonTest(hull, (float(est_x), float(est_y)), True)
                        if dist < 0.0: 
                            out_of_bounds = True

                    if out_of_bounds:
                        cv2.putText(display_img, "VIRTUAL FENCE! ESCAPE", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        # マップ中心に向かって旋回・脱出
                        center_angle = math.atan2(self.map_center_y - est_y, self.map_center_x - est_x)
                        turn_diff = self._wrap_angle(center_angle - est_theta)
                        if abs(turn_diff) > 0.4:
                            if turn_diff > 0: cli.drive_wheels(-14.0, 14.0)
                            else: cli.drive_wheels(14.0, -14.0)
                        else:
                            cli.drive_wheels(20.0, 20.0)
                        time.sleep(0.3)
                        self.target_x, self.target_y = None, None 
                    
                    elif any(math.hypot(est_x - tx, est_y - ty) < 55.0 for tx, ty in self.true_landmarks.values()):
                        cv2.putText(display_img, "WALL BLOCK! STOP", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-16.0, -16.0)
                        time.sleep(0.4)
                        cli.drive_wheels(-12.0, 12.0)
                        time.sleep(0.3)
                        self.target_x, self.target_y = None, None 
                    
                    elif self.target_x is not None and self.target_y is not None:
                        dist_to_target = math.hypot(self.target_x - est_x, self.target_y - est_y)
                        angle_to_target = math.atan2(self.target_y - est_y, self.target_x - est_x)
                        angle_diff = angle_to_target - est_theta
                        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                        
                        if dist_to_target < 25.0:
                            print(f"🎯 目標地点({self.target_x:.0f}, {self.target_y:.0f})に到着しました！")
                            self.target_x, self.target_y = None, None
                            cli.drive_wheels(0.0, 0.0)
                            
                            # 🚀 自律モードなら到着時に少し停止してから次へ
                            if self.autonomous_mode:
                                time.sleep(0.5)
                        
                        elif abs(angle_diff) > 0.35:
                            cv2.putText(display_img, "ADJUSTING HEADING", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            if angle_diff > 0: cli.drive_wheels(-14.0, 14.0)
                            else: cli.drive_wheels(14.0, -14.0)
                        else:
                            cv2.putText(display_img, "NAVIGATING TO TARGET", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            # 目標への微小な向きズレを補正しながらスムーズに前進
                            turn_rate = max(-4.0, min(4.0, angle_diff * 8.0))
                            cli.drive_wheels(25.0 - turn_rate, 25.0 + turn_rate)
                            
                    # 🚀 追加: 自律パトロールモードで目標地点が空の時、自動設定する
                    elif self.autonomous_mode and len(self.waypoints) > 0:
                        self.target_x, self.target_y = self.waypoints[self.current_waypoint_index]
                        self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
                        print(f"🤖 [自律パトロール] 次の経由地({self.target_x:.0f}, {self.target_y:.0f})を設定しました")
                        
                    else:
                        cv2.putText(display_img, "WAITING FOR CLICK...", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cli.drive_wheels(0.0, 0.0)

                    # --- UI表示の更新 ---
                    if self.show_camera_window:
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
                        self._set_estimated_pose(self.current_raw_x, self.current_raw_y,
                                                 raw_theta, est_x, est_y, est_theta-0.2)
                        print("🔄 コンパスを左に修正しました")
                    elif key == ord('d'): 
                        self._set_estimated_pose(self.current_raw_x, self.current_raw_y,
                                                 raw_theta, est_x, est_y, est_theta+0.2)
                        print("🔄 コンパスを右に修正しました")
                    elif key in (ord('v'), ord('V')):  # 🚀 追加: カメラ映像の表示/非表示切替
                        self.show_camera_window = not self.show_camera_window
                        if self.show_camera_window:
                            self.camera_viewer.open_window()
                            print("📷 カメラ映像: 表示 (ON)")
                        else:
                            self.camera_viewer.close_window()
                            print("📷 カメラ映像: 非表示 (OFF - 軽量化モード)")
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
    use_cached_map=True,
    json_filename="slam_data.json",
    camera_settings=None,
    camera_viewer=None,
    piece_height_mm=20.0,
    focal_length_px=235.0,
    calibration_width_px=320,
    max_detection_distance_mm=1000.0,
    force_remap=False,
):
    """
    SLAMマッピングを開始から保存まで一括で実行する。

    引数:
        cli: 接続済みのCozmoクライアント(pycozmo.Client)
        model_path: YOLOモデルファイルのパス
        use_cached_map: 旧呼び出し互換用。値によらず既存JSONがあれば自動で再利用する
        json_filename: 地図JSON。相対パスはmain.pyのあるプロジェクト直下が基準
        camera_settings: Cozmoカメラへ適用する露出・ゲイン設定
        camera_viewer: SLAMと自律走行で共有するCameraViewer
        piece_height_mm: 使用する駒の実際の高さ(mm)。異なる高さの駒の混在は未対応
        focal_length_px: calibration_width_px幅の画像で校正した焦点距離(px)
        calibration_width_px: 校正時の元画像幅（YOLOのinference_sizeとは別）
        max_detection_distance_mm: 検出対象とする最大の推定距離(mm)
        force_remap: Trueのときだけ、JSONがあっても明示的に再マッピングする
    """
    print("\n=== SLAMマッピングを開始します ===")
    slam_bot = CozmoSlamController(
        model_path=model_path,
        use_cached_map=use_cached_map,
        camera_settings=camera_settings,
        camera_viewer=camera_viewer,
        piece_height_mm=piece_height_mm,
        focal_length_px=focal_length_px,
        calibration_width_px=calibration_width_px,
        json_filename=json_filename,
        max_detection_distance_mm=max_detection_distance_mm,
        force_remap=force_remap,
    )
    try:
        slam_bot.run(cli=cli)
    except KeyboardInterrupt:
        print("\n\n🛑 マッピングを終了し、自律走行へ移行します...")
    finally:
        # 既存JSONを参照する走行や、未完成の探索では元のファイルを上書きしない。
        if slam_bot._map_created_this_run:
            slam_bot.save_map_data()
