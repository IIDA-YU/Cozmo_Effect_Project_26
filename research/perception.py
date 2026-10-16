"""停止中の画像からキューブ・Cozmo・ArUcoを検出する。

全体検出では重なった重複枠だけ除去し、隣り合うキューブを分けて扱う。
旧マーカーモデルの面融合は、明示的にmarkerを選んだ場合だけ使用する。
"""
import math
import hashlib

from research.cube_mapping import project_cube


def _overlap_1d(a1, a2, b1, b2):
    return max(0.0, min(a2, b2) - max(a1, b1))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = _overlap_1d(ax1, ax2, bx1, bx2)
    ih = _overlap_1d(ay1, ay2, by1, by2)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + ba - inter)


def _same_cube_faces(a, b, config):
    """2つのYOLO枠が同じキューブの別面らしいかを保守的に判定する。"""
    box_a, box_b = a['box'], b['box']
    if _iou(box_a, box_b) >= 0.30:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    aw, ah = ax2 - ax1, ay2 - ay1
    bw, bh = bx2 - bx1, by2 - by1
    if min(aw, ah, bw, bh) <= 0:
        return False

    vertical_overlap = _overlap_1d(ay1, ay2, by1, by2) / min(ah, bh)
    if vertical_overlap < config.cube_face_vertical_overlap:
        return False

    # 横方向で重なるか、ほぼ接している2面だけを融合する。
    horizontal_gap = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    if horizontal_gap > config.cube_face_gap_ratio * min(ah, bh):
        return False

    size_ratio = max(ah, bh) / min(ah, bh)
    if size_ratio > config.cube_face_size_ratio:
        return False

    center_y_a = (ay1 + ay2) / 2
    center_y_b = (by1 + by2) / 2
    if abs(center_y_a - center_y_b) > 0.45 * max(ah, bh):
        return False

    # 1個の立方体を角から見た2面なら、2枠を合わせた横幅は高さの
    # 約2倍以内に収まる。横に並んだ別キューブまで融合しないための上限。
    combined_width = max(ax2, bx2) - min(ax1, bx1)
    if combined_width > config.cube_face_combined_width_ratio * max(ah, bh):
        return False

    return True


def fuse_cube_face_detections(detections, config):
    """同一キューブの複数面を1検出へまとめる。

    bbox自体を結合すると見かけサイズが大きくなり距離が壊れるため、
    各グループから「高さが大きく、信頼度も高い」代表面を1つだけ残す。
    """
    if len(detections) <= 1:
        return [dict(d, face_group_size=1) for d in detections]

    parent = list(range(len(detections)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            if _same_cube_faces(detections[i], detections[j], config):
                union(i, j)

    groups = {}
    for i, detection in enumerate(detections):
        groups.setdefault(find(i), []).append(detection)

    fused = []
    for group in groups.values():
        # 斜め面は幅が縮みやすい。高さを第一優先にして代表面を選ぶ。
        representative = max(
            group,
            key=lambda d: (
                d['box'][3] - d['box'][1],
                d['confidence'],
                (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]),
            ),
        )
        item = dict(representative)
        item['face_group_size'] = len(group)
        item['face_group_boxes'] = [g['box'] for g in group]
        fused.append(item)
    return fused


def filter_cube_detections(detections, config):
    """全体枠は信頼度順NMS。隣接枠の連鎖的な融合は行わない。"""
    if config.cube_detection_region == 'marker':
        return fuse_cube_face_detections(detections, config)
    kept = []
    for detection in sorted(detections, key=lambda d: -d['confidence']):
        if not any(_iou(detection['box'], d['box']) >= config.cube_duplicate_iou for d in kept):
            kept.append(dict(detection, face_group_size=1))
    return kept


class ResearchPerception:
    def __init__(self, config):
        import cv2
        from ultralytics import YOLO

        self.cv2, self.config = cv2, config
        self.model_sha256 = hashlib.sha256(config.cube_model_path.read_bytes()).hexdigest()
        self.model = YOLO(str(config.cube_model_path))
        self.robot_model = YOLO(str(config.robot_model_path))
        self.robot_in_view = False

        names = self.model.names
        print('キューブモデルのクラス:', names)
        pairs = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_ids = [
            i for i, name in pairs
            if str(name).strip().lower() in {n.lower() for n in config.cube_class_names}
        ]
        if not self.class_ids:
            raise ValueError(
                'cube_class_namesにモデルの荷物クラス名を設定してください。'
                'station等は自動で荷物扱いしません'
            )

        aruco = cv2.aruco
        self.detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
            aruco.DetectorParameters(),
        )
        self.marker_counts = {}
        self.preview_cubes = []
        self.cube_observations = []

    def detect(self, frame, pose):
        """1フレームを処理する。"""
        cfg = self.config

        result = self.model.predict(
            frame,
            classes=self.class_ids,
            conf=cfg.cube_candidate_confidence,
            imgsz=416,
            verbose=False,
        )[0]

        raw_detections = []
        for box in result.boxes:
            raw_detections.append({
                'box': tuple(map(float, box.xyxy[0].tolist())),
                'confidence': float(box.conf[0]),
            })
        detections = filter_cube_detections(raw_detections, cfg)

        cube_candidates = []
        boxes = []
        occluders = []
        self.preview_cubes = []
        self.cube_observations = []
        self.close_cube_unlocalized = False

        for detection in detections:
            coords = detection['box']
            observation = project_cube(coords, frame.shape, pose, cfg)
            observation['box'] = tuple(map(int, coords))
            observation['confidence'] = detection['confidence']
            observation['face_group_size'] = detection.get('face_group_size', 1)
            observation['box_index'] = len(boxes)

            # 画面を大きく占める場合は距離式より衝突回避を優先する。
            visually_close = observation.get('box_height_ratio', 0) >= cfg.cube_emergency_box_height_ratio
            if (observation['status'] in ('CLIPPED BOX', 'TOO CLOSE')
                    or (visually_close and not observation['range_reliable'])):
                self.close_cube_unlocalized = True

            self.cube_observations.append(observation)
            face_text = '' if observation['face_group_size'] == 1 else f' faces×{observation["face_group_size"]}'
            boxes.append({'box': observation['box'], 'label': 'Cube: checking map' + face_text})

            point = observation['point']
            if point is not None:
                occluders.append((*point, cfg.cube_width_mm / 2))
                self.preview_cubes.append(point)
                if observation['status'] in ('READY', 'REVIEW RANGE'):
                    cube_candidates.append(point)

        # 顔が反対を向いている相手にも、既存の機体検出モデルで前進を抑止する。
        robot_result = self.robot_model.predict(
            frame,
            conf=0.65,
            imgsz=416,
            verbose=False,
        )[0]
        self.robot_in_view = False
        for box in robot_result.boxes:
            cls = int(box.cls[0])
            if str(self.robot_model.names[cls]).lower() != 'cozmo':
                continue
            self.robot_in_view = True
            coords = tuple(map(int, box.xyxy[0].tolist()))
            boxes.append({'box': coords, 'label': 'Cozmo: yield'})

        corners, ids, _ = self.detector.detectMarkers(frame)
        seen, confirmed = set(), []
        if ids is not None:
            for corner, mid in zip(corners, ids.flatten()):
                c = corner[0]
                marker_id = int(mid)
                edge = min(math.dist(c[i], c[(i + 1) % 4]) for i in range(4))
                boxes.append({
                    'box': (
                        int(c[:, 0].min()),
                        int(c[:, 1].min()),
                        int(c[:, 0].max()),
                        int(c[:, 1].max()),
                    ),
                    'label': f'ArUco {marker_id}',
                })
                if edge >= cfg.min_marker_pixels:
                    seen.add(marker_id)
                    self.marker_counts[marker_id] = self.marker_counts.get(marker_id, 0) + 1
                    if self.marker_counts[marker_id] >= cfg.marker_confirm_frames:
                        confirmed.append(marker_id)

        self.marker_counts = {
            key: value
            for key, value in self.marker_counts.items()
            if key in seen
        }
        return cube_candidates, confirmed, boxes, occluders
