"""キューブ検出を共通地図へ投影する。単位は mm / rad。

全体検出モデルでは50mm角の本体高さを基準にする。
斜め方向では上面も枠に含まれるため、単眼bbox測距は近似でありKキーで校正する。

    * 近距離: 校正済みbbox距離を位置推定に利用
    * 遠距離: 方位だけを主に利用し、候補位置は安全な距離へ丸める
    * 十分遠い: 地図候補にせず、探索で近づいてから再検出

最終確定は cube_review.py が停止画像の連続性と座標の整合を確認して行う。
"""
import math


def _camera_geometry(box, frame_shape, pose, config):
    """bboxからカメラ原点・方位・単眼距離を求める。"""
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    w, h = x2 - x1, y2 - y1

    focal = config.cube_focal_px * width / 320.0

    # 横幅は斜めから面を見たとき強く縮むため、距離には高さを使う。
    # 全体検出では本体50mmを使う。旧マーカー用の校正倍率は流用しない。
    forward = config.cube_width_mm * focal / h * config.cube_distance_scale
    left = -(0.5 * (x1 + x2) - width / 2) * forward / focal
    radial = math.hypot(forward, left)

    x, y, theta = pose
    camera_origin = (
        x + config.camera_forward_mm * math.cos(theta),
        y + config.camera_forward_mm * math.sin(theta),
    )
    bearing_world = theta + math.atan2(left, forward)
    return w, h, focal, radial, camera_origin, bearing_world


def project_cube(box, frame_shape, pose, config):
    """検出枠からキューブ候補を投影する。

    ``point`` は常に「確定位置」とは限らない。遠距離ではCozmoが近づくための
    案内点として ``cube_candidate_projection_mm`` までに丸める。
    ``range_reliable`` がTrueの観測だけが永続登録の候補になる。
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    observation = dict(
        point=None,
        distance=None,
        status='INVALID BOX',
        size_source='',
        range_reliable=False,
        point_is_bounded=False,
    )
    if not all(math.isfinite(value) for value in (*box, *pose)):
        return observation

    w, h = x2 - x1, y2 - y1
    observation['box_height_ratio'] = h / height
    observation['box_width_ratio'] = w / width
    if min(w, h) < 6:
        observation['status'] = 'SMALL BOX'
        return observation

    if not config.cube_aspect_min <= w / h <= config.cube_aspect_max:
        observation['status'] = 'BAD ASPECT'
        return observation

    # 一部が切れた枠はbboxサイズを距離に使えない。
    if x1 <= 2 or y1 <= 2 or x2 >= width - 2 or y2 >= height - 2:
        observation['status'] = 'CLIPPED BOX'
        return observation

    w, h, focal, distance, camera_origin, bearing_world = _camera_geometry(
        box, frame_shape, pose, config
    )
    observation.update(
        distance=distance,
        estimated_range_mm=distance,
        camera_origin=camera_origin,
        bearing_world=bearing_world,
        box_height_ratio=h / height,
        box_width_ratio=w / width,
        size_source='whole-cube-height' if config.cube_detection_region == 'whole_cube' else 'marker-height',
    )

    if distance < config.cube_map_near_mm:
        observation['status'] = 'TOO CLOSE'
        return observation
    if distance > config.cube_map_far_mm:
        observation['status'] = 'TOO FAR'
        return observation

    # 遠距離のbbox測距値をそのまま地図上の1点として信用しない。
    # 確定範囲内の実測値まで案内点の上限(300mm)へ丸めない。
    navigation_distance = (distance if distance <= config.cube_confirm_far_mm
                           else min(distance, config.cube_candidate_projection_mm))
    observation['point_is_bounded'] = navigation_distance < distance
    observation['navigation_distance_mm'] = navigation_distance
    observation['point'] = (
        camera_origin[0] + navigation_distance * math.cos(bearing_world),
        camera_origin[1] + navigation_distance * math.sin(bearing_world),
    )

    if distance <= config.cube_confirm_far_mm:
        observation['status'] = 'READY'
        observation['range_reliable'] = True
    else:
        observation['status'] = 'REVIEW RANGE'

    # 単眼距離は遠いほど不確か。丸めた遠距離候補はさらに大きい誤差を持つ。
    sigma = max(10.0, distance * (0.10 if observation['range_reliable'] else 0.22))
    observation['sigma_mm'] = sigma
    return observation


def classify_cube(observation, grid):
    """カメラ表示・地図登録で共用する採否。"""
    if observation['status'] != 'READY':
        return observation['status']
    if not observation.get('range_reliable', True):
        return 'REVIEW RANGE'
    if not grid.inside(observation['point']):
        return 'OUTSIDE MAP'
    return 'MAPPED'


def cube_corners(point, size_mm):
    """推定中心を基準に、実寸の正方形を作る（向きは推定しない）。"""
    x, y = point
    half = size_mm / 2
    return [
        (x - half, y - half),
        (x + half, y - half),
        (x + half, y + half),
        (x - half, y + half),
    ]
