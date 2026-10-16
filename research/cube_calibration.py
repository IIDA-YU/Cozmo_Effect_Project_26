"""停止した実物キューブによる距離倍率の校正。元の設定値へ二重適用しない。"""
import json
import hashlib
import math
import statistics
import time
from research.world import save_json


def signature(config):
    camera = {k: getattr(config, k) for k in
              ('cube_width_mm', 'cube_focal_px', 'head_angle', 'camera_forward_mm',
               'cube_detection_region')}
    # 同じbest_powercube.pt名でも、再学習後の古い倍率を引き継がない。
    camera['model_sha256'] = hashlib.sha256(config.cube_model_path.read_bytes()).hexdigest()
    return camera


def load_calibration(config):
    path = config.setup_path.parent / 'cube_camera_calibration.json'
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding='utf-8'))
    scale = data.get('distance_scale')
    if data.get('camera') != signature(config):
        config.cube_distance_scale = 1.0
        print('旧キューブ校正は検出領域・モデル・カメラ条件が異なるため使用しません。Kキーで再校正してください。')
        return False
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.1 <= scale <= 10:
        raise ValueError('キューブ校正倍率が不正です。Kキーで再校正してください')
    config.cube_distance_scale = scale
    return True


def calibrated_scale(distances, measured_depth, current_scale):
    if len(distances) < 5 or not 80 <= measured_depth <= 700:
        raise ValueError('5画像以上と、カメラから80〜700mmの実測距離が必要です')
    if not all(math.isfinite(d) and d > 0 for d in distances):
        raise ValueError('距離推定値が不正です')
    median = statistics.median(distances)
    if (max(distances)-min(distances)) / median > 0.10:
        raise ValueError('検出枠が不安定です。キューブと機体を固定して再試行してください')
    scale = current_scale * measured_depth / median
    if not 0.1 <= scale <= 10:
        raise ValueError('校正倍率が範囲外です。距離と単位mmを確認してください')
    return scale


def calibrate_stopped(cli, camera, perception, config):
    cli.drive_wheels(0, 0)
    print('キューブをカメラ正面中央へ置いてください。')
    print(f'検出領域: {config.cube_detection_region}。モデル変更後は本体全体の枠で再校正してください。')
    print('150〜250mm程度が推奨です。Cozmoの基準中心からキューブ中心までを実測します。')
    try:
        measured = float(input('実測距離 [mm]（空欄で中止） > ')) - config.camera_forward_mm
    except ValueError:
        return False
    distances, last = [], -1
    stopped = time.monotonic()
    deadline = stopped + 15
    while len(distances) < 7 and time.monotonic() < deadline:
        camera.display_once()
        if camera.poll_key(20) in (27, ord('q')):
            return False
        frame, number, stamp = camera.get_frame_sample()
        if frame is None or number == last or stamp < stopped + config.observation_settle_seconds:
            continue
        last = number
        if time.monotonic() - stamp > config.frame_timeout_seconds:
            continue
        _, _, boxes, _ = perception.detect(frame, (0, 0, 0))
        camera.set_detections('research', boxes)
        valid = [o for o in perception.cube_observations if o['status'] in ('READY', 'REVIEW RANGE')
                 and o.get('distance') is not None
                 and o['confidence'] >= config.cube_review_confirm_confidence
                 and abs((o['box'][0]+o['box'][2])/2-frame.shape[1]/2) < frame.shape[1]*0.04]
        if len(valid) == 1:
            # 遠距離候補のpointは安全のため丸められるので、校正には生の単眼距離を使う。
            distances.append(valid[0]['distance'])
    try:
        scale = calibrated_scale(distances, measured, config.cube_distance_scale)
    except ValueError as error:
        print(error)
        return False
    save_json(config.setup_path.parent / 'cube_camera_calibration.json',
              dict(camera=signature(config), distance_scale=scale,
                   measured_camera_depth_mm=measured, sample_count=len(distances)))
    config.cube_distance_scale = scale
    print(f'キューブ距離倍率を {scale:.4f} に校正しました。別の距離でも確認してください。')
    return True
