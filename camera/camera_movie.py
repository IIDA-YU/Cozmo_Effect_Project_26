# ----------------------------------------------------------------------
# camera_movie.py
# Cozmoのカラー映像を表示し、検出結果を重ねるモジュール
#
# YOLOモデルの読み込みはvisual_slam.pyだけが担当する。
# CameraViewerは、全検出モジュールで共有する映像表示と補正に専念する。
# ----------------------------------------------------------------------

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pycozmo


MIN_EXPOSURE_MS = 1
MAX_EXPOSURE_MS = 30
MIN_GAIN = 1.0
MAX_GAIN = 8.0
DEFAULT_EXPOSURE_MS = 5
DEFAULT_GAIN = 1.0
DEFAULT_AUTO_EXPOSURE = False

_CHESS_CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))


@dataclass
class CameraSettings:
    """Cozmo本体へ送る露出設定。"""

    exposure_ms: int = DEFAULT_EXPOSURE_MS
    gain: float = DEFAULT_GAIN
    auto_exposure: bool = DEFAULT_AUTO_EXPOSURE


def normalize_camera_settings(settings=None):
    """設定値をCozmoカメラの安全な範囲へ収め、独立した値として返す。"""
    source = settings if settings is not None else CameraSettings()
    return CameraSettings(
        exposure_ms=int(
            max(
                MIN_EXPOSURE_MS,
                min(MAX_EXPOSURE_MS, source.exposure_ms),
            )
        ),
        gain=float(max(MIN_GAIN, min(MAX_GAIN, source.gain))),
        auto_exposure=bool(source.auto_exposure),
    )


def apply_camera_settings(client, settings, announce=True):
    """露出・ゲイン・自動露出の設定をCozmo本体へ送信する。"""
    normalized = normalize_camera_settings(settings)
    packet = pycozmo.protocol_encoder.SetCameraParams(
        gain=normalized.gain,
        exposure_ms=normalized.exposure_ms,
        auto_exposure_enabled=normalized.auto_exposure,
    )
    client.conn.send(packet)

    if announce:
        mode = "AUTO" if normalized.auto_exposure else "MANUAL"
        print(
            f"📷 カメラ設定: {mode} / "
            f"露出 {normalized.exposure_ms}ms / "
            f"ゲイン {normalized.gain:.2f}"
        )

    return normalized


def configure_cozmo_camera(
    client,
    settings=None,
    color=True,
    announce=True,
    settle_seconds=0.15,
):
    """
    カメラ配信・ライト・露出を一括設定する。

    配信開始直後の初期化で設定が上書きされる個体にも対応できるよう、
    短い待機後に同じ露出設定をもう一度送る。
    """
    normalized = normalize_camera_settings(settings)
    client.enable_camera(enable=True, color=color)

    if hasattr(client, "set_head_light"):
        # 動作確認済みのcozmo_camera_capture.pyと同じ呼び出し方に統一する。
        client.set_head_light(False)

    apply_camera_settings(client, normalized, announce=announce)

    if settle_seconds > 0:
        time.sleep(settle_seconds)
        apply_camera_settings(client, normalized, announce=False)

    return normalized


def cozmo_image_to_bgr(image):
    """PIL/NumPy形式のCozmo画像を、常にOpenCVのBGR画像へ変換する。"""
    image_np = np.asarray(image)

    if image_np.ndim == 2:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)

    if image_np.ndim != 3:
        raise ValueError(f"未対応のカメラ画像形式です: shape={image_np.shape}")

    channels = image_np.shape[2]
    if channels == 1:
        return cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"未対応のカメラ画像形式です: shape={image_np.shape}")


def enhance_chess_frame(frame_bgr):
    """
    明暗差と輪郭を穏やかに補正する。

    色相を大きく変えず、YOLOの入力サイズや検出座標も変えないため、
    元映像に対する検出枠をそのまま重ねられる。
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    luminance, color_a, color_b = cv2.split(lab)
    luminance = _CHESS_CLAHE.apply(luminance)
    contrast = cv2.cvtColor(
        cv2.merge((luminance, color_a, color_b)),
        cv2.COLOR_LAB2BGR,
    )

    # 小さなチェス駒の輪郭だけを軽く強調する。
    blurred = cv2.GaussianBlur(contrast, (0, 0), 0.8)
    return cv2.addWeighted(contrast, 1.10, blurred, -0.10, 0)


class CameraViewer:
    def __init__(
        self,
        cli,
        battery_monitor=None,
        enhance_image=True,
        camera_settings=None,
    ):
        self.cli = cli
        self.battery_monitor = battery_monitor
        self.enhance_image = enhance_image
        self.camera_settings = normalize_camera_settings(camera_settings)
        self.window_name = "Cozmo Camera View"
        self._is_active = False
        self._latest_image = None
        self._camera_handler = None
        self._lock = threading.Lock()
        self.detection_data = {}
        self._data_lock = threading.Lock()

    def set_detections(self, key, detections):
        with self._data_lock:
            self.detection_data[key] = detections

    def on_camera_image(self, cli, image):
        if not self._is_active:
            return

        img_bgr = cozmo_image_to_bgr(image)
        if self.enhance_image:
            img_bgr = enhance_chess_frame(img_bgr)

        image_height, image_width = img_bgr.shape[:2]
        with self._data_lock:
            for det_list in self.detection_data.values():
                for det in det_list:
                    x1, y1, x2, y2 = (int(value) for value in det["box"])
                    x1 = max(0, min(image_width - 1, x1))
                    y1 = max(0, min(image_height - 1, y1))
                    x2 = max(0, min(image_width - 1, x2))
                    y2 = max(0, min(image_height - 1, y2))
                    label = det.get("label", "")

                    if "WARNING" in label:
                        color = (0, 0, 255)    # 危険＝赤色
                    elif "FAR" in label:
                        color = (0, 255, 0)    # 遠い＝緑色
                    elif "Cozmo" in label:
                        color = (0, 255, 255)  # 仲間＝黄色
                    elif "ID:" in label:
                        color = (255, 0, 255)  # ArUcoマーカー＝紫色
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img_bgr,
                        label,
                        (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        if self.battery_monitor:
            img_bgr = self.battery_monitor.draw_overlay(img_bgr)

        with self._lock:
            self._latest_image = img_bgr

    def ensure_camera_settings(self, announce=False):
        """カラー配信と手動露出をまとめて再適用する。"""
        self.camera_settings = configure_cozmo_camera(
            self.cli,
            settings=self.camera_settings,
            color=True,
            announce=announce,
        )

    def ensure_color_camera(self, announce=False):
        """旧呼び出し名との互換性を保つ。"""
        self.ensure_camera_settings(announce=announce)

    def start(self):
        if self._is_active:
            return

        self._is_active = True
        self._camera_handler = self.cli.add_handler(
            pycozmo.event.EvtNewRawCameraImage,
            self.on_camera_image,
        )
        self.ensure_camera_settings(announce=True)
        threading.Thread(target=self._display_loop, daemon=True).start()

    def _display_loop(self):
        while self._is_active:
            with self._lock:
                frame = None if self._latest_image is None else self._latest_image.copy()

            if frame is not None:
                cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)

    def stop(self):
        self._is_active = False

        if self._camera_handler is not None and hasattr(self.cli, "del_handler"):
            self.cli.del_handler(
                pycozmo.event.EvtNewRawCameraImage,
                self._camera_handler,
            )
            self._camera_handler = None

        self.cli.enable_camera(enable=False)
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            # ウィンドウ生成前に停止された場合は、閉じる対象がない。
            pass
