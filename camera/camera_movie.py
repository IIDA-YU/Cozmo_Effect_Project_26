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
EXPOSURE_STEP_MS = 1
GAIN_STEP = 0.25

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
    """
    Cozmo映像の取得・表示・明るさ操作を一元管理する。

    SLAMと併用する場合は ``start(background_display=False)`` とし、
    SLAM側から ``display_once`` と ``poll_key`` を呼ぶ。これにより、複数の
    スレッドが同時にOpenCVのGUIを操作してウィンドウが消える問題を防ぐ。
    """

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
        self._initial_camera_settings = normalize_camera_settings(camera_settings)
        self.camera_settings = normalize_camera_settings(
            self._initial_camera_settings
        )
        self.window_name = "Cozmo Camera View"
        self._is_active = False
        self._latest_image = None
        self._camera_handler = None
        self._display_thread = None
        self._window_created = False
        self._lock = threading.Lock()
        self._settings_lock = threading.RLock()
        self._stop_event = threading.Event()
        self.detection_data = {}
        self._data_lock = threading.Lock()

    def set_detections(self, key, detections):
        with self._data_lock:
            self.detection_data[key] = detections

    def clear_detections(self, key=None):
        with self._data_lock:
            if key is None:
                self.detection_data.clear()
            else:
                self.detection_data.pop(key, None)

    def on_camera_image(self, cli, image):
        if not self._is_active:
            return

        img_bgr = cozmo_image_to_bgr(image)
        if self.enhance_image:
            img_bgr = enhance_chess_frame(img_bgr)

        # 検出枠や文字は表示時に重ねる。映像取得スレッドではGUIを触らない。
        with self._lock:
            self._latest_image = img_bgr

    def get_latest_frame(self):
        """検出処理にも利用できる最新フレームのコピーを返す。"""
        with self._lock:
            if self._latest_image is None:
                return None
            return self._latest_image.copy()

    def _draw_detections(self, frame):
        image_height, image_width = frame.shape[:2]
        with self._data_lock:
            detection_lists = [list(items) for items in self.detection_data.values()]

        for det_list in detection_lists:
            for det in det_list:
                x1, y1, x2, y2 = (int(value) for value in det["box"])
                x1 = max(0, min(image_width - 1, x1))
                y1 = max(0, min(image_height - 1, y1))
                x2 = max(0, min(image_width - 1, x2))
                y2 = max(0, min(image_height - 1, y2))
                label = det.get("label", "")

                if "WARNING" in label:
                    color = (0, 0, 255)
                elif "FAR" in label:
                    color = (0, 255, 0)
                elif "Cozmo" in label:
                    color = (0, 255, 255)
                elif "ID:" in label:
                    color = (255, 0, 255)
                else:
                    color = (255, 255, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

    def _draw_camera_status(self, frame):
        with self._settings_lock:
            settings = normalize_camera_settings(self.camera_settings)

        mode = "AUTO" if settings.auto_exposure else "MANUAL"
        first_line = (
            f"Brightness: {mode}  Exp {settings.exposure_ms}ms  "
            f"Gain {settings.gain:.2f}"
        )
        second_line = "[B] Brighter  [N] Darker  [M] Auto/Manual  [0] Reset"

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 48), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
        cv2.putText(
            frame,
            first_line,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            second_line,
            (8, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _compose_display_frame(self, frame_override=None):
        frame = (
            self.get_latest_frame()
            if frame_override is None
            else frame_override.copy()
        )
        if frame is None:
            return None

        self._draw_detections(frame)
        if self.battery_monitor:
            frame = self.battery_monitor.draw_overlay(frame)
        self._draw_camera_status(frame)
        return frame

    def ensure_camera_settings(self, announce=False):
        """カラー配信と現在の露出設定をまとめて再適用する。"""
        with self._settings_lock:
            self.camera_settings = configure_cozmo_camera(
                self.cli,
                settings=self.camera_settings,
                color=True,
                announce=announce,
            )

    def ensure_color_camera(self, announce=False):
        """旧呼び出し名との互換性を保つ。"""
        self.ensure_camera_settings(announce=announce)

    def _apply_runtime_settings(self):
        with self._settings_lock:
            self.camera_settings = apply_camera_settings(
                self.cli,
                self.camera_settings,
                announce=True,
            )

    def adjust_brightness(self, brighter):
        """実機の露出とゲインを段階的に変更する。"""
        with self._settings_lock:
            settings = normalize_camera_settings(self.camera_settings)
            settings.auto_exposure = False

            if brighter:
                if settings.exposure_ms < MAX_EXPOSURE_MS:
                    settings.exposure_ms += EXPOSURE_STEP_MS
                else:
                    settings.gain += GAIN_STEP
            else:
                if settings.gain > MIN_GAIN:
                    settings.gain -= GAIN_STEP
                else:
                    settings.exposure_ms -= EXPOSURE_STEP_MS

            self.camera_settings = normalize_camera_settings(settings)
        self._apply_runtime_settings()

    def toggle_auto_exposure(self):
        with self._settings_lock:
            settings = normalize_camera_settings(self.camera_settings)
            settings.auto_exposure = not settings.auto_exposure
            self.camera_settings = settings
        self._apply_runtime_settings()

    def reset_brightness(self):
        with self._settings_lock:
            self.camera_settings = normalize_camera_settings(
                self._initial_camera_settings
            )
        self._apply_runtime_settings()

    def handle_key(self, key):
        """カメラ操作キーを処理し、処理済みならTrueを返す。"""
        if key in (ord("b"), ord("B"), ord("="), ord("+")):
            self.adjust_brightness(brighter=True)
            return True
        if key in (ord("n"), ord("N"), ord("-"), ord("_")):
            self.adjust_brightness(brighter=False)
            return True
        if key in (ord("m"), ord("M")):
            self.toggle_auto_exposure()
            return True
        if key == ord("0"):
            self.reset_brightness()
            return True
        return False

    def open_window(self):
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            self._window_created = True

    def display_once(self, frame_override=None):
        """最新映像を1回描画する。cv2.waitKeyは呼び出し側に集約する。"""
        frame = self._compose_display_frame(frame_override=frame_override)
        if frame is None:
            return False
        self.open_window()
        cv2.imshow(self.window_name, frame)
        return True

    def poll_key(self, delay_ms=1):
        """OpenCVのキーを1回取得し、カメラ操作以外のキーを返す。"""
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == 0xFF:
            return -1
        if self.handle_key(key):
            return -1
        return key

    def start(self, background_display=True):
        if not self._is_active:
            self._is_active = True
            self._stop_event.clear()
            self._camera_handler = self.cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage,
                self.on_camera_image,
            )
            self.ensure_camera_settings(announce=True)

        if background_display:
            self.start_background_display()

    def start_background_display(self):
        """単独利用時だけ、GUI更新用スレッドを起動する。"""
        if self._display_thread is not None and self._display_thread.is_alive():
            return
        self._display_thread = threading.Thread(
            target=self._display_loop,
            name="CozmoCameraDisplay",
            daemon=True,
        )
        self._display_thread.start()

    def _display_loop(self):
        while self._is_active and not self._stop_event.is_set():
            self.display_once()
            key = self.poll_key(delay_ms=10)
            if key in (ord("q"), 27):
                self.stop()
                break

    def close_window(self):
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        self._window_created = False

    def stop(self, disable_camera=True, close_window=True):
        self._is_active = False
        self._stop_event.set()

        if self._camera_handler is not None and hasattr(self.cli, "del_handler"):
            self.cli.del_handler(
                pycozmo.event.EvtNewRawCameraImage,
                self._camera_handler,
            )
            self._camera_handler = None

        if disable_camera:
            self.cli.enable_camera(enable=False)
        if close_window:
            self.close_window()

        if (
            self._display_thread is not None
            and self._display_thread.is_alive()
            and threading.current_thread() is not self._display_thread
        ):
            self._display_thread.join(timeout=1.0)
        self._display_thread = None
