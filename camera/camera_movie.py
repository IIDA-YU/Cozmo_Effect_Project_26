# ----------------------------------------------------------------------
# camera_movie.py
# Cozmoのカラー映像を表示し、検出結果を重ねるモジュール
#
# YOLOモデルの読み込みはvisual_slam.pyだけが担当する。
# CameraViewerは、全検出モジュールで共有する映像表示と補正に専念する。
# ----------------------------------------------------------------------

import threading

import cv2
import numpy as np
import pycozmo


_CHESS_CLAHE = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))


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
    def __init__(self, cli, battery_monitor=None, enhance_image=True):
        self.cli = cli
        self.battery_monitor = battery_monitor
        self.enhance_image = enhance_image
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

    def ensure_color_camera(self):
        """他モジュールの起動後でも、カメラをカラー配信へ戻す。"""
        if hasattr(self.cli, "set_head_light"):
            # 白い机での反射・白飛びを増やさないよう、ヘッドライトは消灯する。
            self.cli.set_head_light(enable=False)
        self.cli.enable_camera(enable=True, color=True)

    def start(self):
        if self._is_active:
            return

        self._is_active = True
        self._camera_handler = self.cli.add_handler(
            pycozmo.event.EvtNewRawCameraImage,
            self.on_camera_image,
        )
        self.ensure_color_camera()
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