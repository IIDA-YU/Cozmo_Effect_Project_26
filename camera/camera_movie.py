import pycozmo
import cv2
import numpy as np
import threading


class CameraViewer:
    """
    Cozmoのカメラ映像を取得し、別ウィンドウで表示するクラス。
    battery_monitor を渡すと、映像の右下にバッテリー残量を表示します。
    """
    def __init__(self, cli, battery_monitor=None):
        self.cli = cli
        self.battery_monitor = battery_monitor
        self.window_name = "Cozmo Camera View"
        self._is_active = False

        # 映像共有用の変数と排他制御(ロック)
        self._latest_image = None
        self._lock = threading.Lock()
        self._display_thread = None

    def on_camera_image(self, cli, image):
        """
        カメラから新しい画像フレームを受信したときに呼び出されるコールバック関数
        """
        if not self._is_active:
            return

        # Cozmoから送られてくるPIL形式の画像を、OpenCVで扱えるNumPy配列に変換
        img_np = np.array(image)

        # RGBからBGRに変換 (OpenCVはBGR形式をデフォルトとしているため)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np  # 万が一グレースケール等の場合

        # ===== バッテリー残量を右下にオーバーレイ =====
        if self.battery_monitor is not None:
            img_bgr = self.battery_monitor.draw_overlay(img_bgr)

        # 描画用スレッドに渡すため、最新画像を安全に保存
        with self._lock:
            self._latest_image = img_bgr

    def _display_loop(self):
        """
        ウィンドウ表示とイベント処理を独立して行うスレッド関数
        """
        while self._is_active:
            img = None
            with self._lock:
                if self._latest_image is not None:
                    img = self._latest_image.copy()

            if img is not None:
                cv2.imshow(self.window_name, img)

            # 約30fps相当の間隔でキー入力を待ちつつ、OpenCVのGUIイベントを処理する
            cv2.waitKey(30)

    def start(self):
        """
        カメラを有効化し、映像のストリーミング表示を開始します
        """
        self._is_active = True
        # カメラを有効化（カラー映像）
        self.cli.enable_camera(enable=True, color=True)

        # 画像受信時のイベントハンドラ（コールバック）を登録
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

        # 表示専用のスレッドを起動 (daemon=Trueによりメインプログラム終了時に道連れで終了する)
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()

        print("カメラ映像の表示を開始しました。")

    def stop(self):
        """
        カメラ映像の表示を停止し、ウィンドウを閉じます
        """
        self._is_active = False
        self.cli.enable_camera(enable=False)

        # 表示スレッドの終了を少しだけ待つ
        if self._display_thread is not None:
            self._display_thread.join(timeout=1.0)

        # OpenCVのウィンドウを明示的に破棄
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

        # 破棄後、OSのイベントキューを処理させて完全にウィンドウを消すために waitKey を数回呼ぶ
        for _ in range(5):
            cv2.waitKey(1)

        print("カメラ映像の表示を終了しました。")