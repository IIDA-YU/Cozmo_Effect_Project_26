#----------------------------------------------------------------------
# camera_movie.py
# Cozmoのカメラ映像を表示するためのモジュールクラス
#----------------------------------------------------------------------

import pycozmo, cv2, numpy as np, threading

class CameraViewer:
    def __init__(self, cli, battery_monitor=None):
        self.cli = cli
        self.battery_monitor = battery_monitor
        self.window_name = "Cozmo Camera View"
        self._is_active = False
        self._latest_image = None
        self._lock = threading.Lock()
        self.detection_data = {}
        self._data_lock = threading.Lock()

    def set_detections(self, key, detections):
        with self._data_lock: self.detection_data[key] = detections

    def on_camera_image(self, cli, image):
        if not self._is_active: return
        img_np = np.array(image)
        if len(img_np.shape) == 3: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else: img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

        with self._data_lock:
            for key, det_list in self.detection_data.items():
                for det in det_list:
                    x1, y1, x2, y2 = det["box"]
                    label = det.get("label", "")
                    
                    # ラベルの文字を見て、枠の色を変える
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
                    # 枠線の上に文字（例：Cube FAR 65%）を表示して分かりやすくする
                    cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.battery_monitor: img_bgr = self.battery_monitor.draw_overlay(img_bgr)
        with self._lock: self._latest_image = img_bgr

    def start(self):
        self._is_active = True
        self.cli.enable_camera(enable=True, color=False) 
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)
        threading.Thread(target=self._display_loop, daemon=True).start()

    def _display_loop(self):
        while self._is_active:
            with self._lock:
                if self._latest_image is not None:
                    cv2.imshow(self.window_name, self._latest_image)
            cv2.waitKey(1) 

    def stop(self):
        self._is_active = False
        self.cli.enable_camera(enable=False)
        cv2.destroyAllWindows()