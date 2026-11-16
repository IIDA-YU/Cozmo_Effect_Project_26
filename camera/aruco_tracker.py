# camera/aruco_tracker.py
import cv2
import numpy as np
import pycozmo
import time
import threading
from camera.camera_movie import cozmo_image_to_bgr
from telecommunication.pc_communication import CoordCommunication

# ★ 距離推定パラメータ（Cozmoに貼るArUcoマーカーの実寸に合わせて調整可能）
MARKER_REAL_SIZE_MM = 30.0      # マーカー1辺の実寸(mm)
FOCAL_LENGTH_PIX = 235.0        # Cozmoカメラの焦点距離(px)
ENCOUNTER_DIST_MAX_MM = 450.0   # すれ違い通信を発動する最大距離(mm)


class ArucoCoordinateTracker:
    def __init__(
        self,
        cli,
        target_ip_dict,
        camera_viewer=None,
        comm=None,
        robot_id="Cozmo-A",
        cooldown_sec=4.0,
    ):
        self.cli = cli
        self.target_ip_dict = target_ip_dict or {}
        self.camera_viewer = camera_viewer
        self.robot_id = robot_id
        self.comm = comm if comm is not None else CoordCommunication(robot_id=robot_id)

        # ★ どのCoordCommunicationインスタンス(receiver)が応答しても実座標を返せるように共有登録
        CoordCommunication.register_robot_context(
            cli=self.cli,
            robot_id=self.robot_id,
            get_local_info_cb=self._get_live_pose,
        )
        if self.comm.get_local_info_cb is None:
            self.comm.get_local_info_cb = self._get_live_pose

        # ArUcoマーカー検知の初期設定（OpenCV新旧両対応）
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "DetectorParameters"):
            self.parameters = cv2.aruco.DetectorParameters()
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        else:
            self.detector = None

        # 相手(marker_id)ごとの最終通信時刻と通信中フラグ
        self.cooldown_sec = cooldown_sec
        self.last_exchange_times = {}
        self.exchanging_ids = set()

        self.running = False
        self._latest_raw_image = None
        self._lock = threading.Lock()

        print(f"📡 [ArucoTracker] 初期化完了: 監視対象ID群 {list(self.target_ip_dict.keys())}")

    def _get_live_pose(self):
        if self.cli and self.cli.pose:
            return {
                "id": self.robot_id,
                "x": float(self.cli.pose.position.x),
                "y": float(self.cli.pose.position.y),
            }
        return {"id": self.robot_id, "x": 0.0, "y": 0.0}

    def on_camera_image(self, cli, image):
        with self._lock:
            self._latest_raw_image = image

    def _estimate_marker_distance(self, pts):
        """4頂点(4x2配列)からマーカーのピクセル幅を求め、距離(mm)を推定する"""
        w_px = float(np.linalg.norm(pts[0] - pts[1]))
        h_px = float(np.linalg.norm(pts[1] - pts[2]))
        size_px = max(w_px, h_px, 1.0)
        return (MARKER_REAL_SIZE_MM * FOCAL_LENGTH_PIX) / size_px

    def _can_exchange_with(self, marker_id):
        now = time.time()
        if marker_id in self.exchanging_ids:
            return False
        last_t = self.last_exchange_times.get(marker_id, 0.0)
        return (now - last_t) >= self.cooldown_sec

    def _detect_markers(self, frame):
        if self.detector is not None:
            return self.detector.detectMarkers(frame)
        return cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.parameters)

    def _tracker_loop(self):
        while self.running:
            with self._lock:
                raw_image = self._latest_raw_image

            if raw_image is not None:
                try:
                    frame = cozmo_image_to_bgr(raw_image)
                    corners, ids, _ = self._detect_markers(frame)

                    boxes = []
                    if ids is not None and len(ids) > 0:
                        flat_ids = np.asarray(ids).reshape(-1)

                        for i, raw_id in enumerate(flat_ids):
                            marker_id = int(raw_id)

                            pts = np.asarray(corners[i], dtype=np.float32).reshape(-1, 2)
                            if pts.shape[0] < 4:
                                continue

                            x1 = int(np.min(pts[:, 0]))
                            y1 = int(np.min(pts[:, 1]))
                            x2 = int(np.max(pts[:, 0]))
                            y2 = int(np.max(pts[:, 1]))
                            dist_mm = self._estimate_marker_distance(pts)

                            recently_synced = (
                                time.time() - self.last_exchange_times.get(marker_id, 0.0) < 3.0
                            )
                            status_tag = " [SYNCED]" if recently_synced else ""
                            boxes.append({
                                "box": (x1, y1, x2, y2),
                                "label": f"ID:{marker_id} ({int(dist_mm)}mm){status_tag}",
                            })

                            # 🟢 対応表に登録されているCozmoとすれ違い距離内に入ったら双方向通信を開始
                            if (
                                marker_id in self.target_ip_dict
                                and dist_mm <= ENCOUNTER_DIST_MAX_MM
                                and self._can_exchange_with(marker_id)
                            ):
                                target_ip = self.target_ip_dict[marker_id]
                                self.exchanging_ids.add(marker_id)

                                my_x = float(self.cli.pose.position.x) if self.cli.pose else 0.0
                                my_y = float(self.cli.pose.position.y) if self.cli.pose else 0.0

                                print(
                                    f"🎯 すれ違い検知！Cozmo (ID: {marker_id}, 距離: 約{dist_mm:.0f}mm) "
                                    f"とリアルタイム相互通信を開始 -> {target_ip}"
                                )

                                def exchange_task(m_id, ip, mx, my):
                                    try:
                                        peer_reply = self.comm.exchange_data(
                                            target_ip=ip,
                                            x=mx,
                                            y=my,
                                            robot_id=self.robot_id,
                                        )
                                        if peer_reply:
                                            self.last_exchange_times[m_id] = time.time()
                                            print(
                                                f" ┗ ✅ [ID:{m_id}] 双方向交換完了！"
                                                f" (自機送信: X={mx:.1f}, Y={my:.1f})"
                                            )
                                        else:
                                            self.last_exchange_times[m_id] = (
                                                time.time() - self.cooldown_sec + 1.0
                                            )
                                            print(f" ┗ ❌ [ID:{m_id}] 相手({ip})からの応答なし")
                                    finally:
                                        self.exchanging_ids.discard(m_id)

                                threading.Thread(
                                    target=exchange_task,
                                    args=(marker_id, target_ip, my_x, my_y),
                                    daemon=True,
                                ).start()

                    if self.camera_viewer:
                        self.camera_viewer.set_detections("aruco", boxes)

                except Exception as e:
                    print(f"⚠️ ArUco検知エラー: {e}")

            time.sleep(0.05)

    def start(self):
        self.running = True
        threading.Thread(target=self._tracker_loop, daemon=True).start()
        self.cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)

    def stop(self):
        self.running = False
        if hasattr(self.cli, "del_handler"):
            try:
                self.cli.del_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)
            except Exception:
                pass