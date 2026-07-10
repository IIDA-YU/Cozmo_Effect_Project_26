import time
import random
import threading

BACKWARD_SPEED = -40
TURN_SPEED = 45

class CollisionPrevention:
    """
    Cozmoの回避行動（バック・旋回）と、その状態フラグだけを専門に管理するクラス。
    """
    def __init__(self, cli, face_manager=None, camera_viewer=None):
        self.cli = cli
        self.face_manager = face_manager
        self.camera_viewer = camera_viewer
        
        # 回避中かどうかのフラグをここで一括管理する
        self.is_avoiding_cliff = False
        self.is_avoiding_collision = False

        # ターゲットごとに検知回数と時間を管理する辞書
        self.detection_counts = {}
        self.last_detection_times = {}
        self.last_count_times = {}

    def safe_drive_wheels(self, l_speed, r_speed):
        if hasattr(self.cli, "conn_lock"):
            with self.cli.conn_lock:
                try: self.cli.drive_wheels(l_speed, r_speed)
                except Exception: pass
        else:
            try: self.cli.drive_wheels(l_speed, r_speed)
            except Exception: pass

    def trigger_cliff_avoidance(self):
        """段差回避のスイッチを入れる"""
        if not self.is_avoiding_cliff:
            self.is_avoiding_cliff = True
            threading.Thread(target=self._cliff_routine, daemon=True).start()

    def _cliff_routine(self):
        try:
            print("🚨 【警告】段差を検知！回避します。")
            if self.camera_viewer:
                self.camera_viewer.set_detections("cube", [])
                self.camera_viewer.set_detections("cozmo", [])
                
            self.safe_drive_wheels(0, 0) 
            self.safe_drive_wheels(BACKWARD_SPEED, BACKWARD_SPEED) 
            time.sleep(1.5)                     
            self.safe_drive_wheels(0, 0)         
            
            turn_direction = random.choice([1, -1])
            turn_time = random.uniform(1.5, 2.5)
            self.safe_drive_wheels(turn_direction * TURN_SPEED, -turn_direction * TURN_SPEED)
            time.sleep(turn_time)
        finally:
            self.safe_drive_wheels(0, 0) 
            time.sleep(1.5)
            self.is_avoiding_cliff = False 
            print("✅ 段差の回避完了。前進を再開します。")

    def trigger_obstacle_avoidance(self, target_name, distance=9999, position="center"):
        """障害物回避のスイッチを入れる（機体サイズ考慮・誤検知防止付き）"""
        if self.is_avoiding_cliff or self.is_avoiding_collision:
            return

        current_time = time.time()
        last_det_time = self.last_detection_times.get(target_name, 0)
        last_cnt_time = self.last_count_times.get(target_name, 0)

        # 今回見えた時間を記録
        self.last_detection_times[target_name] = current_time

        # 1.5秒以上見失っていたらカウントリセット
        if current_time - last_det_time > 1.5:
            self.detection_counts[target_name] = 1
            self.last_count_times[target_name] = current_time
            return

        # 前回カウントを増やしてから 0.4秒以上 経過していないとカウントしない
        if current_time - last_cnt_time >= 0.4:
            self.detection_counts[target_name] = self.detection_counts.get(target_name, 1) + 1
            self.last_count_times[target_name] = current_time
            
            # Cozmoの機体幅の半分（中心から角までの距離）を約30mmとする
            COZMO_HALF_WIDTH_MM = 30.0 
            
            # カメラ(中心)からの距離ではなく「機体の角」からの実質距離を計算する
            actual_distance = distance
            if position in ["left", "right"]:
                actual_distance = distance - COZMO_HALF_WIDTH_MM

            print(f"👀 {target_name}を継続検知中... 連続 {self.detection_counts[target_name]} 回目 (カメラ距離: {distance:.1f} mm, 機体からの実質距離: {actual_distance:.1f} mm, 位置: {position})")

            # カウントが2以上（最低0.4秒見え続けている）かつ 実質距離がAVOID_DISTANCE_THRESHOLD_MM以内に入ったら回避
            if self.detection_counts[target_name] >= 2 and actual_distance <= 100.0:  # 100mm以内に入ったら回避
                self.is_avoiding_collision = True
                self.detection_counts[target_name] = 0  # 発動したらカウントをリセット
                threading.Thread(target=self._obstacle_routine, args=(target_name, position), daemon=True).start()

    def _obstacle_routine(self, target_name, position="center"):
        try:
            print(f"⚠️ {target_name}の接近を検知！衝突を回避します。")
            if self.camera_viewer:
                self.camera_viewer.set_detections("cube", [])
                self.camera_viewer.set_detections("cozmo", [])

            if self.face_manager:
                self.face_manager.show_angry_face()
                
            if self.is_avoiding_cliff: return
            self.safe_drive_wheels(0, 0)
            
            if self.is_avoiding_cliff: return
            self.safe_drive_wheels(BACKWARD_SPEED, BACKWARD_SPEED) 
            time.sleep(0.5)
            
            if self.is_avoiding_cliff: return
            
            # 旋回時間（大きく旋回させたい場合は 1.5, 2.5 などに調整してください）
            turn_time = random.uniform(1.5, 2.5)
            
            # 障害物の逆方向へ逃げる
            if position == "left":
                direction = 1  # 右へ旋回
                print("↪️ 障害物が左にあるため、右へ旋回して回避します")
            elif position == "right":
                direction = -1 # 左へ旋回
                print("↩️ 障害物が右にあるため、左へ旋回して回避します")
            else:
                direction = random.choice([1, -1]) # 中央ならランダム
                print("🔄 障害物が正面にいるため、ランダムな方向へ回避します")
                
            self.safe_drive_wheels(direction * TURN_SPEED, -direction * TURN_SPEED)
            
            t = 0.0
            while t < turn_time:
                if self.is_avoiding_cliff: return 
                time.sleep(0.1)
                t += 0.1
                
        finally:
            self.safe_drive_wheels(0, 0) 
            if self.face_manager:
                self.face_manager.show_default_face()

            # 回避行動をした後に1秒止まる
            time.sleep(1.0)
            self.is_avoiding_collision = False
            print(f"✅ {target_name}の回避完了。前進を再開します。")