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
            

    def trigger_obstacle_avoidance(self, target_name, distance=0):
        """障害物回避のスイッチを入れる（すべての対象物で誤検知防止付き）"""
        if self.is_avoiding_cliff or self.is_avoiding_collision:
            return

        current_time = time.time()

        # --- 変更ポイント: 距離に関わらず、見えているならカウントを進める ---
        last_time = self.last_detection_times.get(target_name, 0)
        
        # 前回の検知から1秒以上空いていればカウントをリセット（ノイズ対策）
        if current_time - last_time > 1.0:
            self.detection_counts[target_name] = 1
        else:
            self.detection_counts[target_name] = self.detection_counts.get(target_name, 0) + 1
        
        self.last_detection_times[target_name] = current_time

        print(f"👀 {target_name}を検知中... 連続 {self.detection_counts[target_name]} 回目 (距離: {distance:.1f} mm)")

        # --- 変更ポイント: カウントが3以上溜まっていて、かつ120mm以内に入ったら回避 ---
        if self.detection_counts[target_name] >= 3 and distance <= 120:
            self.is_avoiding_collision = True
            self.detection_counts[target_name] = 0  # 発動したらカウントをリセット
            threading.Thread(target=self._obstacle_routine, args=(target_name,), daemon=True).start()

    def _obstacle_routine(self, target_name):
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
            turn_time = random.uniform(0.8, 1.5)
            direction = random.choice([1, -1])
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
            time.sleep(1.5)

            self.is_avoiding_collision = False
            print(f"✅ {target_name}の回避完了。前進を再開します。")