import os
os.environ["YOLO_OFFLINE"] = "True"  # 💡完全オフライン動作の強制（必ず一番上に書く）

import pycozmo
import cv2
import numpy as np
from ultralytics import YOLO 
import time
import math
import json

class CozmoSlamController:
    def __init__(self, model_path="learning_data/best_powercube.pt"):
        print(f"ローカルAIモデル({model_path})を準備中...")
        self.model = YOLO(model_path) 

        # --- 教室デスク用最適化パラメータ ---
        self.REAL_MARKER_WIDTH_MM = 25.0  
        self.FOCAL_LENGTH = 235.0        
        self.FOV_H_RADIANS = math.radians(60)
        self.MAX_VIEW_DIST = 600.0  
        self.EXPECTED_CUBES = 6     
        self.TARGET_HEAD_ANGLE = -0.15 
        self.MAP_SIZE = 450            
        self.MAP_SCALE = 0.5           
        
        # --- 状態変数 ---
        self.memory_map = np.full((self.MAP_SIZE, self.MAP_SIZE, 3), 127, dtype=np.uint8)
        self.true_landmarks = {}  
        self.persistent_mapped_cubes = [] 
        self.landmark_id_counter = 0
        self.offset_x, self.offset_y, self.offset_theta = 0.0, 0.0, 0.0
        self.current_raw_x, self.current_raw_y = 0.0, 0.0 
        self.target_x = None
        self.target_y = None
        self.latest_image = None
        self.robot_trajectory = [] 
        self.distance_history = {}
        self.ALPHA = 0.2 

    def on_camera_image(self, cli, image):
        img_np = np.array(image)
        self.latest_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    def to_map_coords(self, x_mm, y_mm):
        offset = self.MAP_SIZE // 2
        mx = int(-y_mm * self.MAP_SCALE + offset)
        my = int(-x_mm * self.MAP_SCALE + offset)
        return (mx, my)

    def from_map_coords(self, mx, my):
        offset = self.MAP_SIZE // 2
        y_mm = -(mx - offset) / self.MAP_SCALE
        x_mm = -(my - offset) / self.MAP_SCALE
        return x_mm, y_mm

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            tx, ty = self.from_map_coords(x, y)
            self.target_x = tx
            self.target_y = ty
            print(f"🎯 目標地点を設定しました: X:{self.target_x:.0f}mm, Y:{self.target_y:.0f}mm")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            rx, ry = self.from_map_coords(x, y)
            self.offset_x = rx - self.current_raw_x
            self.offset_y = ry - self.current_raw_y
            self.distance_history.clear() 
            print(f"🚁 誘拐リカバリ！現在地を強制リセットしました")

    def draw_memory_map(self, est_x, est_y, est_theta):
        map_img = self.memory_map.copy()
        
        if len(self.robot_trajectory) > 1:
            for i in range(1, len(self.robot_trajectory)):
                pt1 = self.to_map_coords(self.robot_trajectory[i-1][0], self.robot_trajectory[i-1][1])
                pt2 = self.to_map_coords(self.robot_trajectory[i][0], self.robot_trajectory[i][1])
                cv2.line(map_img, pt1, pt2, (255, 200, 50), 1) 
                
        for (mx, my) in self.persistent_mapped_cubes:
            cv2.circle(map_img, self.to_map_coords(mx, my), int(25 * self.MAP_SCALE), (0, 0, 255), -1)
        
        if len(self.true_landmarks) >= 2:
            landmark_pts = []
            for l_id, (lx, ly) in self.true_landmarks.items():
                landmark_pts.append(self.to_map_coords(lx, ly))
            cv2.polylines(map_img, [np.int32(landmark_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                
        if self.target_x is not None and self.target_y is not None:
            t_pt = self.to_map_coords(self.target_x, self.target_y)
            cv2.drawMarker(map_img, t_pt, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

        r_pt = self.to_map_coords(est_x, est_y)
        dx = int(math.sin(est_theta) * -20)
        dy = int(math.cos(est_theta) * -20)
        cv2.circle(map_img, r_pt, 8, (0, 255, 0), -1)
        cv2.line(map_img, r_pt, (r_pt[0] + dx, r_pt[1] + dy), (0, 0, 0), 2)
        
        return map_img

    def update_fov_map(self, est_x, est_y, est_theta):
        rob_pt = self.to_map_coords(est_x, est_y)
        left_angle = est_theta + (self.FOV_H_RADIANS / 2)
        right_angle = est_theta - (self.FOV_H_RADIANS / 2)
        offset = self.MAP_SIZE // 2
        l_pt = (int(-(est_y + self.MAX_VIEW_DIST*math.sin(left_angle)) * self.MAP_SCALE + offset), int(-(est_x + self.MAX_VIEW_DIST*math.cos(left_angle)) * self.MAP_SCALE + offset))
        r_pt = (int(-(est_y + self.MAX_VIEW_DIST*math.sin(right_angle)) * self.MAP_SCALE + offset), int(-(est_x + self.MAX_VIEW_DIST*math.cos(right_angle)) * self.MAP_SCALE + offset))
        cv2.fillPoly(self.memory_map, np.array([[rob_pt, l_pt, r_pt]], dtype=np.int32), (255, 255, 255))

    def get_map_data(self):
        """マッピングされた情報を辞書形式で返す"""
        return {
            "landmarks": self.true_landmarks,
            "trajectory": self.robot_trajectory,
            "mapped_cubes": self.persistent_mapped_cubes
        }

    def save_map_data(self, json_filename="slam_data.json", img_filename="slam_map.png"):
        """マッピングされた情報をファイルに保存する"""
        data = self.get_map_data()
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        cv2.imwrite(img_filename, self.memory_map)
        print(f"💾 マップデータを保存しました: {json_filename}, {img_filename}")

    def run(self):
        cli = pycozmo.Client()
        cli.start()
        cli.connect()
        cli.wait_for_robot()
        
        print("🤖 首のキャリブレーション（基準合わせ）を実行中...")
        time.sleep(0.5)
        cli.set_head_angle(-1.0)  
        time.sleep(1.5) 
        
        cli.set_head_angle(self.TARGET_HEAD_ANGLE)  
        cli.set_lift_height(0.0)  
        time.sleep(1.5)           
        
        cv2.namedWindow("Cozmo SLAM Map")
        cv2.setMouseCallback("Cozmo SLAM Map", self.mouse_callback)
        
        try:
            cli.enable_camera()
            cli.add_handler(pycozmo.event.EvtNewRawCameraImage, self.on_camera_image)
            
            last_head_cmd_time = time.time()
            print(f"🔄 初期スキャンを開始します（{self.EXPECTED_CUBES}つのキューブを探します）")
            scan_start_time = time.time()
            loop_closed = False
            
            # === 1. 初期スキャンフェーズ ===
            while time.time() - scan_start_time < 60.0 and not loop_closed:
                current_time = time.time()
                if current_time - last_head_cmd_time > 1.5:
                    cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                    last_head_cmd_time = current_time
                
                cli.drive_wheels(-12.0, 12.0) 
                
                if self.latest_image is not None and cli.pose is not None:
                    self.current_raw_x = cli.pose.position.x
                    self.current_raw_y = cli.pose.position.y
                    raw_theta = cli.pose.rotation.angle_z.radians
                    
                    est_x = self.current_raw_x + self.offset_x
                    est_y = self.current_raw_y + self.offset_y
                    est_theta = raw_theta + self.offset_theta
                    
                    self.robot_trajectory.append((est_x, est_y))
                    if len(self.robot_trajectory) > 2000: self.robot_trajectory.pop(0)
                        
                    self.update_fov_map(est_x, est_y, est_theta)
                    
                    try:
                        results = self.model.track(self.latest_image, persist=True, tracker="botsort.yaml", verbose=False)
                        boxes = results[0].boxes
                        
                        best_pred = None
                        min_pixel_offset = 9999
                        
                        for box in boxes:
                            cx, cy, w, h = map(int, box.xywh[0].tolist())
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            
                            if conf < 0.55 or h == 0 or (w/h) < 0.70 or (w/h) > 1.30: 
                                continue
                            
                            pixel_offset = cx - (self.latest_image.shape[1] / 2)
                            if abs(pixel_offset) < min_pixel_offset:
                                min_pixel_offset = abs(pixel_offset)
                                best_pred = {"x": cx, "y": cy, "width": w, "height": h, "id": track_id}

                        if best_pred is not None:
                            cx = best_pred["x"]
                            w = best_pred["width"]
                            t_id = best_pred["id"]
                            pixel_offset = cx - (self.latest_image.shape[1] / 2)
                            
                            if abs(pixel_offset) <= 120: 
                                raw_distance = (self.REAL_MARKER_WIDTH_MM * self.FOCAL_LENGTH) / w
                                
                                if t_id != -1:
                                    if t_id not in self.distance_history:
                                        self.distance_history[t_id] = raw_distance
                                    else:
                                        self.distance_history[t_id] = (self.ALPHA * raw_distance) + ((1 - self.ALPHA) * self.distance_history[t_id])
                                    distance = self.distance_history[t_id]
                                else:
                                    distance = raw_distance

                                angle_offset = (pixel_offset / (self.latest_image.shape[1] / 2)) * (self.FOV_H_RADIANS / 2)
                                observed_x = est_x + distance * math.cos(est_theta - angle_offset)
                                observed_y = est_y + distance * math.sin(est_theta - angle_offset)
                                
                                matched_id = None
                                for l_id, (tx, ty) in self.true_landmarks.items():
                                    if math.hypot(observed_x - tx, observed_y - ty) < 150.0: 
                                        matched_id = l_id; break
                                
                                if matched_id is None:
                                    if len(self.true_landmarks) < self.EXPECTED_CUBES:
                                        self.true_landmarks[self.landmark_id_counter] = (observed_x, observed_y)
                                        self.persistent_mapped_cubes.append((observed_x, observed_y))
                                        print(f"📌 初期スキャンで捕捉！ 確定ランドマーク [{self.landmark_id_counter}]: 距離 {distance:.0f}mm")
                                        self.landmark_id_counter += 1
                                else:
                                    if len(self.true_landmarks) == self.EXPECTED_CUBES and matched_id == 0:
                                        drift_x = observed_x - self.true_landmarks[0][0]
                                        drift_y = observed_y - self.true_landmarks[0][1]
                                        print(f"🎉 ループクロージャ発動！ 空間補正します！")
                                        
                                        drift_x = max(min(drift_x, 100.0), -100.0)
                                        drift_y = max(min(drift_y, 100.0), -100.0)
                                        
                                        for i in range(1, self.EXPECTED_CUBES):
                                            factor = i / self.EXPECTED_CUBES
                                            old_x, old_y = self.true_landmarks[i]
                                            self.true_landmarks[i] = (old_x - drift_x * factor, old_y - drift_y * factor)
                                        
                                        self.persistent_mapped_cubes.clear()
                                        for pt in self.true_landmarks.values():
                                            self.persistent_mapped_cubes.append(pt)
                                        
                                        loop_closed = True 
                    except Exception as e:
                        pass
                    
                    map_img = self.draw_memory_map(est_x, est_y, est_theta)
                    cv2.imshow("Cozmo SLAM Map", map_img)
                    cv2.waitKey(1)
                time.sleep(0.05)
                
            print(f"✅ スキャン完了！（クリック待ちです）")
            cli.drive_wheels(0.0, 0.0) 
            
            # === 2. ナビゲーションフェーズ ===
            while True:
                current_time = time.time()
                if current_time - last_head_cmd_time > 1.5:
                    cli.set_head_angle(self.TARGET_HEAD_ANGLE)
                    last_head_cmd_time = current_time
                
                if self.latest_image is not None and cli.pose is not None:
                    display_img = self.latest_image.copy()
                    self.current_raw_x = cli.pose.position.x
                    self.current_raw_y = cli.pose.position.y
                    raw_theta = cli.pose.rotation.angle_z.radians
                    
                    est_x = self.current_raw_x + self.offset_x
                    est_y = self.current_raw_y + self.offset_y
                    est_theta = raw_theta + self.offset_theta
                    
                    self.robot_trajectory.append((est_x, est_y))
                    if len(self.robot_trajectory) > 2000: self.robot_trajectory.pop(0)
                        
                    self.update_fov_map(est_x, est_y, est_theta)
                    correction_triggered = False
                    
                    try:
                        results = self.model.track(display_img, persist=True, tracker="botsort.yaml", verbose=False)
                        boxes = results[0].boxes
                        
                        best_pred = None
                        min_pixel_offset = 9999
                        
                        for box in boxes:
                            cx, cy, w, h = map(int, box.xywh[0].tolist())
                            conf = float(box.conf[0])
                            track_id = int(box.id[0]) if box.id is not None else -1
                            
                            if conf < 0.55 or h == 0 or (w/h) < 0.70 or (w/h) > 1.30: 
                                continue
                            
                            cv2.rectangle(display_img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0, 255, 0), 2)
                            
                            pixel_offset = cx - (display_img.shape[1] / 2)
                            if abs(pixel_offset) < min_pixel_offset:
                                min_pixel_offset = abs(pixel_offset)
                                best_pred = {"x": cx, "y": cy, "width": w, "height": h, "id": track_id}

                        if best_pred is not None:
                            cx = best_pred["x"]
                            w = best_pred["width"]
                            t_id = best_pred["id"]
                            pixel_offset = cx - (display_img.shape[1] / 2)
                            
                            if abs(pixel_offset) <= 120: 
                                raw_distance = (self.REAL_MARKER_WIDTH_MM * self.FOCAL_LENGTH) / w
                                
                                if t_id != -1:
                                    if t_id not in self.distance_history:
                                        self.distance_history[t_id] = raw_distance
                                    else:
                                        self.distance_history[t_id] = (self.ALPHA * raw_distance) + ((1 - self.ALPHA) * self.distance_history[t_id])
                                    distance = self.distance_history[t_id]
                                else:
                                    distance = raw_distance

                                angle_offset = (pixel_offset / (display_img.shape[1] / 2)) * (self.FOV_H_RADIANS / 2)
                                observed_x = est_x + distance * math.cos(est_theta - angle_offset)
                                observed_y = est_y + distance * math.sin(est_theta - angle_offset)
                                
                                matched_id = None
                                for l_id, (tx, ty) in self.true_landmarks.items():
                                    if math.hypot(observed_x - tx, observed_y - ty) < 150.0: 
                                        matched_id = l_id
                                        break
                                
                                if matched_id is None:
                                    if len(self.true_landmarks) < self.EXPECTED_CUBES:
                                        self.true_landmarks[self.landmark_id_counter] = (observed_x, observed_y)
                                        self.persistent_mapped_cubes.append((observed_x, observed_y))
                                        print(f"🌟 新規キューブを移動中に発見・追加！ [{self.landmark_id_counter}]")
                                        self.landmark_id_counter += 1
                                else:
                                    # SLAM補正処理
                                    tx, ty = self.true_landmarks[matched_id]
                                    angle_to_cube = math.atan2(ty - est_y, tx - est_x)
                                    heading_error = angle_to_cube - (est_theta - angle_offset)
                                    heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
                                    
                                    if abs(heading_error) < 0.5:
                                        self.offset_theta += heading_error * 0.05
                                        new_est_theta = raw_theta + self.offset_theta
                                        correct_rob_x = tx - distance * math.cos(new_est_theta - angle_offset)
                                        correct_rob_y = ty - distance * math.sin(new_est_theta - angle_offset)
                                        
                                        err_x = correct_rob_x - est_x
                                        err_y = correct_rob_y - est_y
                                        
                                        if math.hypot(err_x, err_y) < 100.0:
                                            self.offset_x += err_x * 0.05
                                            self.offset_y += err_y * 0.05
                                            correction_triggered = True
                    except Exception as e:
                        pass

                    out_of_bounds = False
                    if len(self.true_landmarks) >= 3:
                        pts = np.array(list(self.true_landmarks.values()), dtype=np.float32)
                        rect = cv2.minAreaRect(pts)
                        box = cv2.boxPoints(rect)
                        dist = cv2.pointPolygonTest(np.float32(box), (float(est_x), float(est_y)), True)
                        if dist < 20.0: out_of_bounds = True

                    if out_of_bounds:
                        cv2.putText(display_img, "VIRTUAL FENCE! ESCAPE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-25.0, -25.0)
                        time.sleep(0.6)
                        cli.drive_wheels(-15.0, 15.0)
                        time.sleep(0.8)
                        self.target_x, self.target_y = None, None 
                    
                    elif any(math.hypot(est_x - tx, est_y - ty) < 100.0 for tx, ty in self.true_landmarks.values()):
                        cv2.putText(display_img, "WALL BLOCK! STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-15.0, -15.0)
                        time.sleep(0.4)
                        self.target_x, self.target_y = None, None 
                    
                    elif self.target_x is not None and self.target_y is not None:
                        dist_to_target = math.hypot(self.target_x - est_x, self.target_y - est_y)
                        angle_to_target = math.atan2(self.target_y - est_y, self.target_x - est_x)
                        angle_diff = angle_to_target - est_theta
                        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                        
                        if dist_to_target < 15.0:
                            print("🎯 目標地点に到着しました！")
                            self.target_x, self.target_y = None, None
                            cli.drive_wheels(0.0, 0.0)
                        elif abs(angle_diff) > 0.3:
                            cv2.putText(display_img, "ADJUSTING HEADING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            if angle_diff > 0: cli.drive_wheels(-12.0, 12.0)
                            else: cli.drive_wheels(12.0, -12.0)
                        else:
                            cv2.putText(display_img, "NAVIGATING TO TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cli.drive_wheels(15.0, 15.0)
                    else:
                        cv2.putText(display_img, "WAITING FOR CLICK...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cli.drive_wheels(0.0, 0.0)

                    if correction_triggered:
                        cv2.putText(display_img, "SLAM FIXED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    cv2.imshow("Cozmo Camera", display_img)
                    map_img = self.draw_memory_map(est_x, est_y, est_theta)
                    cv2.imshow("Cozmo SLAM Map", map_img)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): 
                        print("🚪 アプリケーションを終了します...")
                        break
                    elif key == ord('a'): 
                        self.offset_theta -= 0.2
                        print("🔄 コンパスを左に修正しました")
                    elif key == ord('d'): 
                        self.offset_theta += 0.2
                        print("🔄 コンパスを右に修正しました")

                time.sleep(0.05)
                
        finally:
            cli.drive_wheels(0.0, 0.0)
            cli.disconnect(); cli.stop()
            cv2.destroyAllWindows()