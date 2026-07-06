import os
os.environ["YOLO_OFFLINE"] = "True"  # 💡完全オフライン動作の強制（必ず一番上に書く）

import pycozmo
import cv2
import numpy as np
from ultralytics import YOLO 
import time
import math

print("ローカルAIモデル(best_powercube.pt)を準備中...")
model = YOLO("learning_data/best_powercube.pt") 

# --- 教室デスク用最適化パラメータ ---
REAL_MARKER_WIDTH_MM = 25.0  
FOCAL_LENGTH = 235.0         
FOV_H_RADIANS = math.radians(60)
MAX_VIEW_DIST = 600.0  
EXPECTED_CUBES = 6     

TARGET_HEAD_ANGLE = -0.15 

MAP_SIZE = 450             
MAP_SCALE = 0.5            
memory_map = np.full((MAP_SIZE, MAP_SIZE, 3), 127, dtype=np.uint8)

# 💡群知能用：仮想フェロモンマップの準備
PHEROMONE_FILE = "shared_pheromone.npy"
try:
    pheromone_map = np.load(PHEROMONE_FILE)
    print("💾 既存の共有フェロモンデータを読み込みました")
except FileNotFoundError:
    pheromone_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    print("📄 新規フェロモンマップを作成しました")

last_phero_sync_time = time.time()

true_landmarks = {}  
persistent_mapped_cubes = [] 
landmark_id_counter = 0
offset_x, offset_y, offset_theta = 0.0, 0.0, 0.0

current_raw_x, current_raw_y = 0.0, 0.0 

target_x = None
target_y = None
latest_image = None
robot_trajectory = [] 

distance_history = {}
ALPHA = 0.2 

def on_camera_image(cli, image):
    global latest_image
    img_np = np.array(image)
    latest_image = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

def to_map_coords(x_mm, y_mm):
    offset = MAP_SIZE // 2
    mx = int(-y_mm * MAP_SCALE + offset)
    my = int(-x_mm * MAP_SCALE + offset)
    return (mx, my)

def from_map_coords(mx, my):
    offset = MAP_SIZE // 2
    y_mm = -(mx - offset) / MAP_SCALE
    x_mm = -(my - offset) / MAP_SCALE
    return x_mm, y_mm

def mouse_callback(event, x, y, flags, param):
    global target_x, target_y, offset_x, offset_y, current_raw_x, current_raw_y, distance_history
    
    if event == cv2.EVENT_LBUTTONDOWN:
        tx, ty = from_map_coords(x, y)
        target_x = tx
        target_y = ty
        print(f"🎯 目標地点を設定しました: X:{target_x:.0f}mm, Y:{target_y:.0f}mm")
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        rx, ry = from_map_coords(x, y)
        offset_x = rx - current_raw_x
        offset_y = ry - current_raw_y
        distance_history.clear() 
        print(f"🚁 誘拐リカバリ！現在地を強制リセットしました")

def draw_memory_map(est_x, est_y, est_theta):
    global memory_map, target_x, target_y, robot_trajectory, pheromone_map
    map_img = memory_map.copy()
    
    # 💡フェロモンの可視化（シアン色で重ね合わせ）
    phero_overlay = np.zeros_like(map_img)
    phero_u8 = np.clip(pheromone_map, 0, 255).astype(np.uint8)
    phero_overlay[:, :, 0] = phero_u8 # Blue
    phero_overlay[:, :, 1] = phero_u8 # Green
    map_img = cv2.addWeighted(map_img, 1.0, phero_overlay, 0.8, 0)

    if len(robot_trajectory) > 1:
        for i in range(1, len(robot_trajectory)):
            pt1 = to_map_coords(robot_trajectory[i-1][0], robot_trajectory[i-1][1])
            pt2 = to_map_coords(robot_trajectory[i][0], robot_trajectory[i][1])
            cv2.line(map_img, pt1, pt2, (255, 200, 50), 1) 
            
    for (mx, my) in persistent_mapped_cubes:
        cv2.circle(map_img, to_map_coords(mx, my), int(25 * MAP_SCALE), (0, 0, 255), -1)
    
    if len(true_landmarks) >= 2:
        landmark_pts = []
        for l_id, (lx, ly) in true_landmarks.items():
            landmark_pts.append(to_map_coords(lx, ly))
        cv2.polylines(map_img, [np.int32(landmark_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
            
    if target_x is not None and target_y is not None:
        t_pt = to_map_coords(target_x, target_y)
        cv2.drawMarker(map_img, t_pt, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

    r_pt = to_map_coords(est_x, est_y)
    dx = int(math.sin(est_theta) * -20)
    dy = int(math.cos(est_theta) * -20)
    cv2.circle(map_img, r_pt, 8, (0, 255, 0), -1)
    cv2.line(map_img, r_pt, (r_pt[0] + dx, r_pt[1] + dy), (0, 0, 0), 2)
    
    return map_img

def update_fov_map(est_x, est_y, est_theta):
    global memory_map
    rob_pt = to_map_coords(est_x, est_y)
    left_angle = est_theta + (FOV_H_RADIANS / 2)
    right_angle = est_theta - (FOV_H_RADIANS / 2)
    offset = MAP_SIZE // 2
    l_pt = (int(-(est_y + MAX_VIEW_DIST*math.sin(left_angle)) * MAP_SCALE + offset), int(-(est_x + MAX_VIEW_DIST*math.cos(left_angle)) * MAP_SCALE + offset))
    r_pt = (int(-(est_y + MAX_VIEW_DIST*math.sin(right_angle)) * MAP_SCALE + offset), int(-(est_x + MAX_VIEW_DIST*math.cos(right_angle)) * MAP_SCALE + offset))
    cv2.fillPoly(memory_map, np.array([[rob_pt, l_pt, r_pt]], dtype=np.int32), (255, 255, 255))

def process_pheromone(est_x, est_y, current_time):
    global pheromone_map, last_phero_sync_time
    
    # 1. 自身のフェロモンを分泌（少し太めのブラシで塗る）
    mx, my = to_map_coords(est_x, est_y)
    if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
        cv2.circle(pheromone_map, (mx, my), 3, 50.0, -1)
        np.clip(pheromone_map, 0, 255, out=pheromone_map)

    # 2. 時間経過によるフェロモンの蒸発（減衰）
    pheromone_map *= 0.995 

    # 3. 3秒ごとにファイルを通じて他機体とフェロモンを同期
    if current_time - last_phero_sync_time > 3.0:
        try:
            if os.path.exists(PHEROMONE_FILE):
                shared_map = np.load(PHEROMONE_FILE)
                # 他のロボットのフェロモンと自分のフェロモンを合成（強い方を採用）
                pheromone_map = np.maximum(pheromone_map, shared_map)
        except Exception:
            pass
        # 合成した最新のフェロモンマップを上書き保存
        np.save(PHEROMONE_FILE, pheromone_map)
        last_phero_sync_time = current_time

def main():
    global latest_image, memory_map, true_landmarks, landmark_id_counter, persistent_mapped_cubes
    global offset_x, offset_y, offset_theta, target_x, target_y, robot_trajectory
    global current_raw_x, current_raw_y, distance_history
    global pheromone_map, last_phero_sync_time
    
    cli = pycozmo.Client()
    cli.start()
    cli.connect()
    cli.wait_for_robot()
    
    print("🤖 首のキャリブレーション（基準合わせ）を実行中...")
    time.sleep(0.5)
    cli.set_head_angle(-1.0)  
    time.sleep(1.5) 
    
    cli.set_head_angle(TARGET_HEAD_ANGLE)  
    cli.set_lift_height(0.0)  
    time.sleep(1.5)           
    
    cv2.namedWindow("Cozmo SLAM Map")
    cv2.setMouseCallback("Cozmo SLAM Map", mouse_callback)
    
    try:
        cli.enable_camera()
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)
        
        last_head_cmd_time = time.time()
        print(f"🔄 初期スキャンを開始します（{EXPECTED_CUBES}つのキューブを探します）")
        scan_start_time = time.time()
        loop_closed = False
        
        # === 1. 初期スキャンフェーズ ===
        while time.time() - scan_start_time < 60.0 and not loop_closed:
            current_time = time.time()
            if current_time - last_head_cmd_time > 1.5:
                cli.set_head_angle(TARGET_HEAD_ANGLE)
                last_head_cmd_time = current_time
            
            cli.drive_wheels(-12.0, 12.0) 
            
            if latest_image is not None and cli.pose is not None:
                current_raw_x = cli.pose.position.x
                current_raw_y = cli.pose.position.y
                raw_theta = cli.pose.rotation.angle_z.radians
                
                est_x = current_raw_x + offset_x
                est_y = current_raw_y + offset_y
                est_theta = raw_theta + offset_theta
                
                robot_trajectory.append((est_x, est_y))
                if len(robot_trajectory) > 2000: robot_trajectory.pop(0)
                    
                update_fov_map(est_x, est_y, est_theta)
                
                # 💡フェロモンの処理を実行
                process_pheromone(est_x, est_y, current_time)

                try:
                    results = model.track(latest_image, persist=True, tracker="botsort.yaml", verbose=False)
                    boxes = results[0].boxes
                    
                    best_pred = None
                    min_pixel_offset = 9999
                    
                    for box in boxes:
                        cx, cy, w, h = map(int, box.xywh[0].tolist())
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        
                        if conf < 0.55 or h == 0 or (w/h) < 0.70 or (w/h) > 1.30: 
                            continue
                        
                        pixel_offset = cx - (latest_image.shape[1] / 2)
                        if abs(pixel_offset) < min_pixel_offset:
                            min_pixel_offset = abs(pixel_offset)
                            best_pred = {"x": cx, "y": cy, "width": w, "height": h, "id": track_id}

                    if best_pred is not None:
                        cx = best_pred["x"]
                        w = best_pred["width"]
                        t_id = best_pred["id"]
                        pixel_offset = cx - (latest_image.shape[1] / 2)
                        
                        if abs(pixel_offset) <= 120: 
                            raw_distance = (REAL_MARKER_WIDTH_MM * FOCAL_LENGTH) / w
                            
                            if t_id != -1:
                                if t_id not in distance_history:
                                    distance_history[t_id] = raw_distance
                                else:
                                    distance_history[t_id] = (ALPHA * raw_distance) + ((1 - ALPHA) * distance_history[t_id])
                                distance = distance_history[t_id]
                            else:
                                distance = raw_distance

                            angle_offset = (pixel_offset / (latest_image.shape[1] / 2)) * (FOV_H_RADIANS / 2)
                            observed_x = est_x + distance * math.cos(est_theta - angle_offset)
                            observed_y = est_y + distance * math.sin(est_theta - angle_offset)
                            
                            matched_id = None
                            for l_id, (tx, ty) in true_landmarks.items():
                                if math.hypot(observed_x - tx, observed_y - ty) < 150.0: 
                                    matched_id = l_id; break
                            
                            if matched_id is None:
                                if len(true_landmarks) < EXPECTED_CUBES:
                                    true_landmarks[landmark_id_counter] = (observed_x, observed_y)
                                    persistent_mapped_cubes.append((observed_x, observed_y))
                                    print(f"📌 初期スキャンで捕捉！ 確定ランドマーク [{landmark_id_counter}]: 距離 {distance:.0f}mm")
                                    landmark_id_counter += 1
                            else:
                                if len(true_landmarks) == EXPECTED_CUBES and matched_id == 0:
                                    drift_x = observed_x - true_landmarks[0][0]
                                    drift_y = observed_y - true_landmarks[0][1]
                                    print(f"🎉 ループクロージャ発動！ 空間補正します！")
                                    
                                    drift_x = max(min(drift_x, 100.0), -100.0)
                                    drift_y = max(min(drift_y, 100.0), -100.0)
                                    
                                    for i in range(1, EXPECTED_CUBES):
                                        factor = i / EXPECTED_CUBES
                                        old_x, old_y = true_landmarks[i]
                                        true_landmarks[i] = (old_x - drift_x * factor, old_y - drift_y * factor)
                                    
                                    persistent_mapped_cubes.clear()
                                    for pt in true_landmarks.values():
                                        persistent_mapped_cubes.append(pt)
                                    
                                    loop_closed = True 
                except Exception as e:
                    pass
                
                map_img = draw_memory_map(est_x, est_y, est_theta)
                cv2.imshow("Cozmo SLAM Map", map_img)
                cv2.waitKey(1)
            time.sleep(0.05)
            
        print(f"✅ スキャン完了！（クリック待ちです）")
        cli.drive_wheels(0.0, 0.0) 
        
        # === 2. ナビゲーションフェーズ ===
        while True:
            current_time = time.time()
            if current_time - last_head_cmd_time > 1.5:
                cli.set_head_angle(TARGET_HEAD_ANGLE)
                last_head_cmd_time = current_time
            
            if latest_image is not None and cli.pose is not None:
                display_img = latest_image.copy()
                current_raw_x = cli.pose.position.x
                current_raw_y = cli.pose.position.y
                raw_theta = cli.pose.rotation.angle_z.radians
                
                est_x = current_raw_x + offset_x
                est_y = current_raw_y + offset_y
                est_theta = raw_theta + offset_theta
                
                robot_trajectory.append((est_x, est_y))
                if len(robot_trajectory) > 2000: robot_trajectory.pop(0)
                    
                update_fov_map(est_x, est_y, est_theta)
                
                # 💡フェロモンの処理を実行
                process_pheromone(est_x, est_y, current_time)

                correction_triggered = False
                
                try:
                    results = model.track(display_img, persist=True, tracker="botsort.yaml", verbose=False)
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
                            raw_distance = (REAL_MARKER_WIDTH_MM * FOCAL_LENGTH) / w
                            
                            if t_id != -1:
                                if t_id not in distance_history:
                                    distance_history[t_id] = raw_distance
                                else:
                                    distance_history[t_id] = (ALPHA * raw_distance) + ((1 - ALPHA) * distance_history[t_id])
                                distance = distance_history[t_id]
                            else:
                                distance = raw_distance

                            angle_offset = (pixel_offset / (display_img.shape[1] / 2)) * (FOV_H_RADIANS / 2)
                            observed_x = est_x + distance * math.cos(est_theta - angle_offset)
                            observed_y = est_y + distance * math.sin(est_theta - angle_offset)
                            
                            matched_id = None
                            for l_id, (tx, ty) in true_landmarks.items():
                                if math.hypot(observed_x - tx, observed_y - ty) < 150.0: 
                                    matched_id = l_id
                                    break
                            
                            if matched_id is not None:
                                tx, ty = true_landmarks[matched_id]
                                angle_to_cube = math.atan2(ty - est_y, tx - est_x)
                                heading_error = angle_to_cube - (est_theta - angle_offset)
                                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
                                
                                if abs(heading_error) < 0.5:
                                    offset_theta += heading_error * 0.05
                                    new_est_theta = raw_theta + offset_theta
                                    correct_rob_x = tx - distance * math.cos(new_est_theta - angle_offset)
                                    correct_rob_y = ty - distance * math.sin(new_est_theta - angle_offset)
                                    
                                    err_x = correct_rob_x - est_x
                                    err_y = correct_rob_y - est_y
                                    
                                    if math.hypot(err_x, err_y) < 100.0:
                                        offset_x += err_x * 0.05
                                        offset_y += err_y * 0.05
                                        correction_triggered = True
                                    
                            else:
                                if len(true_landmarks) < EXPECTED_CUBES:
                                    true_landmarks[landmark_id_counter] = (observed_x, observed_y)
                                    persistent_mapped_cubes.append((observed_x, observed_y))
                                    print(f"🌟 新規キューブを移動中に発見・追加！ [{landmark_id_counter}]")
                                    landmark_id_counter += 1
                except Exception as e:
                    pass

                out_of_bounds = False
                if len(true_landmarks) >= 3:
                    pts = np.array(list(true_landmarks.values()), dtype=np.float32)
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
                    target_x, target_y = None, None 
                
                elif any(math.hypot(est_x - tx, est_y - ty) < 100.0 for tx, ty in true_landmarks.values()):
                    cv2.putText(display_img, "WALL BLOCK! STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cli.drive_wheels(-15.0, -15.0)
                    time.sleep(0.4)
                    target_x, target_y = None, None 
                
                elif target_x is not None and target_y is not None:
                    dist_to_target = math.hypot(target_x - est_x, target_y - est_y)
                    angle_to_target = math.atan2(target_y - est_y, target_x - est_x)
                    angle_diff = angle_to_target - est_theta
                    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                    
                    if dist_to_target < 15.0:
                        print("🎯 目標地点に到着しました！")
                        target_x, target_y = None, None
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
                map_img = draw_memory_map(est_x, est_y, est_theta)
                cv2.imshow("Cozmo SLAM Map", map_img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('a'): 
                    offset_theta -= 0.2
                    print("🔄 コンパスを左に修正しました")
                elif key == ord('d'): 
                    offset_theta += 0.2
                    print("🔄 コンパスを右に修正しました")

            time.sleep(0.05)
            
    finally:
        cli.drive_wheels(0.0, 0.0)
        # 終了時にフェロモンを最終保存
        np.save(PHEROMONE_FILE, pheromone_map)
        print("💾 フェロモンデータを保存して終了します")
        cli.disconnect(); cli.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__": main()