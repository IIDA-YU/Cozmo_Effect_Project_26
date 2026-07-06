import os
os.environ["YOLO_OFFLINE"] = "True"  # 💡完全オフライン動作の強制

import pycozmo
import cv2
import numpy as np
from ultralytics import YOLO 
import time
import math
import random  # 💡追加: 回避時のランダムな方向転換に使用

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

true_landmarks = {}  
persistent_mapped_cubes = [] 
landmark_id_counter = 0

# 誘拐リカバリ（右クリック）用のオフセット
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
        print(f"🚁 誘拐リカバリ！現在地を X:{rx:.0f}mm, Y:{ry:.0f}mm に強制リセットしました")

def draw_memory_map(est_x, est_y, est_theta):
    global memory_map, target_x, target_y, robot_trajectory
    map_img = memory_map.copy()
    
    # ロボットの軌跡（水色の線）
    if len(robot_trajectory) > 1:
        for i in range(1, len(robot_trajectory)):
            pt1 = to_map_coords(robot_trajectory[i-1][0], robot_trajectory[i-1][1])
            pt2 = to_map_coords(robot_trajectory[i][0], robot_trajectory[i][1])
            cv2.line(map_img, pt1, pt2, (255, 200, 50), 1) 
            
    # 発見したキューブ（赤丸）
    for (mx, my) in persistent_mapped_cubes:
        cv2.circle(map_img, to_map_coords(mx, my), int(25 * MAP_SCALE), (0, 0, 255), -1)
    
    # キューブ同士を直線でつなぐ（緑色の線）
    if len(true_landmarks) >= 2:
        landmark_pts = []
        for l_id, (lx, ly) in true_landmarks.items():
            landmark_pts.append(to_map_coords(lx, ly))
        cv2.polylines(map_img, [np.int32(landmark_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
            
    # 目標地点のマーカー
    if target_x is not None and target_y is not None:
        t_pt = to_map_coords(target_x, target_y)
        cv2.drawMarker(map_img, t_pt, (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

    # ロボットの現在位置（緑の点と黒いコンパス）
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

def main():
    global latest_image, memory_map, true_landmarks, landmark_id_counter, persistent_mapped_cubes
    global offset_x, offset_y, offset_theta, target_x, target_y, robot_trajectory
    global current_raw_x, current_raw_y, distance_history
    
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
    
    cv2.namedWindow("Cozmo 2D SLAM Map")
    cv2.setMouseCallback("Cozmo 2D SLAM Map", mouse_callback)
    
    try:
        cli.enable_camera()
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)
        
        last_head_cmd_time = time.time()
        print(f"🗺️ 2D自動マッピングを開始します！（自律探査モード）")
        
        # 💡自律探査用のステート（状態）管理変数
        robot_state = "SCAN" # 初期状態は周囲を見渡す
        state_start_time = time.time()
        turn_direction = 1 # 1:左回転, -1:右回転
        
        while True:
            current_time = time.time()
            if current_time - last_head_cmd_time > 1.5:
                cli.set_head_angle(TARGET_HEAD_ANGLE)
                last_head_cmd_time = current_time
            
            out_of_bounds = False
            wall_block = False

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
                
                try:
                    results = model.track(display_img, persist=True, tracker="botsort.yaml", verbose=False)
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        cx, cy, w, h = map(int, box.xywh[0].tolist())
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        
                        if conf < 0.55 or h == 0 or (w/h) < 0.70 or (w/h) > 1.30: 
                            continue
                        
                        cv2.rectangle(display_img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0, 255, 0), 2)
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
                            cube_x = est_x + distance * math.cos(est_theta - angle_offset)
                            cube_y = est_y + distance * math.sin(est_theta - angle_offset)
                            
                            matched_id = None
                            for l_id, (tx, ty) in true_landmarks.items():
                                if math.hypot(cube_x - tx, cube_y - ty) < 80.0: # 8cm以内は同一キューブ
                                    matched_id = l_id
                                    # 平均を取って位置を滑らかに更新
                                    true_landmarks[l_id] = ((tx*4 + cube_x)/5, (ty*4 + cube_y)/5)
                                    persistent_mapped_cubes[l_id] = true_landmarks[l_id]
                                    break
                            
                            if matched_id is None:
                                if len(true_landmarks) < EXPECTED_CUBES:
                                    true_landmarks[landmark_id_counter] = (cube_x, cube_y)
                                    persistent_mapped_cubes.append((cube_x, cube_y))
                                    print(f"🌟 新規キューブをマップに追加！ [{landmark_id_counter}]")
                                    landmark_id_counter += 1
                except Exception as e:
                    pass

                # 💡安全判定：緑のフェンスやキューブに近づきすぎていないかチェック
                if len(true_landmarks) >= 3:
                    pts = np.array(list(true_landmarks.values()), dtype=np.float32)
                    rect = cv2.minAreaRect(pts)
                    box = cv2.boxPoints(rect)
                    dist = cv2.pointPolygonTest(np.float32(box), (float(est_x), float(est_y)), True)
                    if dist < 20.0: out_of_bounds = True

                if any(math.hypot(est_x - tx, est_y - ty) < 100.0 for tx, ty in true_landmarks.values()):
                    wall_block = True

                # === 💡自律探査AI（ステートマシン）のロジック ===
                if target_x is None and target_y is None: # マウスで目標地点が設定されていない時だけ自動で動く
                    
                    if robot_state == "SCAN":
                        cv2.putText(display_img, "[AUTO] SCANNING...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cli.drive_wheels(-15.0, 15.0) # ゆっくり右回転
                        # 6秒間スキャンしたら前進モードへ
                        if current_time - state_start_time > 6.0:
                            robot_state = "FORWARD"
                            state_start_time = current_time

                    elif robot_state == "FORWARD":
                        cv2.putText(display_img, "[AUTO] EXPLORING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cli.drive_wheels(25.0, 25.0) # 前進
                        
                        # 障害物やフェンスを検知したら回避モードへ移行
                        if out_of_bounds or wall_block:
                            robot_state = "AVOID_BACK"
                            state_start_time = current_time

                    elif robot_state == "AVOID_BACK":
                        cv2.putText(display_img, "[AUTO] OBSTACLE! BACKING UP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(-25.0, -25.0) # バック
                        
                        # 1.5秒バックしたら、ランダムな方向へ切り返す
                        if current_time - state_start_time > 1.5:
                            robot_state = "AVOID_TURN"
                            state_start_time = current_time
                            turn_direction = random.choice([-1, 1]) # 左右どちらに避けるかランダムに決める

                    elif robot_state == "AVOID_TURN":
                        cv2.putText(display_img, "[AUTO] TURNING AWAY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        cli.drive_wheels(-20.0 * turn_direction, 20.0 * turn_direction) # 指定した方向へ回転
                        
                        # 1.5秒回転して向きを変えたら、再び前進モードへ
                        if current_time - state_start_time > 1.5:
                            robot_state = "FORWARD"
                            state_start_time = current_time

                # === マウスクリックによるマニュアル誘導（オーバーライド） ===
                else:
                    dist_to_target = math.hypot(target_x - est_x, target_y - est_y)
                    angle_to_target = math.atan2(target_y - est_y, target_x - est_x)
                    angle_diff = angle_to_target - est_theta
                    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                    
                    if out_of_bounds or wall_block:
                        cv2.putText(display_img, "MANUAL OVERRIDE: BLOCKED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cli.drive_wheels(0.0, 0.0)
                        target_x, target_y = None, None # 危険な場合はマニュアル誘導をキャンセルして自動に戻る
                    elif dist_to_target < 15.0:
                        print("🎯 目標地点に到着しました！自動探索に戻ります。")
                        target_x, target_y = None, None
                        cli.drive_wheels(0.0, 0.0)
                    elif abs(angle_diff) > 0.3:
                        cv2.putText(display_img, "MANUAL: ADJUSTING HEADING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        if angle_diff > 0: cli.drive_wheels(-12.0, 12.0)
                        else: cli.drive_wheels(12.0, -12.0)
                    else:
                        cv2.putText(display_img, "MANUAL: NAVIGATING TO TARGET", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cli.drive_wheels(15.0, 15.0)

                cv2.imshow("Cozmo Camera", display_img)
                map_img = draw_memory_map(est_x, est_y, est_theta)
                cv2.imshow("Cozmo 2D SLAM Map", map_img)
                
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
        cli.disconnect(); cli.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__": main()