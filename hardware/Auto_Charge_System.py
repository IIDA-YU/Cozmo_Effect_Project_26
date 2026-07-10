#----------------------------------------------------------------------
# auto_charge_system.py
# Cozmoの自動充電器回帰システムを管理するためのモジュールクラス
#----------------------------------------------------------------------

import cv2
import time
import numpy as np
from ultralytics import YOLO
import pycozmo

# YOLOモデルをロードする
model = YOLO('best.pt')
is_docking = False 

# グローバル変数
latest_frame = None

def on_camera_image(cli, image):
    global latest_frame
    frame = np.array(image)
    latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def perform_docking_maneuver(cli):
    print(">>> [アクション] 180度回転を開始します...")
    cli.drive_wheels(lwheel_speed=-100.0, rwheel_speed=100.0)
    time.sleep(1.5)
    
    cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0)
    time.sleep(0.5)
    
    print(">>> [アクション] 充電ステーションへバックしています...")
    cli.drive_wheels(lwheel_speed=-70.0, rwheel_speed=-70.0)
    time.sleep(3.0)
    
    cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0)
    print(">>> [アクション] 完了しました！")

def main():
    global is_docking, latest_frame
    
    cli = pycozmo.Client()
    cli.start()
    cli.connect()
    cli.wait_for_robot()
    print(">>> Cozmoとの接続に成功しました！")
    
    # ヘッドを正面に向けた後、カメラを有効化し画像処理を登録する
    print(">>> [システム] カメラを起動しています...")
    cli.enable_camera(enable=True, color=True)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)
    # -----------------------------------------------------------
    
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        frame_to_process = latest_frame.copy()

        if not is_docking:
            results = model(frame_to_process, conf=0.6) 
            
            mark_box = None
            found_station = False

            # 認識結果をループ処理で確認する
            for r in results:
                for box in r.boxes:
                    class_name = model.names[int(box.cls[0])]
                    if class_name == "mark":
                        mark_box = box 
                    elif class_name == "station":
                        found_station = True

            # YOLOからの情報に基づきアクションを決定する
            if mark_box is not None:
                xywh = mark_box.xywh[0]
                cx = float(xywh[0])
                width = float(xywh[2])

                SCREEN_CENTER_X = 160 
                error_x = cx - SCREEN_CENTER_X
                
                TOLERANCE_X = 25    
                TARGET_WIDTH = 60   

                if abs(error_x) > TOLERANCE_X:
                    turn_speed = 25.0
                    if error_x > 0: 
                        cli.drive_wheels(lwheel_speed=turn_speed, rwheel_speed=-turn_speed)
                        print(f"AI: 中央合わせ中（右回転）... ズレ: {error_x:.1f} px")
                    else:           
                        cli.drive_wheels(lwheel_speed=-turn_speed, rwheel_speed=turn_speed)
                        print(f"AI: 中央合わせ中（左回転）... ズレ: {error_x:.1f} px")
                
                elif width < TARGET_WIDTH:
                    cli.drive_wheels(lwheel_speed=40.0, rwheel_speed=40.0)
                    print(f"AI: 接近中... マークのサイズ: {width:.1f} / {TARGET_WIDTH}")
                
                else:
                    cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0)
                    print("AI: 基準位置に到達しました！方向転換を開始します。")
                    is_docking = True
                    perform_docking_maneuver(cli)
                    
            elif found_station and mark_box is None:
                print("AI: 'station'を検知しましたが、'mark'が見つかりません...")
                cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0) 
                
            else:
                print("AI: 何も検知できません。充電ステーションを探索中...")
                cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0) 

            # カメラ映像の表示
            annotated_frame = results[0].plot()
            cv2.imshow("Cozmo's Eye - YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    cli.enable_camera(enable=False)
    cli.disconnect()
    cli.stop()

if __name__ == '__main__':
    main()