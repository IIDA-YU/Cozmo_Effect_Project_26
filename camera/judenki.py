import cv2
import time
import numpy as np
from ultralytics import YOLO
import pycozmo

#YOLOのモデルをロードする
model = YOLO('best.pt')
is_docking = False 

#グローバル変数
latest_frame = None

# Cozmoから新しいカメラ画像が届くたびに呼び出される関数
def on_camera_image(cli, image):
    global latest_frame
    #print(">>> [INFO] Received new camera image from Cozmo.")
    # PIL ImageオブジェクトをNumPy配列に変換する
    frame = np.array(image)
    # OpenCVはBGR形式を使用するため、RGBからBGRに変換する
    latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def perform_docking_maneuver(cli):
    print(">>> [HÀNH ĐỘNG] Bắt đầu quay 180 độ...")
    cli.drive_wheels(lwheel_speed=-100.0, rwheel_speed=100.0)
    time.sleep(1.5)
    
    cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0)
    time.sleep(0.5)
    
    print(">>> [HÀNH ĐỘNG] Đang lùi về trạm sạc...")
    cli.drive_wheels(lwheel_speed=-70.0, rwheel_speed=-70.0)
    time.sleep(3.0)
    
    cli.drive_wheels(lwheel_speed=0.0, rwheel_speed=0.0)
    print(">>> [HÀNH ĐỘNG] Hoàn tất!")

def main():
    global is_docking, latest_frame
    
    cli = pycozmo.Client()
    cli.start()
    cli.connect()
    cli.wait_for_robot()
    print(">>> Đã kết nối với Cozmo!")
    
    # カメラを有効にする: これでCozmoがカメラ画像を送信し始めるようになります
    cli.enable_camera(enable=True, color=True)
    
    # 2. カメラ画像イベントのハンドラーを登録する
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)
    
    cli.set_head_angle(angle=0.0) 
    time.sleep(1) #１秒待ち
    
    while True:
        # 最新のフレームがまだない場合は、少し待ってから次のループに進む
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        # YOLOで処理するためにフレームをコピーする
        frame_to_process = latest_frame.copy()

        if not is_docking:
            # YOLOモデルを使用してフレームを処理する
            results = model(frame_to_process, conf=0.6) 
            
            found_mark = False
            found_station = False

            for r in results:
                for box in r.boxes:
                    class_name = model.names[int(box.cls[0])]
                    if class_name == "mark":
                        found_mark = True
                    elif class_name == "station":
                        found_station = True

            # AIの判断に基づいて行動する
            if found_mark:
                print("AI: Đã thấy 'mark'! -> Tiến hành lùi vào sạc.")
                is_docking = True
                perform_docking_maneuver(cli)
                
            elif found_station and not found_mark:
                print("AI: Thấy 'station' ở xa...")
                
            else:
                print("AI: Đang tìm kiếm trạm sạc...")

            # YOLOの結果をフレームに描画して表示する
            annotated_frame = results[0].plot()
            cv2.imshow("Cozmo's Eye - YOLO", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    #カメラを無効にしてクリーンアップする
    cli.enable_camera(enable=False)
    cli.disconnect()
    cli.stop()

if __name__ == '__main__':
    main()