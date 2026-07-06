import time

from connect.network_check import cozmo_session
from hardware.buttery import BatteryMonitor
from camera.camera_movie import CameraViewer

# ===== 設定（自分の環境に合わせて書き換えてください）=====
COZMO_SSID = "Cozmo_613F59"   # 自分のCozmoのSSID
RETURN_SSID = "R0347"         # 終了後に戻したい教室などのWi-Fi


def main():
    """Cozmoへ接続し、カメラ映像（右下にバッテリー残量）を表示する。"""
    # cozmo_session が「Wi-Fi切替 → 本体接続 → 終了時の後始末」を自動で行う
    with cozmo_session(COZMO_SSID, RETURN_SSID) as cli:

        # バッテリー監視を開始（カメラ映像の右下に残量を表示）
        battery = BatteryMonitor(cli)
        battery.start()

        # カメラ映像表示を開始（バッテリー残量を重ねて表示）
        camera_viewer = CameraViewer(cli, battery_monitor=battery)
        camera_viewer.start()

        try:
            print("--- カメラ映像を表示中です (Ctrl+Cで終了) ---")
            while True:
                time.sleep(1)
        finally:
            # ループを抜けたら（Ctrl+Cなど）必ず後始末する
            camera_viewer.stop()
            battery.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n終了操作(Ctrl+C)を受け付けました。")
    except Exception as e:
        print(f"\nエラー：実行中に問題が発生しました。({e})")