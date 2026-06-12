import pycozmo
import time
import cv2

from hardware.map import MapCalculator, CozmoExplorer
from hardware.shortest_path import go_to_goal_simple
from hardware.perfect_turnning import rotate_to_target_deg
from camera.aruco_tracker import ArucoCoordinateTracker

def run_test_sequence(cli, sender):
    """
    ステップ1〜4を実行するテストシナリオ
    """
    # # --- ステップ1: 精密旋回 ---
    # print("\n[ステップ1] 精密旋回テストを実行...")
    # rotate_to_target_deg(cli, 90.0, tolerance_deg=0.3)
    # time.sleep(1)
    
    # # --- ステップ2: マッピング ---
    # print("\n[ステップ2] 自律マッピングモードを開始...")
    # explorer = CozmoExplorer(cli)
    # explorer.run_4_sides_mapping()
    # if explorer.pheromones:
    #     MapCalculator.calculate_size(explorer.pheromones, data_type="軌跡")
    # if explorer.corners:
    #     MapCalculator.calculate_size(explorer.corners, data_type="角座標")
    # time.sleep(1)
    
    # # --- ステップ3: ゴール移動 ---
    # print("\n[ステップ3] ゴール移動を実行...")
    # go_to_goal_simple(cli)
    # time.sleep(1)
    
    # --- 🟢 ステップ4: ArUcoマーカー検知 ＆ 特定相手への座標送信 ---
    print("\n[ステップ4] 特定のCozmoへのArUco検知・座標送信テストを開始... (Ctrl+Cで終了)")
    cli.enable_camera(enable=True, color=True)
    
    # 📌 【テスト設定】複数台のCozmoを識別するための対応表
    # 実際にはここに他のメンバーのPCのIPアドレスを入れます
    COZMO_IP_DICT = {
        1: "10.18.81.102",
        2: "172.18.67.198",  
        3: "127.0.0.1",  
        4: "127.0.0.1",
        5: "127.0.0.1",
    }
    
    # 辞書を渡してインスタンスを生成
    tracker = ArucoCoordinateTracker(cli, target_ip_dict=COZMO_IP_DICT)
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, tracker.on_camera_image)

    # 崖検知用の安全フラグ
    is_safe_to_drive = [True]

    # 引数の順番や数がどれだけ変わっても、自動で必要な崖データを探し出す関数
    def check_cliff(*args, **kwargs):
        for arg in args:
            if hasattr(arg, 'cliff_detected') and arg.cliff_detected:
                if is_safe_to_drive[0]:
                    print("\n⚠️ 崖（段差）を検知しました！緊急停止し、後退します。")
                    is_safe_to_drive[0] = False
                    cli.drive_wheels(0, 0)
                    if hasattr(cli.drive_distance(-50, 50), "wait"):
                        cli.drive_distance(-50, 50).wait()
                break

    # エラーにならない「安全なイベント名」をプログラム自身に見つけさせる
    cliff_event = getattr(pycozmo.event, 'EvtRobotStateUpdated', None)
    if not cliff_event:
        cliff_event = getattr(pycozmo.event, 'EvtRobotState', None)
        
    if cliff_event:
        cli.add_handler(cliff_event, check_cliff)
    else:
        print("⚠️ 警告: 現在のpycozmoバージョンでは崖検知イベントが見つかりません。")

    print("🤖 前進しながら探索を開始します...")
    
    try:
        while not getattr(sender, 'shutdown_requested', False):
            if is_safe_to_drive[0]:
                cli.drive_wheels(30, 30)
            time.sleep(0.2)
            
        print("\n👋 受信側からの遠隔指令により、前進ループを終了します。")

    except KeyboardInterrupt:
        print("\nカメラモード終了")
        
    finally:
        cli.drive_wheels(0, 0)
        
        tracker.running = False           
        cli.enable_camera(enable=False)   
        
        # 登録したのと同じイベント名を使って安全に解除する
        if cliff_event:
            if hasattr(cli, 'del_handler'):
                cli.del_handler(cliff_event, check_cliff)
            elif hasattr(cli, 'remove_handler'):
                cli.remove_handler(cliff_event, check_cliff)
        
        sender.stop()
        cv2.destroyAllWindows()