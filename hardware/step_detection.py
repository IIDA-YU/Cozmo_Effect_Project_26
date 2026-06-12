import pycozmo
import time

# ロボットが移動中かどうかを管理するフラグ
is_moving = False

def on_robot_state(cli, state):
    """ロボットのセンサー情報が更新されるたびに呼ばれる関数"""
    global is_moving

    # 段差（崖）を検知した場合
    if state.cliff_detected:
        if is_moving:
            print("段差を検知しました！緊急停止します。")
            
            # 1. 即座にタイヤの回転を止める
            cli.drive_wheels(0, 0)
            is_moving = False
            
            # 2. 安全確保のため、50mm後退する (引数は 距離, 速度)
            # 後退させるために距離をマイナスにします
            print("安全のため後退します。")
            cli.drive_distance(-50, 50).wait()
            
            # 3. アニメーションを再生
            cli.play_anim("anim_fistbump_fail_01").wait()

def main():
    global is_moving
    
    with pycozmo.connect() as cli:
        cli.wait_for_robot()
        
        # センサーを床に向けやすくするため頭を下げる
        cli.set_head_angle(0.0).wait()
        
        # イベントハンドラを登録
        cli.add_handler(pycozmo.event.EvtRobotStateUpdated, on_robot_state)
        
        print("前進を開始します...")
        cli.drive_wheels(40, 40)
        is_moving = True
        
        try:
            # 走行中はプログラムが終了しないように待機
            while is_moving:
                time.sleep(0.1)
                
            print("処理を終了しました。")
            
        except KeyboardInterrupt:
            # 強制終了時 (Ctrl+C) は安全のためにタイヤを止める
            print("\nユーザーにより中断されました。")
            cli.drive_wheels(0, 0)

if __name__ == '__main__':
    main()