import pycozmo
import math
import time

def normalize_angle_rad(angle_rad):
    """角度を -pi(180 度) から pi(180 度) に正規化し、境界線の計算バグを防ぐ"""
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi

def rotate_to_target_deg(cli, target_turn_deg, tolerance_deg=0.3):
    """ジャイロ誤差を見ながら、その場旋回を高精度で行う"""
    
    # 【安全装置】Cozmoの姿勢情報が取得できているか確認
    if not cli.pose:
        print("エラー: Cozmoの姿勢情報が取得できません。少し待ってから再実行してください。")
        return

    # スタート時の角度と目標角度（ラジアン）を計算
    start_rad = cli.pose.rotation.angle_z.radians
    target_rad = normalize_angle_rad(start_rad + math.radians(target_turn_deg))
    
    deadline = time.time() + 10.0 # タイムアウト設定(10 秒)
    stable_count = 0
    required_stable_count = 6 # 安定とみなす連続フレーム数

    while time.time() < deadline:
        # 万が一途中で姿勢情報が消えた場合のフェイルセーフ
        if not cli.pose:
            break
            
        current_rad = cli.pose.rotation.angle_z.radians
        error_rad = normalize_angle_rad(target_rad - current_rad)
        error_deg = math.degrees(error_rad)
        abs_error_deg = abs(error_deg)
        
        if abs_error_deg <= tolerance_deg:
            stable_count += 1
            if stable_count >= required_stable_count:
                break # 目標誤差範囲内で安定したらループを抜ける
        else:
            stable_count = 0 # 追加：誤差が広がったらカウントをリセットする

        # 誤差が小さいほど低速にし、オーバーシュートを抑える（段階的 P 制御）
        if abs_error_deg > 20:
            speed = 35
        elif abs_error_deg > 8:
            speed = 25
        elif abs_error_deg > 3:
            speed = 16
        else:
            speed = 10
            
        # 誤差の方向（プラスかマイナスか）に応じて左右のモーターを回す
        if error_deg > 0:
            cli.drive_wheels(-speed, speed) # 左旋回
        else:
            cli.drive_wheels(speed, -speed) # 右旋回

        time.sleep(0.03)

    # 最終停止処理
    cli.drive_wheels(0, 0)
    time.sleep(0.2)

def main():
    print("Cozmoに接続しています...")
    with pycozmo.connect() as cli:
        cli.wait_for_robot()
        print("ジャイロセンサーを使った精密旋回モジュール")

        time.sleep(1)
        
        # 姿勢情報取得チェック
        if not cli.pose:
            print("エラー: 姿勢情報が取得できませんでした。")
            return
            
        start_angle = cli.pose.rotation.angle_z.degrees
        print(f"旋回前の角度: {start_angle:.2f} 度")

        target_turn_deg = 90.0
        print(f"左へ {target_turn_deg} 度 旋回します...")
        
        # 自作の高精度旋回関数を呼び出す
        rotate_to_target_deg(cli, target_turn_deg, tolerance_deg=0.3)

        time.sleep(1)
        end_angle = cli.pose.rotation.angle_z.degrees
        print(f"旋回後の角度: {end_angle:.2f} 度")

        # 最終的な精度を計算
        actual_turn = math.degrees(
            normalize_angle_rad(math.radians(end_angle) - math.radians(start_angle))
        )
        error = actual_turn - target_turn_deg
        
        print(f"実際の旋回量: {actual_turn:.2f} 度")
        print(f"誤差: {error:.2f} 度")
        
        if abs(error) <= 0.1:
            print("素晴らしい精度です！ジャイロセンサーが正常に機能しています。")
            try:
                anim_action = cli.play_anim("anim_pounce_success_02")
                if anim_action is not None and hasattr(anim_action, "wait"):
                    anim_action.wait()
            except ValueError as e:
                print(f"アニメーション再生をスキップします: {e}")
        else:
            print("少し誤差が大きいです。キャタピラの滑りや、机の傾きが原因かもしれません。")

if __name__ == '__main__':
    main()