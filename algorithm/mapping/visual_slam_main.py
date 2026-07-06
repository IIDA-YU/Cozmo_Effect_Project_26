from visual_slam import CozmoSlamController

def main():
    # 1. クラスのインスタンス化
    # モデルのパスを変更する場合は引数に指定可能です
    slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt")
    
    # 2. SLAMとナビゲーションの実行 (qキーで終了するまでループします)
    try:
        slam_bot.run()
    except KeyboardInterrupt:
        print("\n強制終了が検知されました。")
    
    # 3. 終了後にマッピングされた情報を取得
    map_data = slam_bot.get_map_data()
    
    # 取得したデータをPython上で処理・確認する例
    print("\n=== 終了後のデータレポート ===")
    print(f"発見したランドマークの数: {len(map_data['landmarks'])}")
    for l_id, coords in map_data['landmarks'].items():
        print(f" - ID[{l_id}]: (X: {coords[0]:.1f}, Y: {coords[1]:.1f})")
    
    print(f"記録された軌跡ポイント数: {len(map_data['trajectory'])}")
    print("==============================\n")
    
    # 4. JSONファイルとマップ画像として保存
    slam_bot.save_map_data(json_filename="slam_data.json", img_filename="slam_map.png")

if __name__ == "__main__":
    main()