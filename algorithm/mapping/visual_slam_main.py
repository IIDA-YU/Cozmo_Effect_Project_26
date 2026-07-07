from visual_slam import CozmoSlamController

def main():
    # 📝 1台目（マップを作る側）の設定
    slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt", use_cached_map=False)
    
    # 📝 2台目（受信したマップを使う側）の場合は上を消して、こちらを有効にする
    # slam_bot = CozmoSlamController(model_path="learning_data/best_powercube.pt", use_cached_map=True)
    
    try:
        slam_bot.run()
    except KeyboardInterrupt:
        print("\n強制終了が検知されました。")
    
    # 終了後にマッピングされた情報を保存
    slam_bot.save_map_data(json_filename="slam_data.json", img_filename="slam_map.png")

if __name__ == "__main__":
    main()