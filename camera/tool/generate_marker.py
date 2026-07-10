#----------------------------------------------------------------------
# generate_marker.py
# Cozmo用のArUcoマーカーを生成するためのスクリプト
#----------------------------------------------------------------------

import cv2

def main():
    # 4x4マス、50種類の辞書を指定
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 400
    border_size = 50  # 白い余白のサイズ

    # 🟢 Cozmo用のマーカーIDリスト（2, 3, 4, 5）
    cozmo_ids = [1, 2, 3, 4, 5]

    for marker_id in cozmo_ids:
        # マーカーを生成して白いフチを追加
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        marker_with_border = cv2.copyMakeBorder(
            marker_img, border_size, border_size, border_size, border_size,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        # 🟢 変更：ご希望のファイル名フォーマット
        filename = f"aruco_marker_c__id{marker_id}.png"
        cv2.imwrite(filename, marker_with_border)
        print(f"✅ ID:{marker_id} のCozmo用マーカーを {filename} として保存しました。")

if __name__ == "__main__":
    main()