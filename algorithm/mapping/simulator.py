import cv2
import numpy as np
import random
import math

# --- シミュレータの設定 ---
WIDTH, HEIGHT = 800, 500
NUM_ROBOTS = 15      # ロボットの台数
SPEED = 5.0          # 基本スピード

# 仮想フェロモンマップ（0.0 〜 255.0 の数値で管理）
pheromone_map = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# 障害物（倉庫の棚など）： (x, y, width, height)
obstacles = [
    (300, 100, 50, 150),
    (300, 350, 50, 150),
    (500, 200, 50, 200)
]

class VirtualCozmo:
    def __init__(self):
        self.reset_position()

    def reset_position(self):
        # 左端のランダムな位置からスタートし、右端をゴールとする
        self.x = random.uniform(20, 100)
        self.y = random.uniform(20, HEIGHT - 20)
        self.target_x = WIDTH - 50
        self.target_y = random.uniform(50, HEIGHT - 50)
    
    def update(self, phero_map):
        # 1. ゴールへの方向を計算
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)

        # ゴールに到着したらスタート地点に戻る
        if dist < 20:
            self.reset_position()
            return

        vx = (dx / dist) * SPEED
        vy = (dy / dist) * SPEED

        # 2. 仮想フェロモン（渋滞）の回避ロジック
        # 少し先の座標を確認し、フェロモンが濃い（他の機体が密集している）場合は経路を逸らす
        ahead_x = int(self.x + vx * 10)
        ahead_y = int(self.y + vy * 10)

        if 0 <= ahead_x < WIDTH and 0 <= ahead_y < HEIGHT:
            congestion = phero_map[ahead_y, ahead_x]
            if congestion > 100:  # 渋滞の閾値
                # 混雑を避けるため、進行方向を少し直角にずらす
                vx += vy * 1.5
                vy -= vx * 1.5

        # 3. 位置の更新と画面端の制限
        self.x = max(0, min(WIDTH - 1, self.x + vx))
        self.y = max(0, min(HEIGHT - 1, self.y + vy))

        # 4. フェロモンの分泌（自分の現在地にフェロモンを落とす）
        cv2.circle(phero_map, (int(self.x), int(self.y)), 12, 30.0, -1)

def main():
    global pheromone_map
    robots = [VirtualCozmo() for _ in range(NUM_ROBOTS)]

    print("シミュレータを起動しました。[Q]キーで終了します。")

    while True:
        # A. フェロモンの蒸発（毎フレーム少しずつ薄くする）
        pheromone_map *= 0.96 

        # B. 描画用のキャンバスを作成
        # フェロモンマップをシアン色（青＋緑）に変換して背景にする
        phero_disp = np.clip(pheromone_map, 0, 255).astype(np.uint8)
        canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        canvas[:, :, 0] = phero_disp  # Blue
        canvas[:, :, 1] = phero_disp  # Green

        # C. 障害物（棚）の描画
        for (ox, oy, ow, oh) in obstacles:
            cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (150, 150, 150), -1)

        # D. ロボットの更新と描画
        for r in robots:
            r.update(pheromone_map)
            # ロボット本体を赤い丸で描画
            cv2.circle(canvas, (int(r.x), int(r.y)), 6, (0, 0, 255), -1)

        # E. 画面表示
        cv2.imshow("COZMO Swarm Intelligence Simulator", canvas)

        # Qキーで終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    