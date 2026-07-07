# pheromone_navi_log.py
import os
import json
import math
from datetime import datetime

class PheromoneTracker:
    """
    仮想フェロモンの記録・蒸発・ファイル保存を一元管理する汎用モジュール。
    main.py からはこのクラスを呼び出すだけで全ての機能が動作します。
    """
    
    def __init__(self, precision: int = 1, log_filename="forbidden_zones.json", evaporation_rate: float = 0.1):
        """
        :param precision: 座標を丸める小数点以下の桁数
        :param log_filename: 保存・読み込みを行うJSONファイル名
        :param evaporation_rate: 1回の蒸発処理で減少するフェロモン濃度の割合（0.1 = 10%減少）
        """
        self.precision = precision
        self.log_filename = log_filename
        self.evaporation_rate = evaporation_rate
        
        # フェロモンリスト。中身は {"x": float, "y": float, "intensity": float} の辞書
        self.pheromones = []
        
        # 起動時に自動でファイルから過去のログを読み込む
        self.load_forbidden_zones()

    def load_forbidden_zones(self):
        """ファイルから過去の仮想フェロモン（進入禁止エリア）データを読み込む"""
        if os.path.exists(self.log_filename):
            try:
                with open(self.log_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 過去のログを濃度 1.0（最大値）として復元
                    self.pheromones = [
                        {"x": item['x'], "y": item['y'], "intensity": item.get('intensity', 1.0)}
                        for item in data.get("zones", [])
                    ]
                print(f"✓ [Pheromone] 仮想フェロモンログを読み込みました ({len(self.pheromones)} 件): {self.log_filename}")
            except Exception as e:
                print(f"✗ [Pheromone] ログ読み込みエラー: {e}。新規ログとして開始します。")
                self.pheromones = []
        else:
            print(f"ℹ [Pheromone] ログファイルが見つからないため、新規にファイルを生成します: {self.log_filename}")
            self.pheromones = []

    def save_forbidden_zones(self):
        """現在のフェロモンデータをファイルに書き出して保存する"""
        data = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zones": [
                {"x": p["x"], "y": p["y"], "intensity": round(p["intensity"], 2)} 
                for p in self.pheromones
            ]
        }
        try:
            with open(self.log_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"💾 [Pheromone] 仮想フェロモンのログをファイルに保存しました: {self.log_filename}")
        except Exception as e:
            print(f"✗ [Pheromone] ログ保存エラー: {e}")

    def update(self, x: float, y: float) -> bool:
        """
        現在の座標に新しいフェロモンを設置する（直前と同じ場所なら重複防止）
        """
        pos_x = round(x, self.precision)
        pos_y = round(y, self.precision)
        
        # すでに全く同じ座標にフェロモンがあるかチェック
        for p in self.pheromones:
            if p["x"] == pos_x and p["y"] == pos_y:
                return False
                
        # 新しいフェロモンを濃度 1.0 で追加
        new_pheromone = {"x": pos_x, "y": pos_y, "intensity": 1.0}
        self.pheromones.append(new_pheromone)
        print(f"➕ [Pheromone] 新しいフェロモンを設置: ({pos_x}, {pos_y})")
        
        # 設置されたらリアルタイム自動保存
        self.save_forbidden_zones()
        return True

    def evaporate(self):
        """
        ★新規：フェロモンを蒸発させる処理★
        呼び出されるたびに全体の濃度を減少させ、強度が 0 以下になったものは消滅させます。
        """
        active_pheromones = []
        for p in self.pheromones:
            # 濃度を減少させる
            p["intensity"] -= self.evaporation_rate
            
            # まだ濃度が残っているものだけを残す
            if p["intensity"] > 0.0:
                active_pheromones.append(p)
            else:
                print(f"💨 [Pheromone] フェロモンが蒸発し消滅しました: ({p['x']}, {p['y']})")
                
        self.pheromones = active_pheromones
        # 蒸発した結果をファイルに反映
        self.save_forbidden_zones()

    def is_near_pheromone(self, x: float, y: float, threshold: float = 50.0) -> bool:
        """指定した座標の近くにフェロモン（進入禁止）があるか判定"""
        for p in self.pheromones:
            distance = math.hypot(x - p["x"], y - p["y"])
            if distance < threshold:
                return True
        return False

    def get_pheromones(self) -> list:
        """現在のフェロモンリスト（座標と濃度を含む）を取得"""
        return self.pheromones

    def clear(self):
        """フェロモンデータをリセット"""
        self.pheromones = []
        self.save_forbidden_zones()