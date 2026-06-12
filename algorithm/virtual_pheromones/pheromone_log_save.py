# pheromone_log_save.py
import os
import json
import math
from datetime import datetime

class ForbiddenZoneManager:
    """仮想フェロモン（進入禁止エリア）のログファイルを管理するクラス"""
    
    def __init__(self, log_filename="forbidden_zones.json"):
        self.log_filename = log_filename
        self.forbidden_zones = []
        # 起動時に自動で既存のログファイルを読み込む
        self.load_forbidden_zones()

    def load_forbidden_zones(self):
        """ファイルから通行禁止エリア（仮想フェロモン）のデータを読み込む"""
        if os.path.exists(self.log_filename):
            try:
                with open(self.log_filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSONの配列から座標 (x, y) のタプルリストを復元
                    self.forbidden_zones = [(item['x'], item['y']) for item in data.get("zones", [])]
                print(f"✓ [Module] 仮想フェロモンログを読み込みました ({len(self.forbidden_zones)} 件): {self.log_filename}")
            except Exception as e:
                print(f"✗ [Module] ログ読み込みエラー: {e}。新規ログとして開始します。")
                self.forbidden_zones = []
        else:
            print(f"ℹ [Module] ログファイルが見つからないため、新規にファイルを生成します: {self.log_filename}")
            self.forbidden_zones = []

    def save_forbidden_zones(self):
        """現在の通行禁止エリアをファイルに書き出して保存する"""
        data = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "zones": [{"x": p[0], "y": p[1]} for p in self.forbidden_zones]
        }
        try:
            with open(self.log_filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"💾 [Module] 仮想フェロモンのログをファイルに保存しました: {self.log_filename}")
        except Exception as e:
            print(f"✗ [Module] ログ保存エラー: {e}")

    def add_forbidden_zone(self, x, y):
        """新しい通行禁止座標（仮想フェロモン）を動的に追加し、自動保存する"""
        if (x, y) not in self.forbidden_zones:
            self.forbidden_zones.append((x, y))
            print(f"➕ [Module] 新しい仮想フェロモンを追加: ({x:.1f}, {y:.1f})")
            self.save_forbidden_zones()  # 追加されたら自動保存

    def is_inside_forbidden_zone(self, x, y, threshold=50.0):
        """現在の座標が通行禁止エリア（仮想フェロモン）に近いか判定"""
        for fx, fy in self.forbidden_zones:
            distance = math.hypot(x - fx, y - fy)  # ユークリッド距離
            if distance < threshold:
                return True
        return False