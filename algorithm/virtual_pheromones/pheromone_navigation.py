"""
フェロモン・ナビゲーション・モジュール (pheromone_navigation.py)
特定のハードウェア（Cozmo等）に依存せず、座標データのみでフェロモン（軌跡）を管理・活用する汎用モジュールです。
"""

import math


class PheromoneTracker:
    def __init__(self, precision: int = 1):
        """
        フェロモントラッカーの初期化
        :param precision: 座標を丸める小数点以下の桁数（1なら0.1単位、0なら1単位で判定）
        """
        self.precision = precision
        self.pheromones = []

    def update(self, x: float, y: float) -> bool:
        """
        現在の座標を元に、新しい場所であればフェロモンを記録する
        :param x: 現在のX座標
        :param y: 現在のY座標
        :return: 新しく追加された場合は True、重複などで追加されなかった場合は False
        """
        # 座標の四捨五入（微小なブレで別座標と判定されるのを防ぐ）
        pos = (
            round(x, self.precision), 
            round(y, self.precision)
        )
        
        # 配列が空、または直前の座標と異なる場合のみ新しいフェロモンとして記録
        if not self.pheromones or self.pheromones[-1] != pos:
            self.pheromones.append(pos)
            return True
        
        return False

    def is_near_pheromone(self, x: float, y: float, threshold: float = 0.5) -> bool:
        """
        指定した座標の近くに、過去に置いたフェロモン（既知のルート）があるか判定する
        （※元のコードにはありませんでしたが、フェロモンを『回避』または『追従』するために必須のアルゴリズムです）
        :param x: 判定したいX座標
        :param y: 判定したいY座標
        :param threshold: 近くにあると判定する距離の閾値
        """
        for p_x, p_y in self.pheromones:
            distance = math.sqrt((x - p_x) ** 2 + (y - p_y) ** 2)
            if distance < threshold:
                return True
        return False

    def get_pheromones(self) -> list:
        """全軌跡（フェロモン）データを取得"""
        return self.pheromones

    def get_count(self) -> int:
        """フェロモンの総数を取得"""
        return len(self.pheromones)

    def clear(self):
        """フェロモンデータをリセット"""
        self.pheromones = []
