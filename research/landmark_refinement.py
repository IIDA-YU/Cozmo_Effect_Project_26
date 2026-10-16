"""停止中のチェス座標を集め、安定した平均だけをまとめて採用する。

このクラスへ渡す座標は、対象駒を使わずに求めた自己位置で投影する。
毎フレーム地図を動かさず、外れ値除去・最低標本数・ばらつき・更新間隔を
すべて満たすまで現在の確定座標を保持する。単位はmm、秒。
"""
import math
from statistics import median


class LandmarkRefiner:
    def __init__(self, landmarks, config):
        self.original = {str(k): tuple(p) for k, p in landmarks.items()}
        self.config = config
        self.samples = {}
        self.last_update = {}
        self.history = []

    def offer(self, key, point, current, now, anchors):
        """独立した新画像1枚分を追加し、採用可能なら更新案を返す。"""
        cfg = self.config
        state = dict(id=key, samples=0, status='COLLECTING')
        if (not all(math.isfinite(v) for v in (*point, now))
                or math.dist(point, self.original[key]) > cfg.chess_refine_max_shift_mm):
            state['status'] = 'SHIFT TOO LARGE'
            return None, state

        samples = [s for s in self.samples.get(key, [])
                   if 0 <= now - s['time'] <= cfg.chess_refine_window_seconds]
        if not samples or now - samples[-1]['time'] >= cfg.chess_refine_sample_interval_seconds:
            samples.append(dict(point=tuple(point), time=now, anchors=sorted(anchors)))
        samples = samples[-cfg.chess_refine_window_frames:]
        self.samples[key] = samples
        center = tuple(median(s['point'][axis] for s in samples) for axis in (0, 1))
        good = [s for s in samples if math.dist(s['point'], center) <= cfg.chess_refine_outlier_mm]
        state['samples'] = len(good)
        if not good:
            state['status'] = 'UNSTABLE'
            return None, state
        mean = tuple(sum(s['point'][axis] for s in good) / len(good) for axis in (0, 1))
        spread = math.sqrt(sum(math.dist(s['point'], mean)**2 for s in good) / len(good))
        shift = math.dist(mean, current)
        state.update(mean=list(mean), spread_mm=spread, shift_mm=shift)
        if shift < cfg.chess_refine_min_shift_mm:
            state['status'] = 'STABLE'
            return None, state
        if len(good) < cfg.chess_refine_min_samples:
            return None, state
        if (len(good) < math.ceil(len(samples) * 0.75)
                or spread > cfg.chess_refine_spread_mm):
            state['status'] = 'UNSTABLE'
            return None, state
        if good[-1]['time'] - good[0]['time'] < cfg.chess_refine_min_span_seconds:
            return None, state
        if now - self.last_update.get(key, -math.inf) < cfg.chess_refine_cooldown_seconds:
            state['status'] = 'COOLDOWN'
            return None, state
        state['status'] = 'READY'
        proposal = dict(id=key, before=list(current), after=list(mean), time=now,
                        samples=len(good), spread_mm=spread, shift_mm=shift,
                        anchor_ids=sorted({a for s in good for a in s['anchors']}))
        return proposal, state

    def commit(self, updates):
        """確定後は標本を撮り直す。旧座標を基準にした別駒の標本も混ぜない。"""
        if not updates:
            return
        self.samples.clear()
        for update in updates:
            self.last_update[update['id']] = update['time']
            self.history.append(dict(update))
