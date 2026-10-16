"""停止画像でキューブの存在を確認する。距離・自己位置の確定とは分ける。"""
import math

from research.perception import _iou


class CubeStopLock:
    """弱い候補が残る間は停止。新しい画像だけをupdateへ渡す。

    十分な信頼度の同じ枠が2枚続けば存在確認。見失った場合もすぐには
    発進せず、所定枚数の空振りを確認する。時間切れで確定扱いにしない。
    """

    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.lost = []

    @property
    def pending(self):
        return sum(not t['confirmed'] for t in self.tracks)

    def update(self, observations):
        cfg = self.config
        prior = list(self.tracks)
        used = set()
        self.lost = []
        for obs in sorted(observations, key=lambda o: -o.get('confidence', 0)):
            box = obs.get('box')
            if (box is None or len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or min(box[2] - box[0], box[3] - box[1]) < 6
                    or obs.get('confidence', 0) < cfg.cube_candidate_confidence):
                continue
            choices = [(_iou(box, t['box']), i, t) for i, t in enumerate(prior) if i not in used]
            overlap, index, track = max(choices, key=lambda v: v[0]) if choices else (0, -1, None)
            if overlap < cfg.cube_stopped_match_iou:
                track = dict(box=box, hits=0, misses=0, confirmed=False, observation=obs)
                self.tracks.append(track)
            else:
                used.add(index)
            high = obs.get('confidence', 0) >= cfg.cube_review_confirm_confidence
            track['hits'] = track['hits'] + 1 if high else 0
            track.update(box=box, misses=0, observation=obs)
            if (obs.get('previously_confirmed', False)
                    or track['hits'] >= max(cfg.cube_confirm_frames, cfg.cube_review_confirm_frames)):
                track['confirmed'] = True
            obs['visual_confirmed'] = track['confirmed']
            obs['visual_hits'] = track['hits']
        for i, track in enumerate(prior):
            if i not in used:
                track['hits'] = 0
                track['misses'] += 1
                if track['misses'] >= cfg.cube_stop_missing_frames:
                    self.lost.append(track['observation'])
                    self.tracks.remove(track)
        return self.pending > 0
