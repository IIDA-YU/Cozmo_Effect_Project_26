"""未確認チェスを見る位置を選ぶ。移動経路とカメラの見通しを別々に確認する。"""
import math

from research.world import segment_distance, wrap


class LandmarkViewpoints:
    def __init__(self):
        self.attempts = {}

    def record(self, landmark_id, position):
        history = self.attempts.setdefault(str(landmark_id), [])
        history.append(tuple(position[:2]))
        self.attempts[str(landmark_id)] = history[-64:]

    @staticmethod
    def clear_view(position, landmark_id, planner):
        """走行用に膨らませた障害物ではなく実物の大きさで遮蔽を調べる。"""
        cfg, world = planner.config, planner.world
        landmark = world.landmarks[landmark_id]
        heading = math.atan2(landmark[1] - position[1], landmark[0] - position[0])
        camera = (position[0] + cfg.camera_forward_mm * math.cos(heading),
                  position[1] + cfg.camera_forward_mm * math.sin(heading))
        radius = math.sqrt(2) * cfg.cube_width_mm / 2 + cfg.landmark_occlusion_margin_mm
        blockers = [(c['x'], c['y'], radius) for c in world.cubes()]
        blockers += [(x, y, radius) for x, y, _ in planner.transient_obstacles]
        blockers += [(p[0], p[1], 20.0) for key, p in world.landmarks.items()
                     if str(key) != str(landmark_id)]
        return not any(segment_distance((x, y), camera, landmark) <= r for x, y, r in blockers)

    def choose(self, pose, unseen, planner, reachable, start):
        cfg, world = planner.config, planner.world
        grid = world.grid
        choices = []
        for landmark_id in sorted(unseen):
            landmark = world.landmarks.get(landmark_id)
            if landmark is None:
                continue
            history = self.attempts.get(landmark_id, [])
            for key in sorted(reachable):
                position = grid.cells[key]
                distance = math.dist(position, landmark) - cfg.camera_forward_mm
                if not cfg.landmark_view_min_mm <= distance <= cfg.landmark_view_max_mm:
                    continue
                heading = math.atan2(position[1] - landmark[1], position[0] - landmark[0])
                # 前の観測と同じ視線へ戻らない。左右へ回り込む候補を優先する。
                repeated = sum(
                    math.dist(position, previous) < cfg.landmark_vantage_separation_mm
                    or abs(wrap(heading - math.atan2(previous[1] - landmark[1], previous[0] - landmark[0]))) < math.radians(12)
                    for previous in history
                )
                clear = self.clear_view(position, landmark_id, planner)
                score = (240 * repeated + 50 * len(history)
                         + 0.35 * math.dist(pose[:2], position)
                         + 0.45 * abs(distance - cfg.landmark_vantage_distance_mm)
                         + 60 * world.pheromone_at(key))
                choices.append((not clear, score, landmark_id, key))
        if not choices:
            return None
        # 見通しのある候補を常に優先。遮蔽しかない場合も異なる位置で再観測する。
        clear_choices = [c for c in choices if not c[0]]
        finalists = sorted(clear_choices or choices)[:16]
        best = None
        for blocked, score, landmark_id, key in finalists:
            route = planner._weighted_route(reachable, start, key)
            if not route:
                continue
            route_cost = sum(1 + cfg.pheromone_route_weight * world.pheromone_at(k)
                             + cfg.trail_route_weight * (k in world.all_trail_keys()) for k in route)
            total = score + 0.15 * grid.cell * route_cost
            if best is None or total < best[0]:
                best = total, landmark_id, key, route, not blocked
        return best
