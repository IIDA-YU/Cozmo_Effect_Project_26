"""仮想フェロモンと実走行カバー率を使った分散型マップ探索。"""
import heapq
import math

from research.world import wrap
from research.landmark_search import LandmarkViewpoints


class Explorer:
    """研究用の探索経路計画器。

    探索方針:
      1. カメラでまだ見ていない領域（frontier）を優先する。
      2. 自機・共有済み他機の軌跡から遠い地点を優先する。
      3. 仮想フェロモンが濃いセルを経路コストとして避ける。
      4. 6駒再確認と視認率95%を両方満たすまで探索する。

    このため「中央で向きを変えるだけで95%」にならず、地図全体へ物理的に
    移動する。full共有時は他機の軌跡/フェロモンも評価に入るため、複数台で
    同じ場所を重複探索しにくい。
    """

    def __init__(self, world, config):
        self.world, self.config = world, config
        self.path = []
        self.heading = None
        self.goal = None
        self.predicted = set()
        self.motion_target = None
        self.motion_heading = None
        self.transient_obstacles = []
        self.active_verify_point = None
        self.mode = 'coverage'
        self.inspection_visited = set()
        self.inspection_target_key = None
        self.inspection_scan_queue = []
        self.inspection_pass = 0
        self.return_path = []
        self.return_pose = None

        # v7: 6駒確認用の周回探索状態。中央だけ全周旋回し、
        # それ以外では各駒の内側観測点へ移動して対象方向だけを見る。
        self.landmark_target_id = None
        self.landmark_waiting_observation = False
        self.landmark_attempted = set()
        self.landmark_views = LandmarkViewpoints()
        self.landmark_view_clear = None
        self.center_scan_queue = []
        self.center_scan_started = False
        self.center_scan_done = False

    # ------------------------------------------------------------------
    # 障害物
    # ------------------------------------------------------------------
    def _cube_safe_radius(self):
        cfg = self.config
        robot_radius = math.hypot(cfg.robot_length_mm / 2, cfg.robot_width_mm / 2)
        cube_radius = math.hypot(cfg.cube_width_mm / 2, cfg.cube_width_mm / 2)
        return robot_radius + cube_radius + cfg.cube_safety_margin_mm

    def set_transient_obstacles(self, detections):
        if not getattr(self.config, 'transient_cube_obstacle_enabled', True):
            self.transient_obstacles = []
            return
        safe_radius = self._cube_safe_radius()
        merged = []
        for detection in detections:
            x, y = float(detection[0]), float(detection[1])
            if not all(math.isfinite(value) for value in (x, y)):
                continue
            if any(math.hypot(x - ox, y - oy) <= self.config.cube_merge_mm
                   for ox, oy, _ in merged):
                continue
            merged.append((x, y, safe_radius))
        self.transient_obstacles = merged

    def obstacles(self):
        radius = self._cube_safe_radius()
        obstacles = [(cube['x'], cube['y'], radius) for cube in self.world.cubes()]
        for x, y, r in self.transient_obstacles:
            # 確認対象の候補も回避対象に残す。確認のために衝突余白を解除しない。
            if any(math.hypot(x - ox, y - oy) <= self.config.cube_merge_mm
                   for ox, oy, _ in obstacles):
                continue
            obstacles.append((x, y, r))
        # 基準境界より内側へ補正された駒にも機体を接触させない。
        chess_radius = math.hypot(self.config.robot_length_mm / 2, self.config.robot_width_mm / 2) + 20.0
        for key, point in self.world.landmarks.items():
            if math.dist(point, self.world.reference_landmarks[key]) >= self.config.chess_refine_min_shift_mm:
                obstacles.append((*point, chess_radius))
        return obstacles

    def occluders(self):
        known = [(cube['x'], cube['y'], self.config.cube_width_mm * 0.8)
                 for cube in self.world.cubes()]
        temporary = [(x, y, self.config.cube_width_mm * 0.8)
                     for x, y, _ in self.transient_obstacles]
        chess = [(point[0], point[1], 20)
                 for point in self.world.landmarks.values()]
        return known + temporary + chess

    # ------------------------------------------------------------------
    # 共通状態
    # ------------------------------------------------------------------
    def _map_center(self):
        polygon = self.world.grid.polygon
        return (sum(p[0] for p in polygon) / len(polygon),
                sum(p[1] for p in polygon) / len(polygon))

    def _near_center(self, point):
        return math.dist(point, self._map_center()) <= self.config.center_scan_radius_mm

    def _inward_heading(self, point):
        center = self._map_center()
        return math.atan2(center[1] - point[1], center[0] - point[0])

    def _landmark_vantage(self, landmark):
        """境界上の駒から地図中央方向へ下がった安全な観測点を返す。"""
        center = self._map_center()
        dx, dy = center[0] - landmark[0], center[1] - landmark[1]
        length = max(1.0, math.hypot(dx, dy))
        distance = self.config.landmark_vantage_distance_mm
        return (landmark[0] + dx / length * distance,
                landmark[1] + dy / length * distance)

    def _coverage_ratio(self):
        total = len(self.world.grid.cells)
        return 0.0 if total == 0 else len(self.world.covered) / total

    def _trail_coverage_ratio(self):
        return self.world.trail_coverage_ratio(self.config.trail_coverage_radius_mm)

    def _missing_expected_cubes(self):
        expected = int(getattr(self.config, 'expected_cubes', 0))
        return expected > 0 and len(self.world.cubes()) < expected

    def _turn_command(self, delta, current_heading):
        self.motion_heading = wrap(current_heading + delta)
        self.motion_target = None
        speed = math.copysign(self.config.turn_speed_mm_s, delta)
        return -speed, speed

    def can_advance(self, pose, distance):
        grid = self.world.grid
        end = (pose[0] + distance * math.cos(pose[2]),
               pose[1] + distance * math.sin(pose[2]))
        check = grid.free_segment
        if not grid.inside(pose[:2], grid.margin) or self.mode == 'return':
            check = grid.free_map_segment
        return check(pose[:2], end, self.obstacles())

    def _smooth_path(self, position, points):
        """安全な直線でショートカットし、細かな蛇行を減らす。"""
        grid = self.world.grid
        obstacles = self.obstacles()
        result = []
        remaining = list(points)
        while remaining:
            check = grid.free_segment if grid.inside(position, grid.margin) else grid.free_entry_segment
            index = len(remaining) - 1
            while index > 0 and not check(position, remaining[index], obstacles):
                index -= 1
            position = remaining[index]
            result.append(position)
            remaining = remaining[index + 1:]
        return result

    # ------------------------------------------------------------------
    # フェロモンを使った経路計画
    # ------------------------------------------------------------------
    def _weighted_route(self, reachable, start, target):
        """最短距離だけでなくフェロモン/既走行セルを避けるDijkstra経路。"""
        if target == start:
            return [start]
        allowed = set(reachable)
        trail_keys = self.world.all_trail_keys()
        obstacles = self.obstacles()
        grid = self.world.grid
        pq = [(0.0, start)]
        cost = {start: 0.0}
        previous = {start: None}
        while pq:
            current_cost, key = heapq.heappop(pq)
            if current_cost != cost.get(key):
                continue
            if key == target:
                break
            x, y = map(int, key.split(','))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = f'{x+dx},{y+dy}'
                if nxt not in allowed:
                    continue
                if not grid.free_segment(grid.cells[key], grid.cells[nxt], obstacles):
                    continue
                extra = (
                    self.config.pheromone_route_weight * self.world.pheromone_at(nxt)
                    + self.config.trail_route_weight * (1.0 if nxt in trail_keys else 0.0)
                )
                new_cost = current_cost + 1.0 + extra
                if new_cost < cost.get(nxt, float('inf')):
                    cost[nxt] = new_cost
                    previous[nxt] = key
                    heapq.heappush(pq, (new_cost, nxt))
        if target not in previous:
            return []
        return self.world.grid.route(previous, target)

    def _exploration_keys(self, reachable_keys):
        """マップ全体へ均等に散らした観測候補点を作る。"""
        grid = self.world.grid
        spacing = max(grid.cell, float(self.config.exploration_waypoint_spacing_mm))
        step = max(1, int(round(spacing / grid.cell)))
        keys = []
        for key in reachable_keys:
            ix, iy = map(int, key.split(','))
            if ix % step == 0 and iy % step == 0:
                keys.append(key)
        if len(keys) < 4:
            keys = sorted(reachable_keys)
        return keys

    def _best_view(self, point, pose, unknown, occluders):
        """候補地点で未観測領域を多く見られる向きを選ぶ。

        地図中央付近では360度候補を許可するが、淵付近では地図中央方向から
        大きく外れる向きを候補から外す。これにより、境界で地図外を向いて
        何度も旋回する挙動を防ぐ。
        """
        cfg = self.config
        grid = self.world.grid
        if not unknown:
            return self._inward_heading(point), set()

        inward = self._inward_heading(point)
        near_edge = not grid.inside(point, grid.margin + cfg.edge_scan_guard_mm)
        headings = [i * math.tau / 8 for i in range(8)]
        if near_edge and not self._near_center(point):
            half = math.radians(cfg.edge_inward_half_angle_degrees)
            filtered = [h for h in headings if abs(wrap(h - inward)) <= half]
            headings = filtered or [inward]

        best = None
        for theta in headings:
            cx = point[0] + cfg.camera_forward_mm * math.cos(theta)
            cy = point[1] + cfg.camera_forward_mm * math.sin(theta)
            visible = grid.visible(
                (cx, cy, theta), cfg.view_near_mm, cfg.view_far_mm,
                math.radians(cfg.view_fov_degrees), occluders,
            ) & unknown
            turn = abs(wrap(theta - pose[2])) / math.pi
            inward_penalty = 0.0
            if near_edge:
                inward_penalty = abs(wrap(theta - inward)) / math.pi
            candidate = (len(visible) - 0.15 * turn - 0.35 * inward_penalty, theta, visible)
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best[1], best[2]

    def _choose_coverage(self, pose, reachable, start):
        """未観測領域 + 低フェロモン + 未走行領域を統合して次地点を決める。"""
        world, cfg = self.world, self.config
        grid = world.grid
        unknown = set(grid.cells) - world.covered
        candidates = self._exploration_keys(reachable.keys())
        if not candidates:
            return False

        trail_ratio = self._trail_coverage_ratio()
        need_roaming = trail_ratio < cfg.return_home_trail_ratio
        min_move = float(cfg.exploration_min_move_mm)

        # 実走行率が不足している間は、同じ場所で方向だけ変更する候補を除外する。
        if need_roaming:
            far = [key for key in candidates if math.dist(pose[:2], grid.cells[key]) >= min_move]
            if far:
                candidates = far

        best = None
        occluders = self.occluders()
        map_scale = max(cfg.exploration_waypoint_spacing_mm, grid.cell)
        for key in candidates:
            point = grid.cells[key]
            route = self._weighted_route(reachable, start, key)
            if not route:
                continue
            route_points = [grid.cells[k] for k in route]
            travel_mm = max(0, len(route_points) - 1) * grid.cell + math.dist(pose[:2], grid.cells[start])
            heading, visible = self._best_view(point, pose, unknown, occluders)

            trail_distance = world.trail_distance(point)
            if not math.isfinite(trail_distance):
                trail_distance = map_scale * 2
            novelty = min(trail_distance / map_scale, 1.5)
            low_pheromone = 1.0 - world.pheromone_at(key)
            movement = min(math.dist(pose[:2], point) / max(min_move, 1.0), 1.5)
            frontier = min(len(visible) / 8.0, 2.0)
            route_pheromone = sum(world.pheromone_at(k) for k in route) / max(1, len(route))
            route_revisit = sum(1 for k in route if k in world.all_trail_keys()) / max(1, len(route))
            turn = abs(wrap(heading - pose[2])) / math.pi
            travel_penalty = travel_mm / (map_scale * 4.0)

            # 「見える量」だけでなく「まだ走っていない場所」を強く評価する。
            # 共有モードでは他機のtrail/pheromoneもworldへ統合される。
            score = (
                3.2 * frontier
                + 4.0 * novelty
                + 2.0 * low_pheromone
                + 1.4 * movement
                - 1.8 * route_pheromone
                - 1.2 * route_revisit
                - 0.55 * travel_penalty
                - 0.35 * turn
            )
            if best is None or score > best[0]:
                best = score, route, heading, visible, point

        if best is None:
            return False

        _, route, self.heading, self.predicted, point = best
        points = [grid.cells[key] for key in route]
        self.path = self._smooth_path(pose[:2], points)
        self.goal = point
        self.mode = 'coverage'
        self.inspection_target_key = None
        self.inspection_scan_queue = []
        return True

    def _begin_or_continue_inspection_scan(self, pose):
        """旧巡回確認との互換用。必要時だけ視野が重なる角度で周囲を確認する。"""
        if self.inspection_target_key is None:
            return 0, 0
        if not self.inspection_scan_queue:
            degrees = min(self.config.inspection_scan_step_degrees,
                          self.config.view_fov_degrees * 0.8)
            count = max(2, int(math.ceil(360 / degrees)))
            step = math.tau / count
            self.inspection_scan_queue = [wrap(pose[2] + step * i) for i in range(1, count)]
        while self.inspection_scan_queue:
            heading = self.inspection_scan_queue[0]
            delta = wrap(heading - pose[2])
            if abs(delta) > math.radians(self.config.turn_tolerance_degrees):
                return self._turn_command(delta, pose[2])
            self.inspection_scan_queue.pop(0)
        self.inspection_visited.add(self.inspection_target_key)
        self.inspection_target_key = None
        return 0, 0

    def _follow_landmark_path(self, pose):
        """landmark_searchで設定済みの経路を1ステップ進める。"""
        cfg, grid = self.config, self.world.grid
        while self.path and math.dist(pose[:2], self.path[0]) < 12:
            self.path.pop(0)
        if self.path:
            target = self.path[0]
            obstacles = self.obstacles()
            check = grid.free_segment if grid.inside(pose[:2], grid.margin) else grid.free_entry_segment
            if not check(pose[:2], target, obstacles):
                self.path, self.goal = [], None
                return 0, 0
            heading = math.atan2(target[1] - pose[1], target[0] - pose[0])
            delta = wrap(heading - pose[2])
            if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
                return self._turn_command(delta, pose[2])
            self.motion_target = target
            speed = min(cfg.drive_speed_mm_s, max(15.0, math.dist(pose[:2], target) * 1.6))
            return speed, speed
        return None

    def landmark_command(self, pose, unseen_ids):
        """未確認チェス駒を優先して周回する。

        中央付近だけは360度を分割してその場確認する。中央スキャン後、または
        中央以外では、未確認駒への見通しがある安全な観測点へ移動する。見えなければ
        同じ視線を避けて別方向へ回り込み、観測を繰り返す。
        """
        cfg, grid = self.config, self.world.grid
        unseen = {str(v) for v in unseen_ids}
        self.motion_target = self.motion_heading = None
        if not unseen:
            self.cancel_landmark_route()
            if self.mode == 'landmark_search':
                self.mode = 'coverage'
                self.path, self.goal = [], None
            return None

        # 中央だけは全周確認を許可。1角度ごとに必ず停止観測が入る。
        if self._near_center(pose[:2]) and not self.center_scan_done:
            self.mode = 'landmark_search'
            self.path, self.goal = [], None
            if not self.center_scan_started:
                step = math.radians(max(20.0, cfg.center_scan_step_degrees))
                count = max(4, int(round(math.tau / step)))
                self.center_scan_queue = [wrap(pose[2] + step * i) for i in range(1, count + 1)]
                self.center_scan_started = True
            if self.center_scan_queue:
                heading = self.center_scan_queue[0]
                delta = wrap(heading - pose[2])
                if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
                    return self._turn_command(delta, pose[2])
                self.center_scan_queue.pop(0)
                return 0, 0
            self.center_scan_done = True

        # 前回「駒を向いて停止」した次の呼び出しは、観測済み。
        if self.landmark_waiting_observation:
            if self.landmark_target_id in unseen:
                self.landmark_attempted.add(self.landmark_target_id)
                self.landmark_views.record(self.landmark_target_id, pose)
            self.landmark_waiting_observation = False
            self.landmark_target_id = None
            self.path, self.goal = [], None

        if self.landmark_target_id not in unseen or self.goal is None or self.mode != 'landmark_search':
            self.landmark_target_id = None
            self.path, self.goal = [], None

        reachable, start = grid.reachable(pose[:2], self.obstacles())
        if start is None:
            return None

        if self.landmark_target_id is None:
            selected = self.landmark_views.choose(pose, unseen, self, reachable, start)
            if selected is None:
                return None
            _, key, target_key, route, self.landmark_view_clear = selected
            self.landmark_target_id = key
            self.mode = 'landmark_search'
            points = [grid.cells[k] for k in route]
            self.path = self._smooth_path(pose[:2], points)
            self.goal = grid.cells[target_key]

        moving = self._follow_landmark_path(pose)
        if moving is not None:
            return moving

        landmark = self.world.landmarks[self.landmark_target_id]
        heading = math.atan2(landmark[1] - pose[1], landmark[0] - pose[0])
        delta = wrap(heading - pose[2])
        if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
            return self._turn_command(delta, pose[2])

        # この向きで次ループのobserve_stoppedに入り、駒が見えるか確認する。
        self.landmark_waiting_observation = True
        return 0, 0

    def cancel_landmark_route(self):
        """キューブ確認や自己位置補正で中断された経路を引き継がない。"""
        self.landmark_target_id = None
        self.landmark_waiting_observation = False
        if self.mode == 'landmark_search':
            self.path, self.goal, self.heading = [], None, None
            self.mode = 'coverage'

    def recovery_command(self, pose):
        """安全な経路が見つからないとき、淵では中央方向だけを向く。"""
        heading = self._inward_heading(pose[:2])
        delta = wrap(heading - pose[2])
        if abs(delta) > math.radians(self.config.turn_tolerance_degrees):
            return self._turn_command(delta, pose[2])
        distance = min(self.config.exploration_min_move_mm, 70.0)
        if self.can_advance(pose, distance):
            self.motion_target = (pose[0] + distance * math.cos(pose[2]),
                                  pose[1] + distance * math.sin(pose[2]))
            return self.config.drive_speed_mm_s * 0.7, self.config.drive_speed_mm_s * 0.7
        return 0, 0

    def choose(self, pose):
        grid = self.world.grid
        reachable, start = grid.reachable(pose[:2], self.obstacles())
        if start is None:
            return False
        return self._choose_coverage(pose, reachable, start)

    def command(self, pose):
        """探索中の左右車輪速度を返す。"""
        cfg, grid = self.config, self.world.grid
        self.motion_target = self.motion_heading = None

        # 目標地点に到着して観測まで済んだら、次の低フェロモン地点を選び直す。
        if (self.goal is not None and self.predicted
                and not (self.predicted - self.world.covered)
                and math.dist(pose[:2], self.goal) <= cfg.inspection_reached_mm):
            self.path, self.goal = [], None

        if self.goal is None and not self.choose(pose):
            return None

        while self.path and math.dist(pose[:2], self.path[0]) < 12:
            self.path.pop(0)

        if self.path:
            target = self.path[0]
            obstacles = self.obstacles()
            if grid.inside(pose[:2], grid.margin):
                segment_is_free = grid.free_segment(pose[:2], target, obstacles)
            else:
                segment_is_free = grid.free_entry_segment(pose[:2], target, obstacles)
            if not segment_is_free:
                self.path, self.goal = [], None
                return 0, 0

            heading = math.atan2(target[1] - pose[1], target[0] - pose[0])
            delta = wrap(heading - pose[2])
            if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
                return self._turn_command(delta, pose[2])

            self.motion_target = target
            speed = min(cfg.drive_speed_mm_s, max(15.0, math.dist(pose[:2], target) * 1.6))
            return speed, speed

        if self.heading is not None:
            delta = wrap(self.heading - pose[2])
            if abs(delta) > math.radians(cfg.turn_tolerance_degrees):
                return self._turn_command(delta, pose[2])

        self.goal = None
        self.heading = None
        self.predicted = set()
        return 0, 0

    # ------------------------------------------------------------------
    # 探索終了後の初期位置への帰還
    # ------------------------------------------------------------------
    def start_return(self, pose, home_pose):
        """現在地から初期位置までの帰還経路を作る。"""
        grid = self.world.grid
        obstacles = self.obstacles()
        home = (float(home_pose[0]), float(home_pose[1]))
        home_theta = float(home_pose[2])

        self.return_pose = (home[0], home[1], home_theta)
        self.return_path = []
        self.path = []
        self.predicted = set()
        self.heading = None
        self.goal = home
        self.mode = 'return'

        if math.dist(pose[:2], home) <= self.config.return_home_position_tolerance_mm:
            return True

        previous, start = grid.reachable(pose[:2], obstacles)
        if start is None:
            return False

        candidates = []
        home_is_safe = grid.inside(home, grid.margin)
        for key in previous:
            point = grid.cells[key]
            if home_is_safe:
                can_finish = grid.free_segment(point, home, obstacles)
            else:
                can_finish = grid.free_map_segment(point, home, obstacles)
            if not can_finish:
                continue

            route = grid.route(previous, key)
            cost = max(0, len(route) - 1) * grid.cell + math.dist(point, home)
            candidates.append((cost, key, route))

        if not candidates:
            return False

        _, _, route = min(candidates, key=lambda item: item[0])
        self.return_path = [grid.cells[key] for key in route]
        if not self.return_path or math.dist(self.return_path[-1], home) > 1.0:
            self.return_path.append(home)
        return True

    def return_command(self, pose):
        """帰還中の左右車輪速度を返す。None は帰還完了。"""
        self.motion_target = self.motion_heading = None
        if self.return_pose is None:
            return None

        cfg, grid = self.config, self.world.grid
        home_x, home_y, home_theta = self.return_pose
        home = (home_x, home_y)
        position_tolerance = cfg.return_home_position_tolerance_mm
        angle_tolerance = math.radians(cfg.return_home_angle_tolerance_degrees)

        while self.return_path:
            is_final = len(self.return_path) == 1
            tolerance = position_tolerance if is_final else 12.0
            if math.dist(pose[:2], self.return_path[0]) >= tolerance:
                break
            self.return_path.pop(0)

        if self.return_path:
            target = self.return_path[0]
            is_final = len(self.return_path) == 1
            obstacles = self.obstacles()

            if is_final:
                segment_is_free = grid.free_map_segment(pose[:2], target, obstacles)
            elif grid.inside(pose[:2], grid.margin):
                segment_is_free = grid.free_segment(pose[:2], target, obstacles)
            else:
                segment_is_free = grid.free_entry_segment(pose[:2], target, obstacles)

            if not segment_is_free:
                return 0, 0

            heading = math.atan2(target[1] - pose[1], target[0] - pose[0])
            delta = wrap(heading - pose[2])
            if abs(delta) > math.radians(7):
                return self._turn_command(delta, pose[2])

            self.motion_target = target
            speed = min(
                cfg.drive_speed_mm_s,
                max(12.0, math.dist(pose[:2], target) * 1.5),
            )
            return speed, speed

        delta = wrap(home_theta - pose[2])
        if abs(delta) > angle_tolerance:
            return self._turn_command(delta, pose[2])

        self.goal = None
        return None
