"""機体・カメラに依存しない地図、探索記録、キューブ台帳。単位はmm/rad。"""
import hashlib
import json
import math
import os
import tempfile
from collections import deque
from pathlib import Path


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as f:
            name = f.name
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()[:24]


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def hull(points):
    points = sorted(set(tuple(p) for p in points))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    result = lower[:-1] + upper[:-1]
    if len(result) < 3:
        raise ValueError('チェス座標から面積のある領域を作れません')
    return result


def segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    t = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy) / max(dx*dx+dy*dy, 1e-9)))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


class Grid:
    def __init__(self, landmarks, cell_mm=50, margin_mm=70):
        self.polygon = hull(landmarks)
        self.cell = cell_mm
        self.margin = margin_mm
        xs, ys = zip(*self.polygon)
        nx = math.ceil((max(xs)-min(xs))/cell_mm)+2
        ny = math.ceil((max(ys)-min(ys))/cell_mm)+2
        if nx * ny > 20000:
            raise ValueError('地図が大きすぎます。座標の単位mmを確認してください')
        self.cells = {}
        for ix in range(math.floor(min(xs)/cell_mm), math.ceil(max(xs)/cell_mm)+1):
            for iy in range(math.floor(min(ys)/cell_mm), math.ceil(max(ys)/cell_mm)+1):
                p = ((ix+0.5)*cell_mm, (iy+0.5)*cell_mm)
                if self.inside(p):
                    self.cells[f'{ix},{iy}'] = p
        self.walkable = {k for k,p in self.cells.items() if self.inside(p, margin_mm)}
        if not self.walkable:
            raise ValueError('機体幅の余裕を取れる走行領域がありません')

    def inside(self, p, margin=0):
        if len(p)<2 or not all(math.isfinite(v) for v in p[:2]):
            return False
        for a,b in zip(self.polygon, self.polygon[1:]+self.polygon[:1]):
            length = math.dist(a,b)
            if ((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])) / length < margin:
                return False
        return True

    def key(self, p):
        return f'{math.floor(p[0]/self.cell)},{math.floor(p[1]/self.cell)}'

    def free_segment(self, a, b, obstacles):
        """通常探索中の1区間が安全に走行できるか確認する。

        通常の探索では、区間の始点・終点とも boundary_margin_mm の
        内側にあることを要求する。
        """
        return (self.inside(a, self.margin) and self.inside(b, self.margin)
                and all(segment_distance((x,y), a,b) > r for x,y,r in obstacles))

    def free_entry_segment(self, a, b, obstacles):
        """淵付近の初期位置から最初の安全セルへ入る1区間を判定する。

        初期位置設定ではチェス枠の淵近くまで選択できるため、探索開始時だけ
        始点に boundary_margin_mm を要求しない。終点は必ず通常の安全領域内
        とし、キューブ等の障害物との距離条件は通常走行と同じにする。

        地図はチェス座標の凸包なので、始点・終点が地図内なら、その間の
        直線も地図の外へ飛び出さない。
        """
        return (self.inside(a) and self.inside(b, self.margin)
                and all(segment_distance((x,y), a,b) > r for x,y,r in obstacles))

    def free_map_segment(self, a, b, obstacles):
        """地図内の2点を結ぶ区間を、帰還時の最終区間として判定する。

        初期位置は boundary_margin_mm より外側（枠の淵付近）にも設定できる。
        そのため帰還の最後だけは通常の安全余白を要求せず、
        「地図の内側であること」と「既知障害物に接触しないこと」を確認する。
        通常探索ではこの判定を使用しない。
        """
        return (self.inside(a) and self.inside(b)
                and all(segment_distance((x,y), a,b) > r for x,y,r in obstacles))

    def reachable(self, position, obstacles):
        free = {k for k in self.walkable
                if all(math.hypot(self.cells[k][0]-x,self.cells[k][1]-y)>r for x,y,r in obstacles)}
        starts = sorted(free, key=lambda k: math.dist(position,self.cells[k]))

        # 通常の探索中は boundary_margin_mm の内側だけを走る。
        if self.inside(position, self.margin):
            start = next(
                (k for k in starts if self.free_segment(position, self.cells[k], obstacles)),
                None,
            )
        else:
            # 初期配置だけはチェス枠の淵近くまで指定できる。
            # その場合は、最初の1区間だけ地図内を通って最寄りの安全領域へ入り、
            # 以後は従来通り margin 内のセルだけを経路に使う。
            start = next(
                (
                    k for k in starts
                    if self.free_entry_segment(position, self.cells[k], obstacles)
                ),
                None,
            )

        if start is None:
            return {}, None
        previous = {start: None}
        queue = deque([start])
        while queue:
            k = queue.popleft()
            x,y = map(int,k.split(','))
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nxt = f'{x+dx},{y+dy}'
                if nxt in free and nxt not in previous and self.free_segment(self.cells[k],self.cells[nxt],obstacles):
                    previous[nxt] = k
                    queue.append(nxt)
        return previous, start

    @staticmethod
    def route(previous, target):
        path = []
        while target is not None:
            path.append(target)
            target = previous[target]
        return path[::-1]

    def visible(self, pose, near, far, fov, occluders=()):
        x,y,theta = pose
        # セル中心を観測した近似。既知物体の背後は観測済みにしない。
        visible = set()
        for k,p in self.cells.items():
            d = math.dist((x,y),p)
            if not near <= d <= far:
                continue
            if abs(wrap(math.atan2(p[1]-y,p[0]-x)-theta)) > fov/2:
                continue
            if any(math.dist((x,y),(ox,oy)) < d and
                   segment_distance((ox,oy),(x,y),p) <= radius for ox,oy,radius in occluders):
                continue
            visible.add(k)
        return visible


class World:
    def __init__(self, setup, robot_id):
        self.setup = setup
        self.robot_id = robot_id
        self.map_id = setup['map_id']
        self.trial_id = setup['trial_id']
        self.grid = Grid(setup['landmarks'].values(), setup['cell_mm'], setup['margin_mm'])
        # 共有セル/95%の分母は試行開始時の基準地図に固定する。
        # 局所的な駒の補正は、表示・照合・遮蔽・回り込み用の別座標へ反映する。
        self.landmarks = {str(k): tuple(p) for k, p in setup['landmarks'].items()}
        self.reference_landmarks = dict(self.landmarks)
        self.covered = set()  # 蒸発しない、二値の探索フェロモン
        self.local_covered = set()
        self.trails = {}      # 機体ごとの通過セル。重複受信で加算しない
        # 仮想フェロモン濃度。0.0～3.0程度を想定し、停止観測ごとに蒸発する。
        # full共有では相手機体の濃度も受け取り、重複探索を避ける判断に使う。
        self.pheromones = {}
        self.records = {}     # 観測元ID+連番。位置照合で代表荷物IDを決める
        self.sequence = 0
        self.local_found = set()

    def cubes(self):
        # 全観測をID順に統合するため、受信順に依存しない。
        groups = []
        radius = self.setup['cube_merge_mm']
        for key, r in sorted(self.records.items()):
            group = next((g for g in groups if all(r['source'] != v['source'] and math.hypot(r['x']-v['x'],r['y']-v['y']) <= radius
                                                  for _,v in g)), None)
            if group is None:
                groups.append([(key,r)])
            else:
                group.append((key,r))
        return [{'id':g[0][0], 'aliases':[k for k,_ in g],
                 'x':sum(r['x'] for _,r in g)/len(g), 'y':sum(r['y'] for _,r in g)/len(g),
                 'state':'uncollected', 'assigned_robot':None} for g in groups]

    def add_cube(self, x, y, elapsed):
        if not self.grid.inside((x,y)):
            return None
        # 静止荷物の第1段階では、確定した観測は書き換えない。
        local = [(k,r) for k,r in self.records.items() if r['source']==self.robot_id]
        near = [(math.hypot(x-r['x'],y-r['y']),k) for k,r in local]
        if near and min(near)[0] <= self.setup['cube_merge_mm']:
            return None
        self.sequence += 1
        key = f'{self.robot_id}:{self.sequence:04d}'
        self.records[key] = dict(x=x,y=y,source=self.robot_id,elapsed=elapsed)
        self.local_found.add(key)
        return key

    def upsert_cube(self, x, y, elapsed, record_id=None):
        """追跡器が同定したIDの座標を更新する。新IDを距離だけで結合しない。"""
        if not all(math.isfinite(v) for v in (x, y, elapsed)) or not self.grid.inside((x, y)):
            return None
        if record_id is None:
            self.sequence += 1
            record_id = f'{self.robot_id}:{self.sequence:04d}'
            self.records[record_id] = dict(x=x, y=y, source=self.robot_id, elapsed=elapsed,
                                          revision=1, updated_elapsed=elapsed)
            self.local_found.add(record_id)
            return record_id
        old = self.records[record_id]
        if old['source'] != self.robot_id:
            raise ValueError('他機の観測IDを更新できません')
        if math.dist((x, y), (old['x'], old['y'])) < 0.5:
            return record_id
        self.records[record_id] = dict(old, x=x, y=y,
                                      revision=old.get('revision', 0) + 1, updated_elapsed=elapsed)
        return record_id

    def observe(self, pose, visible):
        new = visible - self.local_covered
        self.local_covered |= visible
        self.covered |= visible
        k = self.grid.key(pose[:2])
        if k in self.grid.cells:
            self.trails.setdefault(self.robot_id,set()).add(k)
            self.covered.add(k)
        return len(new)


    def evaporate_pheromones(self, rate):
        """仮想フェロモンを一定割合で蒸発させる。"""
        rate = max(0.0, min(float(rate), 0.999))
        if rate <= 0:
            return
        keep = {}
        for key, value in self.pheromones.items():
            value = float(value) * (1.0 - rate)
            if value >= 0.03:
                keep[key] = value
        self.pheromones = keep

    def deposit_pheromone(self, pose, amount=1.0, spread_cells=1):
        """現在地と近傍セルへ仮想フェロモンを付与する。

        直前に通った場所ほど濃くなり、planner はこの濃度を経路コストへ
        加える。これにより同じ場所の往復・その場旋回を抑える。
        """
        center = self.grid.key(pose[:2])
        if center not in self.grid.cells:
            return
        cx, cy = map(int, center.split(','))
        spread = max(0, int(spread_cells))
        for dx in range(-spread, spread + 1):
            for dy in range(-spread, spread + 1):
                key = f'{cx+dx},{cy+dy}'
                if key not in self.grid.cells:
                    continue
                ring = max(abs(dx), abs(dy))
                factor = 1.0 if ring == 0 else 0.35 / ring
                self.pheromones[key] = min(3.0, self.pheromones.get(key, 0.0) + float(amount) * factor)

    def pheromone_at(self, key_or_point):
        """指定セルのフェロモン濃度を0～1へ正規化して返す。"""
        key = key_or_point if isinstance(key_or_point, str) else self.grid.key(key_or_point)
        return min(1.0, max(0.0, self.pheromones.get(key, 0.0) / 3.0))

    def all_trail_keys(self):
        result = set()
        for keys in self.trails.values():
            result.update(keys)
        return result

    def trail_distance(self, point):
        """自機・共有済み他機の軌跡までの最短距離。軌跡なしなら無限大。"""
        keys = self.all_trail_keys()
        if not keys:
            return float('inf')
        return min(math.dist(point, self.grid.cells[key]) for key in keys if key in self.grid.cells)

    def trail_coverage_cells(self, radius_mm):
        """軌跡の近傍として物理的に巡回したとみなせる走行可能セル。"""
        trails = [self.grid.cells[key] for key in self.all_trail_keys() if key in self.grid.cells]
        if not trails:
            return set()
        radius = float(radius_mm)
        return {
            key for key in self.grid.walkable
            if any(math.dist(self.grid.cells[key], point) <= radius for point in trails)
        }

    def trail_coverage_ratio(self, radius_mm):
        total = len(self.grid.walkable)
        if total == 0:
            return 0.0
        return len(self.trail_coverage_cells(radius_mm)) / total

    def snapshot(self, mode='full'):
        return {'records':{k: dict(v) for k, v in self.records.items()},
                'covered':sorted(self.covered) if mode=='full' else [],
                'trails':{k:sorted(v) for k,v in self.trails.items()} if mode=='full' else {},
                'pheromones':dict(self.pheromones) if mode=='full' else {}}

    def merge(self, data):
        # 検証をすべて済ませてから反映。不正データによる部分更新を防ぐ。
        if not isinstance(data,dict):
            raise ValueError('snapshot形式不正')
        records = data.get('records',{})
        covered = data.get('covered',[])
        trails = data.get('trails',{})
        pheromones = data.get('pheromones', {})
        if not isinstance(records,dict) or len(records)>2000:
            raise ValueError('荷物台帳形式不正')
        checked = {}
        for k,r in records.items():
            if not isinstance(k,str) or not isinstance(r,dict):
                raise ValueError('荷物形式不正')
            source = r.get('source')
            if source not in self.setup['robots'] or not k.startswith(source+':'):
                raise ValueError('未知の荷物発見元')
            for name in ('x','y','elapsed'):
                if not isinstance(r.get(name),(int,float)) or isinstance(r[name],bool) or not math.isfinite(r[name]):
                    raise ValueError('荷物数値不正')
            if not self.grid.inside((r['x'],r['y'])) or r['elapsed'] < 0:
                raise ValueError('荷物範囲不正')
            revision = r.get('revision', 0)
            if type(revision) is not int or revision < 0:
                raise ValueError('荷物の版番号が不正です')
            updated = r.get('updated_elapsed', r['elapsed'])
            if type(updated) not in (int, float) or not math.isfinite(updated) or updated < r['elapsed']:
                raise ValueError('荷物の更新時刻が不正です')
            if k in self.records:
                previous = self.records[k]
                if revision < previous.get('revision', 0):
                    continue  # 遅延した旧版で位置を巻き戻さない。
                if revision == previous.get('revision', 0) and previous != r:
                    raise ValueError('同じ荷物観測ID・版の内容が異なります')
                if revision > previous.get('revision', 0) and source == self.robot_id:
                    raise ValueError('自機の観測IDを他機から更新できません')
            checked[k] = dict(r)
        def check_cells(values):
            if not isinstance(values,list) or len(values)>len(self.grid.cells):
                raise ValueError('探索セル形式不正')
            if any(not isinstance(k,str) or k not in self.grid.cells for k in values):
                raise ValueError('地図外の探索セル')
            return set(values)
        cells = check_cells(covered)
        if not isinstance(trails,dict) or any(k not in self.setup['robots'] for k in trails):
            raise ValueError('軌跡形式不正')
        paths = {k:check_cells(v) for k,v in trails.items()}
        if not isinstance(pheromones, dict) or len(pheromones) > len(self.grid.cells):
            raise ValueError('フェロモン形式不正')
        checked_pheromones = {}
        for key, value in pheromones.items():
            if key not in self.grid.cells or type(value) not in (int, float) or isinstance(value, bool):
                raise ValueError('フェロモン形式不正')
            if not math.isfinite(value) or not 0 <= value <= 3.0:
                raise ValueError('フェロモン濃度不正')
            checked_pheromones[key] = float(value)
        self.records.update(checked)
        self.covered |= cells
        for k,v in paths.items():
            self.trails.setdefault(k,set()).update(v)
        # 相手から届いた濃度は、現在持つ濃度より高い場合だけ反映する。
        # 同じ通信を複数回受けても濃度が加算され続けない。
        for key, value in checked_pheromones.items():
            self.pheromones[key] = max(self.pheromones.get(key, 0.0), value)

    def export(self):
        return {'schema_version':1, 'map_id':self.map_id,'trial_id':self.trial_id,
                'robot_id':self.robot_id, 'cubes':self.cubes(),
                'landmarks':{k:list(p) for k,p in self.landmarks.items()},
                'local_covered':sorted(self.local_covered), **self.snapshot()}
