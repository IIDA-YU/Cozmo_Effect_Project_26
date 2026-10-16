"""研究用の共通地図画面と、Cozmo の初期姿勢設定。

初期姿勢は、これまでと同じように

1. Cozmo の中心位置をクリック
2. Cozmo が向いている前方をクリック

の2操作で設定する。

設定後はキー操作で少しずつ位置・角度を補正できる。
座標の単位は mm、角度の単位は rad。
"""

import math

from research.world import hull
from research.cube_mapping import cube_corners


# Cozmo 本体のおおよその実寸。
# 地図上では、この大きさを縮尺に合わせて描画する。
DEFAULT_ROBOT_LENGTH_MM = 90.0   # 奥行（前後）9 cm
DEFAULT_ROBOT_WIDTH_MM = 60.0    # 横幅 6 cm
DEFAULT_CUBE_SIZE_MM = 50.0       # キューブ 5 cm × 5 cm


class MapWindow:
    """チェス地図・Cozmo・探索情報を表示する OpenCV ウィンドウ。"""

    NAME = "ant-COZMO Research"
    WIDTH = 900
    HEIGHT = 790
    CENTER_X = 450
    CENTER_Y = 390

    def __init__(
        self,
        landmarks,
        robot_length_mm=DEFAULT_ROBOT_LENGTH_MM,
        robot_width_mm=DEFAULT_ROBOT_WIDTH_MM,
        cube_size_mm=DEFAULT_CUBE_SIZE_MM,
        create_window=True,
    ):
        import cv2
        import numpy as np

        self.cv2 = cv2
        self.np = np
        self.landmarks = landmarks
        self.show_window = create_window

        # チェス駒の座標を外周で結び、研究領域を作る。
        self.polygon = hull(landmarks.values())
        xs, ys = zip(*self.polygon)

        # 地図の中心を画面中央へ合わせる。
        self.cx = (max(xs) + min(xs)) / 2
        self.cy = (max(ys) + min(ys)) / 2

        # 地図全体がウィンドウへ収まる縮尺を求める。
        self.scale = min(
            520 / max(max(xs) - min(xs), 1),
            740 / max(max(ys) - min(ys), 1),
        )

        # Cozmo の実寸。robot() で地図縮尺に合わせて描画する。
        self.robot_length_mm = float(robot_length_mm)
        self.robot_width_mm = float(robot_width_mm)
        self.cube_size_mm = float(cube_size_mm)

        # マウスで最後にクリックされた「地図座標」。
        self.last_click = None

        if create_window:
            cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.NAME, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        """画面上のクリック位置を研究用の mm 座標へ変換する。"""
        if event == self.cv2.EVENT_LBUTTONDOWN:
            self.last_click = (
                self.cx - (y - self.CENTER_Y) / self.scale,
                self.cy - (x - self.CENTER_X) / self.scale,
            )

    def pixel(self, point):
        """研究用の mm 座標を OpenCV の画面座標へ変換する。"""
        return (
            int(self.CENTER_X - (point[1] - self.cy) * self.scale),
            int(self.CENTER_Y - (point[0] - self.cx) * self.scale),
        )

    def draw(
        self,
        world=None,
        poses=None,
        current=None,
        status="",
        target=None,
        selected=None,
        help_text=None,
        cube_candidates=None,
        mapping_status="",
        localization=None,
        route=None,
        view=None,
    ):
        """現在の地図状態を1回描画する。"""
        cv2, np = self.cv2, self.np
        landmarks = world.landmarks if world is not None else self.landmarks
        img = np.full((self.HEIGHT, self.WIDTH, 3), 245, dtype=np.uint8)
        candidate_items = [p if isinstance(p, dict) else {'point': p}
                           for p in (cube_candidates or [])]

        # 地図外候補も実座標のまま画面内へ収める。地図・探索分母は変更しない。
        if cube_candidates or landmarks != self.landmarks:
            extent = list(self.polygon) + list(landmarks.values()) + [tuple(p['point']) for p in candidate_items]
            xs, ys = zip(*extent)
            pad = self.cube_size_mm
            self.cx, self.cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            self.scale = min(520 / max(max(xs) - min(xs) + 2 * pad, 1),
                             720 / max(max(ys) - min(ys) + 2 * pad, 1))

        # 探索中だけ表示する情報。
        if world:
            half = max(1, int(world.grid.cell * self.scale / 2))

            # 探索済みセル。
            for key in world.covered:
                p = self.pixel(world.grid.cells[key])
                cv2.rectangle(
                    img,
                    (p[0] - half, p[1] - half),
                    (p[0] + half, p[1] + half),
                    (220, 237, 220),
                    -1,
                )

            # Cozmo が通過した軌跡。
            for keys in world.trails.values():
                for key in keys:
                    cv2.circle(
                        img,
                        self.pixel(world.grid.cells[key]),
                        2,
                        (160, 90, 10),
                        -1,
                    )

            # 検出済みキューブ。
            # 画面上で固定ピクセルサイズにせず、実物 50 mm × 50 mm を
            # 地図の縮尺 self.scale に合わせて表示する。
            confirmed_cubes = world.cubes()
            for cube in confirmed_cubes:
                self._draw_cube(img, (cube['x'], cube['y']),
                                'CUBE ' + cube['id'].split(':')[-1], confirmed=True)

        # 距離条件などで登録を保留した推定位置は、輪郭だけの50mm角で表示する。
        # 確定済みキューブの近くは二重表示しない。
        if cube_candidates:
            confirmed_positions = []
            if world is not None:
                confirmed_positions = [(c["x"], c["y"]) for c in world.cubes()]
            for candidate in candidate_items:
                point = candidate['point']
                merge_mm = world.setup['cube_merge_mm'] if world else self.cube_size_mm * 0.9
                if any(math.hypot(point[0] - x, point[1] - y) <= merge_mm
                       for x, y in confirmed_positions):
                    continue
                label = 'CUBE POSITION CHECK' if candidate.get('visual_confirmed') else 'Cube? CHECK'
                if world is not None and not world.grid.inside(point):
                    label += ' OUTSIDE'
                self._draw_cube(img, point, label, confirmed=False)

        # 次の観測地点への経路とカメラの向きが、動作の目的を示す。
        if current and route:
            points = [self.pixel(current)] + [self.pixel(point) for point in route]
            cv2.polylines(img, [np.array(points, np.int32)], False, (190, 120, 40), 2)
        if current and view:
            near, far, fov, offset = view
            theta = current[2]
            origin = (current[0] + offset * math.cos(theta),
                      current[1] + offset * math.sin(theta))
            for sign in (-1, 1):
                angle = theta + sign * math.radians(fov) / 2
                a = (origin[0] + near * math.cos(angle), origin[1] + near * math.sin(angle))
                b = (origin[0] + far * math.cos(angle), origin[1] + far * math.sin(angle))
                cv2.line(img, self.pixel(a), self.pixel(b), (185, 190, 170), 1)

        # チェス駒を結んだ研究領域の外枠。
        cv2.polylines(
            img,
            [np.array([self.pixel(p) for p in self.polygon], np.int32)],
            True,
            (70, 100, 70),
            2,
        )

        # 各チェス駒の位置。
        for marker_id, point in landmarks.items():
            px = self.pixel(point)
            seen = str(marker_id) in set((localization or {}).get('seen_ids', []))
            color = (50, 150, 40) if seen else (50, 50, 220)
            cv2.circle(img, px, 5, color, -1)
            refined = math.dist(point, self.landmarks.get(marker_id, point)) >= 0.1
            cv2.putText(
                img,
                f"Chess {marker_id}" + (' OK' if seen else '') + (' AVG' if refined else ''),
                (px[0] + 6, px[1] - 8),
                0,
                0.4,
                color,
                1,
            )

        # 初期姿勢設定画面だけ配置を表示する。探索/結果画面は現在地のみ。
        setup_poses = (poses or {}) if world is None and current is None else {}
        for name, pose in setup_poses.items():
            self.robot(
                img,
                name,
                (pose["x"], pose["y"], pose["theta"]),
                (180, 70, 130),
                selected=(name == selected),
            )

        # 探索中の自機位置。
        if current:
            self.robot(img, "SELF", current, (20, 160, 20))

        # 位置だけをクリックし、まだ向きを決めていない状態。
        if target:
            cv2.drawMarker(
                img,
                self.pixel(target),
                (180, 100, 0),
                cv2.MARKER_CROSS,
                18,
                2,
            )

        # 上部の操作状況。
        cv2.rectangle(img, (0, 0), (self.WIDTH, 70), (245, 245, 245), -1)
        cv2.putText(img, status[:100], (12, 26), 0, 0.50, (20, 20, 20), 1)
        if mapping_status:
            cv2.putText(img, mapping_status[:100], (12, 52), 0, 0.47, (20, 60, 130), 1)
        # 縮尺バーと、保存地図に対する観測間隔・残差を表示する。
        bar = min(100.0, 180.0 / self.scale)
        cv2.line(img, (18, self.HEIGHT - 54), (18 + int(bar*self.scale), self.HEIGHT - 54), (60,60,60), 2)
        cv2.putText(img, f'{bar:.0f} mm (map scale)', (18, self.HEIGHT - 63), 0, .43, (60,60,60), 1)
        if localization:
            residual = localization.get('residual_mm')
            seen_ids = set(localization.get('seen_ids', []))
            missing = sorted(set(map(str, landmarks)) - seen_ids)
            text = 'POSE: ' + localization.get('status', '')
            text += f' / CHESS {len(seen_ids)}/{len(landmarks)}'
            if missing:
                text += ' / missing=' + ','.join(missing)
            if residual is not None:
                text += f' / residual={residual:.1f}mm'
            cv2.putText(img, text[:110], (12, 73), 0, .45, (100,60,20), 1)
            collecting = [s for s in localization.get('refinement', [])
                          if s['status'] in ('COLLECTING', 'UNSTABLE') and s.get('shift_mm', 0) >= 8]
            if collecting:
                note = 'CHESS AVERAGING: ' + ', '.join(f"{s['id']} ({s['samples']} samples)" for s in collecting)
                cv2.putText(img, note[:110], (12, 94), 0, .43, (100,60,20), 1)
            for row, pair in enumerate(localization.get('pairs', [])[:2]):
                text = (f"Chess {'-'.join(pair['ids'])}: map={pair['map_mm']:.0f}mm "
                        f"observed={pair['observed_mm']:.0f}mm delta={pair['error_mm']:+.0f}mm")
                cv2.putText(img, text, (12, 114 + row*19), 0, .43, (100,60,20), 1)


        # 下部の簡易ヘルプ。
        # 初期位置設定と探索実験では操作が異なるため、表示を分ける。
        if help_text is None:
            if world is not None:
                help_text = "Q: stop | P: pause/resume | I: intervention"
                cv2.putText(img, f'Orange: cube {self.cube_size_mm:g}x{self.cube_size_mm:g} mm | '
                            'Green: observed | Blue: route | Boundary: reference',
                            (12, 748), 0, 0.44, (40, 70, 110), 1)
            else:
                help_text = (
                    "X:up / Y:left | WASD: move 5mm | J/L: rotate | "
                    "U: undo | R: reset | Enter: save"
                )
        cv2.putText(img, help_text, (12, 775), 0, 0.47, (20, 20, 20), 1)

        if self.show_window:
            cv2.imshow(self.NAME, img)
        return img

    def _draw_cube(self, img, point, label, confirmed):
        """地図座標の50mm角を画素へ変換する。描画だけの固定サイズにしない。"""
        cv2, np = self.cv2, self.np
        corners = np.array([self.pixel(p) for p in cube_corners(point, self.cube_size_mm)], np.int32)
        color = (0, 90, 230) if confirmed else (0, 180, 220)
        if confirmed:
            cv2.fillConvexPoly(img, corners, (100, 185, 255))
        cv2.polylines(img, [corners], True, color, 2 if confirmed else 1)
        inset = np.array([self.pixel(p) for p in cube_corners(point, self.cube_size_mm * 0.48)], np.int32)
        cv2.polylines(img, [inset], True, color, 1)
        text_width = cv2.getTextSize(label, 0, 0.4, 1)[0][0]
        tx = max(6, min(int(corners[:, 0].max()) + 4, self.WIDTH - text_width - 6))
        ty = max(86, min(int(corners[:, 1].min()) - 4, 730))
        cv2.putText(img, label, (tx, ty), 0, 0.4, color, 1)

    def _robot_corners(self, pose):
        """Cozmo の実寸から、四隅の地図座標を求める。"""
        x, y, theta = pose
        half_length = self.robot_length_mm / 2
        half_width = self.robot_width_mm / 2

        # 前方向と、その左方向の単位ベクトル。
        forward = (math.cos(theta), math.sin(theta))
        left = (-math.sin(theta), math.cos(theta))

        corners = []
        for forward_sign, left_sign in (
            (1, 1),
            (1, -1),
            (-1, -1),
            (-1, 1),
        ):
            corners.append(
                (
                    x
                    + forward_sign * half_length * forward[0]
                    + left_sign * half_width * left[0],
                    y
                    + forward_sign * half_length * forward[1]
                    + left_sign * half_width * left[1],
                )
            )
        return corners

    def robot(self, img, label, pose, color, selected=False):
        """Cozmo を 90 mm × 60 mm の実寸比で地図上へ描画する。"""
        cv2, np = self.cv2, self.np
        center = self.pixel(pose)
        corners = np.array(
            [self.pixel(point) for point in self._robot_corners(pose)],
            np.int32,
        )

        # 本体を薄く塗り、外形をはっきり表示する。
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, corners, color)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [corners], True, color, 3 if selected else 2)

        # 中心点。
        cv2.circle(img, center, 3, color, -1)

        # 前方向。矢印の長さも mm を基準にする。
        arrow_length_mm = self.robot_length_mm * 0.72
        forward_point = (
            pose[0] + arrow_length_mm * math.cos(pose[2]),
            pose[1] + arrow_length_mm * math.sin(pose[2]),
        )
        cv2.arrowedLine(
            img,
            center,
            self.pixel(forward_point),
            color,
            2,
            tipLength=0.25,
        )

        # 微調整対象は [SELECT] を付けて分かりやすくする。
        shown_label = f"{label} [SELECT]" if selected else label
        cv2.putText(
            img,
            shown_label,
            (center[0] + 8, center[1] + 17),
            0,
            0.43,
            color,
            1,
        )

    def nearest_robot(self, poses, point):
        """クリック位置の近くにある Cozmo 名を返す。"""
        if not poses:
            return None

        # 本体の対角線半分 + 少しのクリック余裕。
        select_radius = math.hypot(
            self.robot_length_mm / 2,
            self.robot_width_mm / 2,
        ) + 15

        candidates = []
        for name, pose in poses.items():
            distance = math.hypot(point[0] - pose["x"], point[1] - pose["y"])
            if distance <= select_radius:
                candidates.append((distance, name))

        return min(candidates)[1] if candidates else None

    def close(self):
        if self.show_window:
            self.cv2.destroyWindow(self.NAME)


def _position_is_valid(grid, point, poses, ignore_name=None):
    """初期位置として使えるか確認する。

    初期配置では従来の ``grid.margin`` は使用しない。
    チェス駒で囲まれた枠の内側であれば、淵の近くまで指定できる。

    自律走行中の安全余白 ``boundary_margin_mm`` は別処理で維持される。
    """
    if not grid.inside(point):
        return False, "チェスの駒で囲まれた枠内をクリックしてください"

    # 他の機体との間隔チェックは従来の条件を維持する。
    for name, pose in poses.items():
        if name == ignore_name:
            continue
        distance = math.hypot(point[0] - pose["x"], point[1] - pose["y"])
        if distance < 2 * grid.margin:
            return False, "初期位置が他の機体に近すぎます"

    return True, ""


def _move_selected_pose(poses, name, dx, dy, grid):
    """選択中の Cozmo を少しだけ移動する。"""
    if name not in poses:
        return False

    pose = poses[name]
    new_position = (pose["x"] + dx, pose["y"] + dy)
    valid, message = _position_is_valid(
        grid,
        new_position,
        poses,
        ignore_name=name,
    )
    if not valid:
        print(message)
        return False

    pose["x"], pose["y"] = new_position
    return True


def choose_initial_poses(
    window,
    grid,
    robots,
    camera,
    adjust_step_mm=5.0,
    adjust_angle_degrees=2.0,
):
    """全 Cozmo の初期位置と向きをマウスで設定する。

    基本操作
    --------
    - 1回目のクリック: Cozmo の中心位置
    - 2回目のクリック: Cozmo の前方向
    - W/A/S/D: 選択中の位置を 5 mm 単位で微調整
    - J/L: 選択中の向きを少しずつ回転
    - U: 直前の機体をやり直す
    - R: 全機体を最初からやり直す
    - Enter: 全機体の設定を保存

    全機体を設定した後は、地図上の Cozmo をクリックすると
    微調整する機体を変更できる。
    """
    poses = {}
    position = None
    names = list(robots)
    index = 0
    selected_name = None
    angle_step = math.radians(adjust_angle_degrees)

    print(
        "機体ごとに中心位置→前方の順にクリックしてください。\n"
        "W/A/S/D: 位置微調整、J/L: 向き微調整、"
        "U: 直前をやり直す、R: 全リセット、Enter: 確定。"
    )

    while True:
        complete = index == len(names)

        if complete:
            selected_text = selected_name or "なし"
            status = (
                f"設定完了 / 調整対象: {selected_text} / "
                "Cozmoをクリックして選択 / ENTER: 保存"
            )
        elif position is None:
            status = f"{names[index]}: 中心位置をクリックしてください"
        else:
            status = (
                f"{names[index]}: 前方向をクリックしてください / "
                "WASDで中心位置を微調整できます"
            )

        window.draw(
            poses=poses,
            status=status,
            target=position,
            selected=selected_name,
        )
        camera.display_once()
        key = camera.poll_key(20)

        # Q / Esc: 設定を保存せず終了。
        if key in (27, ord("q"), ord("Q")):
            return None

        # R: 最初から設定し直す。
        if key in (ord("r"), ord("R")):
            poses = {}
            index = 0
            position = None
            selected_name = None
            continue

        # U: 現在のクリック、または直前の機体設定だけを取り消す。
        if key in (ord("u"), ord("U")):
            if position is not None:
                position = None
            elif index > 0:
                index -= 1
                removed_name = names[index]
                poses.pop(removed_name, None)
                selected_name = names[index - 1] if index > 0 else None
            continue

        # 全機体の入力が終わった後だけ Enter で保存する。
        if key in (10, 13) and complete:
            return poses

        # --------------------------------------------------------------
        # キーボードによる微調整
        # --------------------------------------------------------------
        move_keys = {
            ord("w"): (adjust_step_mm, 0),
            ord("W"): (adjust_step_mm, 0),
            ord("s"): (-adjust_step_mm, 0),
            ord("S"): (-adjust_step_mm, 0),
            ord("a"): (0, adjust_step_mm),
            ord("A"): (0, adjust_step_mm),
            ord("d"): (0, -adjust_step_mm),
            ord("D"): (0, -adjust_step_mm),
        }

        if key in move_keys:
            dx, dy = move_keys[key]

            # 向きを決める前なら、クリックした中心位置を微調整する。
            if position is not None:
                candidate = (position[0] + dx, position[1] + dy)
                valid, message = _position_is_valid(grid, candidate, poses)
                if valid:
                    position = candidate
                else:
                    print(message)

            # すでに設定済みなら、選択中の機体を微調整する。
            elif selected_name is not None:
                _move_selected_pose(poses, selected_name, dx, dy, grid)

            continue

        # J/L: 選択中 Cozmo の向きだけを微調整する。
        if selected_name in poses and key in (
            ord("j"), ord("J"), ord("l"), ord("L")
        ):
            direction = 1 if key in (ord("j"), ord("J")) else -1
            theta = poses[selected_name]["theta"] + direction * angle_step
            poses[selected_name]["theta"] = (theta + math.pi) % (2 * math.pi) - math.pi
            continue

        # --------------------------------------------------------------
        # マウスクリック処理
        # --------------------------------------------------------------
        click = window.last_click
        window.last_click = None

        if click is None:
            continue

        # 全機体を設定した後は、Cozmo をクリックして調整対象を変更する。
        if complete:
            clicked_robot = window.nearest_robot(poses, click)
            if clicked_robot is not None:
                selected_name = clicked_robot
            continue

        # 1回目: 中心位置。
        if position is None:
            valid, message = _position_is_valid(grid, click, poses)
            if not valid:
                print(message)
                continue
            position = click
            continue

        # 2回目: 前方向。
        # 2点が近すぎると角度が不安定になるため、従来通り30mm以上とする。
        if math.dist(position, click) < 30:
            print("向きを決める点は、中心位置から30mm以上離してクリックしてください")
            continue

        robot_name = names[index]
        poses[robot_name] = {
            "x": position[0],
            "y": position[1],
            "theta": math.atan2(
                click[1] - position[1],
                click[0] - position[0],
            ),
            "marker_id": robots[robot_name],
        }

        # 直前に設定した機体を、そのまま微調整対象にする。
        selected_name = robot_name
        index += 1
        position = None
