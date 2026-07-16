#----------------------------------------------------------------------
# battery.py
# Cozmoのバッテリー残量を管理するためのモジュールクラス
#----------------------------------------------------------------------

import os
import time
import threading
from collections import deque

import pycozmo
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ===== バッテリー電圧の目安 =====
FULL_VOLTAGE = 4.2    # 満充電あたりの電圧(V) … これ以上は100%
EMPTY_VOLTAGE = 3.6   # 空に近い電圧(V) … これ以下は0%

# ===== 実測（平均）と充電判定の設定 =====
WINDOW_SECONDS = 1.0   # この秒数ぶんを実測して平均する（直近1秒）
CHARGING_VOLTAGE = 4.2  # この電圧以上を「満充電(100%)」とみなす
CHARGING_MIN_HITS = 3   # 充電判定に必要な最低サンプル数
CHARGING_RATIO = 0.8    # 1秒ぶんの8割以上が100%なら「充電中」とみなす

# ===== 「充電中」表示用の日本語フォント =====
# cv2.putText は日本語（全角文字）を描画できないため、
# 「充電中」の文字だけはPillow(PIL)を使って別方式で描画する。
# 環境ごとにインストール場所が違うので、よくあるパスを順番に探す。
_JP_FONT_CANDIDATES = [
    "C:/Windows/Fonts/meiryo.ttc",                      # Windows
    "C:/Windows/Fonts/msgothic.ttc",                    # Windows
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",   # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",        # macOS
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",     # Linux
]
JP_FONT_SIZE = 18


def _find_jp_font_path():
    """使用できる日本語フォントのパスを探す。見つからなければNoneを返す。"""
    for path in _JP_FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


_JP_FONT_PATH = _find_jp_font_path()
# フォントが見つかった場合だけPIL用のフォントオブジェクトを用意しておく
_JP_FONT = ImageFont.truetype(_JP_FONT_PATH, JP_FONT_SIZE) if _JP_FONT_PATH else None


def _read_voltage(pkt):
    """RobotStateパケットから電圧を取り出す（属性名の違いを吸収）。なければNone。"""
    return getattr(pkt, "battery_voltage", getattr(pkt, "batt_voltage", None))


def voltage_to_percent(voltage):
    """電圧(V)をおおよその残量(%)に変換し、0〜100に収めて返す。"""
    percent = (voltage - EMPTY_VOLTAGE) / (FULL_VOLTAGE - EMPTY_VOLTAGE) * 100
    return max(0.0, min(100.0, percent))


def check_battery(cli, timeout=5.0):
    """
    バッテリー残量を一度だけ取得し、コンソールへ表示する（起動時の確認などに使用）。
    """
    data_received = threading.Event()

    def on_robot_state(cli_instance, pkt):
        if data_received.is_set():
            return

        voltage = _read_voltage(pkt)
        if voltage is not None:
            percent = voltage_to_percent(voltage)
            print(f"🔋 バッテリー電圧: {voltage:.2f} V （残量目安: {percent:.0f} %）")
            if voltage <= EMPTY_VOLTAGE:
                print("⚠️ バッテリーが少なくなっています。充電をおすすめします。")
            elif voltage >= 4.1:
                print("✨ 十分に充電されています。")
        else:
            print("❌ バッテリー情報を取得できませんでした。")

        data_received.set()
        try:
            cli_instance.del_handler(pycozmo.protocol_encoder.RobotState, on_robot_state)
        except ValueError:
            pass

    print("🔍 バッテリーデータを取得しています...")
    cli.add_handler(pycozmo.protocol_encoder.RobotState, on_robot_state)
    data_received.wait(timeout=timeout)


class BatteryMonitor:
    """
    CozmoのRobotStateを購読し、直近 WINDOW_SECONDS 秒ぶんの電圧を実測し続けるクラス。

    - 表示には1秒ぶんの「平均電圧」を使うため、値が安定する。
    - 1秒ぶんのほとんどが満充電電圧以上なら「充電中」と判定する。
    - カメラ映像の右下へ、コンパクトな残量インジケータを描画できる。
    """

    def __init__(self, cli):
        self.cli = cli
        self._samples = deque()        # (時刻, 電圧) を貯めるバッファ
        self._lock = threading.Lock()
        self._started = False

    # ---------- データ取得 ----------
    def _on_robot_state(self, cli, pkt):
        voltage = _read_voltage(pkt)
        if voltage is None:
            return

        now = time.time()
        with self._lock:
            self._samples.append((now, voltage))
            # 直近 WINDOW_SECONDS 秒より古いサンプルを捨てる
            cutoff = now - WINDOW_SECONDS
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()

    def start(self):
        """RobotStateの購読を開始する。"""
        if not self._started:
            self.cli.add_handler(pycozmo.protocol_encoder.RobotState, self._on_robot_state)
            self._started = True

    def stop(self):
        """RobotStateの購読を停止する。"""
        if self._started:
            try:
                self.cli.del_handler(pycozmo.protocol_encoder.RobotState, self._on_robot_state)
            except ValueError:
                pass
            self._started = False

    # ---------- 状態の計算 ----------
    def _read_state(self):
        """直近1秒の平均から (残量%, 充電中か) を返す。データが無ければ (None, False)。"""
        with self._lock:
            voltages = [v for _, v in self._samples]

        if not voltages:
            return None, False

        avg_voltage = sum(voltages) / len(voltages)
        percent = voltage_to_percent(avg_voltage)

        # 100%（満充電電圧以上）のサンプルが、ほとんどを占めていれば「充電中」
        high_hits = sum(1 for v in voltages if v >= CHARGING_VOLTAGE)
        charging = (len(voltages) >= CHARGING_MIN_HITS
                    and high_hits / len(voltages) >= CHARGING_RATIO)
        return percent, charging

    @property
    def percent(self):
        """直近1秒の平均残量(%)。まだ取得できていなければNone。"""
        return self._read_state()[0]

    @property
    def is_charging(self):
        """充電中ならTrue。"""
        return self._read_state()[1]

    # ---------- 描画 ----------
    def draw_overlay(self, frame):
        """カメラ映像(frame)の右下にバッテリー残量をコンパクトに描画して返す。"""
        if frame is None or frame.ndim != 3:
            return frame

        percent, charging = self._read_state()

        # 表示する文字・塗り色(BGR)を決める
        if charging:
            # 充電中は実際の残量(%)を表示すると誤解を招くため、
            # 数値の代わりに「充電中」という文字を表示する
            label = "充電中"
            fill_ratio = 1.0
            fill_color = (255, 200, 80)    # 水色（充電中）
        elif percent is None:
            label = "--%"
            fill_ratio = 0.0
            fill_color = (160, 160, 160)   # グレー（取得待ち）
        else:
            label = f"{percent:.0f}%"
            fill_ratio = percent / 100.0
            if percent >= 50:
                fill_color = (90, 200, 90)    # 緑
            elif percent >= 20:
                fill_color = (60, 170, 255)   # オレンジ
            else:
                fill_color = (60, 60, 230)    # 赤

        h, w = frame.shape[:2]

        # コンパクトなレイアウト（右下）
        margin, pad, gap = 12, 8, 7
        icon_w, icon_h = 26, 13
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thick = 0.45, 1

        if charging and _JP_FONT is not None:
            # 「充電中」は日本語なのでPillowでサイズを計測する
            jp_bbox = _JP_FONT.getbbox(label)
            tw, th = jp_bbox[2] - jp_bbox[0], jp_bbox[3] - jp_bbox[1]
        else:
            if charging:
                # 日本語フォントが見つからない環境向けのフォールバック表示
                label = "CHARGING"
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

        panel_w = pad + icon_w + 3 + gap + tw + pad
        panel_h = max(icon_h, th) + pad + 6
        x2, y2 = w - margin, h - margin
        x1, y1 = x2 - panel_w, y2 - panel_h

        # 半透明の角丸パネル
        overlay = frame.copy()
        self._rounded_rect(overlay, (x1, y1), (x2, y2), (35, 35, 35), radius=8)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cy = (y1 + y2) // 2

        # バッテリーアイコン本体
        bx1 = x1 + pad
        by1 = cy - icon_h // 2
        bx2 = bx1 + icon_w
        by2 = by1 + icon_h
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (235, 235, 235), 1, cv2.LINE_AA)
        # 端子（プラス側の出っ張り）
        cv2.rectangle(frame, (bx2 + 1, by1 + 4), (bx2 + 3, by2 - 4), (235, 235, 235), -1)

        # 残量の塗りつぶし
        inner_w = int((bx2 - bx1 - 3) * fill_ratio)
        if inner_w > 0:
            cv2.rectangle(frame, (bx1 + 1, by1 + 1), (bx1 + 1 + inner_w, by2 - 1), fill_color, -1)

        # 充電中は稲妻マークを重ねる
        if charging:
            self._draw_bolt(frame, (bx1 + icon_w // 2, cy), icon_h)

        # ラベル表示（「充電中」の日本語はPillowで、それ以外はcv2で描画）
        text_x = bx2 + 3 + gap
        if charging and _JP_FONT is not None:
            text_y_top = cy - th // 2 - jp_bbox[1]  # PILは左上基準なので補正する
            frame = self._draw_jp_text(frame, label, (text_x, text_y_top), (245, 245, 245))
        else:
            text_y = cy + th // 2
            cv2.putText(frame, label, (text_x, text_y), font, font_scale,
                        (245, 245, 245), font_thick, cv2.LINE_AA)
        return frame

    @staticmethod
    def _rounded_rect(img, top_left, bottom_right, color, radius):
        """角丸の塗りつぶし四角形を描く補助関数。"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1)

    @staticmethod
    def _draw_jp_text(frame, text, top_left, color_bgr):
        """
        日本語テキストをフレームに描画する補助関数。
        cv2.putText は日本語（全角文字）を描画できないため、
        一度PIL画像に変換して文字を描き、またOpenCVの形式（BGR）に戻している。
        """
        x, y = top_left
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(img_pil)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR→RGBに変換
        draw.text((x, y), text, font=_JP_FONT, fill=color_rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _draw_bolt(frame, center, size):
        """充電中を表す稲妻マークを描く補助関数。"""
        cx, cy = center
        s = size * 0.5
        pts = np.array([
            [cx + 0.20 * s, cy - 1.00 * s],
            [cx - 0.45 * s, cy + 0.10 * s],
            [cx + 0.05 * s, cy + 0.10 * s],
            [cx - 0.20 * s, cy + 1.00 * s],
            [cx + 0.45 * s, cy - 0.10 * s],
            [cx - 0.05 * s, cy - 0.10 * s],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (30, 30, 30), cv2.LINE_AA)


if __name__ == "__main__":
    # このファイルだけを直接実行したときのテスト用コードです
    print("🤖 Cozmoに接続しています...")
    with pycozmo.connect() as cli:
        check_battery(cli)