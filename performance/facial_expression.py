import time
import threading
from PIL import Image, ImageDraw

class FaceManager:
    """
    Cozmoの表情を管理するためのモジュールクラスです。
    他のプログラムから呼び出されて使用されることを前提としています。
    """
    def __init__(self, cli):
        self.cli = cli               # Cozmoへの接続クライアント
        self.current_frame = None    # 現在表示している画像フレーム
        self.stop_flag = False       # 画像送信スレッドを止めるためのフラグ
        self.stream_thread = None    # 画像送信専用スレッド

    def _frame_streamer(self):
        """画像を高速で送信し続ける専用のスレッド（内部用）"""
        while not self.stop_flag:
            if self.current_frame is not None:
                try:
                    # 非常に短い間隔で送信し、デフォルトの目が割り込む隙を与えない
                    self.cli.display_image(self.current_frame)
                except Exception:
                    pass
            time.sleep(0.03) # 約30FPSで更新

    def create_angry_face(self):
        """怒った顔の画像を作成する関数（目は動かない固定版）"""
        img = Image.new('1', (128, 32), color=0)
        draw = ImageDraw.Draw(img)
        
        # 1. 眉毛 (鋭い角度)
        draw.line((10, 0, 45, 12), fill=1, width=4)
        draw.line((118, 0, 83, 12), fill=1, width=4)
        
        # 2. 目 (固定された細長いスリット)
        draw.line((20, 18, 45, 18), fill=1, width=5)
        draw.line((83, 18, 108, 18), fill=1, width=5)

        # 3. への字の口 (大きく曲げる)
        draw.arc((40, 22, 88, 45), start=180, end=0, fill=1, width=3)
        
        return img

    def show_angry_face(self):
        """怒った顔を表示し、Cozmoのデフォルトの表情をブロックします"""
        self.current_frame = self.create_angry_face()
        
        # スレッドがまだ動いていなければ新しく開始する
        if self.stream_thread is None or not self.stream_thread.is_alive():
            self.stop_flag = False
            self.stream_thread = threading.Thread(target=self._frame_streamer, daemon=True)
            self.stream_thread.start()

    def show_default_face(self):
        """画像送信スレッドを停止し、Cozmoのデフォルトの表情に戻します"""
        self.stop_flag = True
        if self.stream_thread is not None:
            # スレッドが安全に終了するのを少しだけ待つ
            self.stream_thread.join(timeout=0.2)
            self.stream_thread = None
        self.current_frame = None