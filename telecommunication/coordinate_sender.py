import time
import threading
import queue
from telecommunication.pc_communication import CoordCommunication

class CoordinateSender:
    """Cozmoの座標を裏側で取得し、指定した環境（PC-B）へ非同期で送信し続ける独立モジュール"""
    
    def __init__(self, cli, target_ip, robot_id="Cozmo-01", interval=0.2):
        self.cli = cli
        self.target_ip = target_ip
        self.robot_id = robot_id
        self.interval = interval
        self.env_client = CoordCommunication()
        self.running = False
        
        # ★ 追加: 最新の座標1つだけを保持するキュー（待ち行列）
        self.send_queue = queue.Queue(maxsize=1)

    def _collect_loop(self):
        """座標を取得してキューに入れる（高速で回る）"""
        while self.running:
            if self.cli.pose:
                x = self.cli.pose.position.x
                y = self.cli.pose.position.y
                
                # すでに未送信の古い座標があれば捨てて、最新の座標を入れる
                if self.send_queue.full():
                    try:
                        self.send_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.send_queue.put((x, y))
                
            time.sleep(self.interval)

    def _send_loop(self):
        """キューから座標を取り出して通信送信する（通信が遅れても他を止めない）"""
        while self.running:
            try:
                # 座標データが来るのを最大0.5秒待つ
                x, y = self.send_queue.get(timeout=0.5)
                # 🟢 P2P送信（ここで多少時間がかかってもCozmoの動作は止まらない）
                self.env_client.send_data(self.target_ip, x, y, robot_id=self.robot_id)
            except queue.Empty:
                continue

    def start(self):
        """非同期の座標送信スレッドを2つ起動する"""
        if not self.running:
            self.running = True
            threading.Thread(target=self._collect_loop, daemon=True).start()
            threading.Thread(target=self._send_loop, daemon=True).start()
            print(f"[{self.robot_id}] 非同期の座標送信スレッドを起動しました (宛先: {self.target_ip})")

    def stop(self):
        """送信を安全に停止する"""
        self.running = False