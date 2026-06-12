import time
import threading
from telecommunication.pc_communication import CoordCommunication

class CoordinateSender:
    """Cozmoの座標を裏側で取得し、指定した環境（PC-B）へ送信し続ける独立モジュール"""
    
    def __init__(self, cli, target_ip, robot_id="Cozmo-01", interval=0.2):
        self.cli = cli
        self.target_ip = target_ip
        self.robot_id = robot_id
        self.interval = interval
        self.env_client = CoordCommunication()
        self.running = False
        self.shutdown_requested = False  

    def _loop(self):
        """別スレッドで回り続けるメインループ"""
        while self.running:
            if self.cli.pose:
                x = self.cli.pose.position.x
                y = self.cli.pose.position.y
                
                # 🟢 送信結果を受け取る
                result = self.env_client.send_data(self.target_ip, x, y, robot_id=self.robot_id)
                
            time.sleep(self.interval)

    def start(self):
        """座標の送信をバックグラウンドで開始する"""
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self._loop, daemon=True)
            thread.start()
            print(f"[{self.robot_id}] 座標送信スレッドを起動しました (宛先: {self.target_ip})")

    def stop(self):
        """座標の送信を安全に停止する"""
        self.running = False