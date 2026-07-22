import time, threading

class RoamingController:
    """純粋に前進を維持するだけの運転手"""
    def __init__(self, cli, collision_preventer):
        self.cli = cli
        self.preventer = collision_preventer
        self.is_running = False

    def _drive_loop(self):
        was_avoiding = False  # ★追加：前回まで回避中だったかを記憶する
        while self.is_running:
            # 現在、回避中かどうか
            is_avoiding = self.preventer.is_avoiding_cliff or self.preventer.is_avoiding_collision
            
            if is_avoiding:
                was_avoiding = True  
            else:
                if was_avoiding:
                    # ★追加：回避終了直後。AIが安全確認をするための時間を0.5秒稼ぐ
                    time.sleep(0.5)
                    was_avoiding = False
                    continue  # すぐに前進せず、もう一度AIの検知結果（is_avoiding）を確認しに戻る
                
                # 崖や障害物から逃げている最中（直後）でなければ前進
                try: self.cli.drive_wheels(55, 55)
                except: pass
            time.sleep(0.1)

    def start(self):
        self.is_running = True
        # 頭とリフトの姿勢をセット
        if hasattr(self.cli, "conn_lock"):
            with self.cli.conn_lock:
                a = self.cli.set_head_angle(-0.1); a.wait() if a else None
                b = self.cli.set_lift_height(0.0); b.wait() if b else None
        threading.Thread(target=self._drive_loop, daemon=True).start()

    def stop(self):
        self.is_running = False
        try: self.cli.drive_wheels(0, 0)
        except: pass