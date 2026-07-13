#----------------------------------------------------------------------
# step_detection.py
# 段差をセンサーで検知するためのモジュールクラス
#----------------------------------------------------------------------

import pycozmo

class StepDetector:
    """段差（崖）センサーだけを監視し、見つけたら回避の専門家に報告するクラス"""
    def __init__(self, cli, collision_preventer):
        self.cli = cli
        self.preventer = collision_preventer
        self.is_running = False

    def on_robot_state(self, cli, pkt=None, *args, **kwargs):
        if not self.is_running or pkt is None: return
        
        if hasattr(pkt, "status"):
            is_cliff = bool(pkt.status & 0x4000)
            if is_cliff:
                # 崖を見つけたら即座に回避モジュールへ報告
                self.preventer.trigger_cliff_avoidance()

    def start(self):
        self.is_running = True
        self.cli.add_handler(pycozmo.protocol_encoder.RobotState, self.on_robot_state)
        print("🤖 段差検知センサーを起動しました。")
        
        

    def stop(self):
        self.is_running = False
        try:
            self.cli.del_handler(pycozmo.protocol_encoder.RobotState, self.on_robot_state)
        except Exception:
            pass