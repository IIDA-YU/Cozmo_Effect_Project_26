"""すべての車輪命令をここで直列化。段差・持上げ・通信停止で停止を保持。"""
import threading
import time


class SafetyClient:
    def __init__(self, cli, camera, config):
        import pycozmo
        self.raw, self.camera, self.config = cli,camera,config
        self.packet_type = pycozmo.protocol_encoder.RobotState
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.state_time = 0.0
        self.status = 0
        self.deadline = 0.0
        self.latched_reason = ''
        self.moving = False
        self.last_voltage = None
        self.handler = cli.add_handler(self.packet_type,self.on_state)
        self.thread = threading.Thread(target=self.watchdog,daemon=True)
        self.thread.start()

    def __getattr__(self, name):
        return getattr(self.raw,name)

    def on_state(self, cli, packet):
        self.state_time = time.monotonic()
        self.status = int(packet.status)
        self.last_voltage = getattr(packet,'battery_voltage',None)
        if self.status & (0x4000 | 0x08 | 0x20):
            self.latch('cliff / picked up / falling: reposition, then restart')

    def _send_wheels(self, lwheel_speed, rwheel_speed, **kwargs):
        """pycozmoへの車輪命令を直列化して送信する。"""
        connection_lock = getattr(self.raw, 'conn_lock', None)
        if connection_lock is None:
            self.raw.drive_wheels(lwheel_speed, rwheel_speed, **kwargs)
            return
        with connection_lock:
            self.raw.drive_wheels(lwheel_speed, rwheel_speed, **kwargs)

    def latch(self, reason):
        with self.lock:
            self.latched_reason = reason
            self._send_wheels(0,0)
            self.moving = False

    def drive_wheels(self, lwheel_speed, rwheel_speed, **kwargs):
        with self.lock:
            moving = bool(lwheel_speed or rwheel_speed)
            if moving:
                if self.latched_reason:
                    raise RuntimeError(self.latched_reason)
                _,_,stamp = self.camera.get_frame_sample()
                if time.monotonic()-stamp>self.config.frame_timeout_seconds:
                    raise RuntimeError('カメラ画像が古いため走行を停止しました')
                if time.monotonic()-self.state_time>self.config.state_timeout_seconds:
                    raise RuntimeError('機体状態が届いていないため走行を停止しました')
                self.deadline = time.monotonic()+0.3
            self._send_wheels(lwheel_speed,rwheel_speed,**kwargs)
            self.moving = moving

    def watchdog(self):
        while not self.stop_event.wait(0.03):
            with self.lock:
                if not self.moving:
                    continue
                _,_,stamp = self.camera.get_frame_sample()
                now = time.monotonic()
                try:
                    if now-self.state_time>self.config.state_timeout_seconds or now-stamp>self.config.frame_timeout_seconds:
                        self.latch('camera / robot state timeout: restart')
                    elif now>self.deadline:
                        self._send_wheels(0,0)
                        self.moving = False
                except Exception:
                    self.latched_reason = 'stop command failed: connection lost'

    def acknowledge_placement(self):
        # 手で初期位置に置いた後だけ呼ぶ。探索中の持上げは再初期化が必要。
        if self.status & (0x4000 | 0x08 | 0x20):
            raise RuntimeError('床に置いてから確定してください')
        if time.monotonic()-self.state_time>self.config.state_timeout_seconds:
            raise RuntimeError('機体状態を受信できていません')
        self.latched_reason = ''

    def stop(self):
        self.stop_event.set()
        try:
            self.drive_wheels(0,0)
        finally:
            self.raw.del_handler(self.packet_type,self.handler or self.on_state)
            self.thread.join(timeout=1)
