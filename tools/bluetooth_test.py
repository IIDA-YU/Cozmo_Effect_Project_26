"""機体に接続せず、Bluetooth COMの相手IDと往復通信を確認する。"""
import time
from telecommunication.bluetooth_communication import BluetoothNetwork


def run(config):
    from application.controller import validate_ports
    validate_ports(config,{'robots':config.robot_markers})
    network = BluetoothNetwork(config.bluetooth_ports)
    seen = set()
    try:
        network.start()
        print('両PCでテストを起動してください。30秒間、PING/PONGを確認します。')
        started,last = time.monotonic(),0
        while time.monotonic()-started<30:
            now = time.monotonic()
            if now-last>=2:
                for peer in config.bluetooth_ports:
                    network.send(peer,dict(kind='test_ping',source=config.robot_id,target=peer))
                print('接続状態:',network.status)
                last = now
            for peer,msg,_ in network.receive():
                if msg.get('source')!=peer or msg.get('target')!=config.robot_id:
                    print('相手ID不一致:',peer,msg)
                    continue
                if msg.get('kind')=='test_ping':
                    network.send(peer,dict(kind='test_pong',source=config.robot_id,target=peer))
                elif msg.get('kind')=='test_pong':
                    seen.add(peer)
                    print('往復確認OK:',peer)
            time.sleep(.05)
    finally:
        network.stop()
    print('往復確認できた相手:',sorted(seen))
    if set(config.bluetooth_ports)-seen:
        raise RuntimeError('未確認の相手があります。COM設定・SPP対応・起動状態を確認してください')
