import socket
import json
import time

class CoordCommunication:
    def __init__(self, port=50007):
        self.port = port
        self.running = True

    def get_local_ip(self):
        """教室のWi-Fiや有線LANなど、アクティブなネットワークのIPアドレスを確実に取得する"""
        try:
            # ダミーのUDPソケットを作成して接続を試みることで、現在使用中のNICのIPを取得
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except:
                return "127.0.0.1"

    def start_server(self):
        """受信側（表示PC：PC-B）として起動"""
        local_ip = self.get_local_ip()

        print(f"\n=== 座標受信サーバー (表示用) ===")
        print(f"このPCのIPアドレス: {local_ip}")
        print(f"ポート番号: {self.port}")
        print("---------------------------------")
        print("※ 送信側PCの main.py 内の FIELD_IP にこのIPアドレスを設定してください。")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen(5)
            print("接続待機中... (Ctrl+Cで終了)")

            while self.running:
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                    with conn:
                        data = conn.recv(1024)
                        if data:
                            msg = json.loads(data.decode('utf-8'))
                            # 画面にリアルタイム表示
                            if msg['id'] != "Cozmo-A":
                                print(f"[{time.strftime('%H:%M:%S')}] {msg['id']}: X={msg['x']:>6.2f}, Y={msg['y']:>6.2f}")
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\nサーバーを停止します。")
                    break
                except Exception as e:
                    print(f"受信エラー: {e}")

    def send_data(self, target_ip, x, y, robot_id="Cozmo-01"):
        """送信側（操作PC：PC-A）からデータを送る"""
        data = {"id": robot_id, "x": float(x), "y": float(y)}
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)  # ロボットの制御を阻害しないようタイムアウトを短めに設定
                s.connect((target_ip, self.port))
                s.sendall(json.dumps(data).encode('utf-8'))
                return True
        except Exception:
            return False

if __name__ == "__main__":
    comm = CoordCommunication()
    print("【PC間 座標共有ツール】")
    print("1: 受信モード (このPCの画面に他のPCからの座標を表示する)")
    print("2: 送信テスト (他のPCにテストデータを送る)")
    
    try:
        choice = input("モードを選択 (1 or 2): ").strip()
        if choice == '1':
            comm.start_server()
        elif choice == '2':
            ip = input("送信先PC of IPアドレスを入力: ").strip()
            print(f"{ip} にテスト送信を開始します...")
            for i in range(5):
                success = comm.send_data(ip, i*10, i*20)
                print(f"送信 {i+1}/5: {'成功' if success else '失敗'}")
                time.sleep(1)
    except KeyboardInterrupt:
        pass