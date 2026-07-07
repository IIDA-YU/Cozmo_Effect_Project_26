import socket
import os

FILE_TO_SYNC = "slam_data.json"
PORT = 9999

def start_sender():
    """送信側（サーバー）: マップデータを提供する"""
    if not os.path.exists(FILE_TO_SYNC):
        print(f"❌ エラー: {FILE_TO_SYNC} が存在しません。先にSLAMを実行してください。")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # ポートの再利用を許可
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    
    # 自分のIPアドレスを取得して表示
    local_ip = socket.gethostbyname(socket.gethostname())
    print(f"📡 送信サーバーを起動しました。")
    print(f"受信側のPCでIPアドレス【 {local_ip} 】を指定して接続してください...")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"🔗 接続されました: {addr}")
            
            with open(FILE_TO_SYNC, 'rb') as f:
                data = f.read()
                client_socket.sendall(data)
                
            print("✅ マップデータの送信が完了しました！")
            client_socket.close()
    except KeyboardInterrupt:
        print("\n⏹️ サーバーを停止しました。")
    finally:
        server_socket.close()

def start_receiver(server_ip):
    """受信側（クライアント）: マップデータをダウンロードする"""
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"🔄 {server_ip}:{PORT} に接続中...")
        client_socket.connect((server_ip, PORT))
        
        with open(FILE_TO_SYNC, 'wb') as f:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                f.write(data)
                
        print(f"🎉 データの受信に成功しました！ {FILE_TO_SYNC} を保存しました。")
        print("CozmoSlamController(use_cached_map=True) で実行してください。")
    except Exception as e:
        print(f"❌ 接続エラー: {e}")
        print("IPアドレスが間違っているか、ファイアウォールでブロックされている可能性があります。")
    finally:
        client_socket.close()

if __name__ == "__main__":
    print("=== SLAMマップ データ同期ツール ===")
    print("1: 【送信】マップを送る (サーバー起動)")
    print("2: 【受信】マップを受け取る (ダウンロード)")
    choice = input("選択してください (1 または 2): ")

    if choice == '1':
        start_sender()
    elif choice == '2':
        ip = input("送信側PCのIPアドレスを入力してください: ")
        start_receiver(ip)
    else:
        print("無効な入力です。")