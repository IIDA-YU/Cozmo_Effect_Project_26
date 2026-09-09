"""SLAMのJSONをTCPで転送する。送信元と同じ座標原点・向きで使用すること。

例: python map_sync.py --file slam_data.json
別の場所に保存した場合は --file でSLAMの json_filename と同じパスを指定する。
同じネットワークの信頼できるPC間で使用する（認証・暗号化は行わない）。
"""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import tempfile

FILE_TO_SYNC = "slam_data.json"
PORT = 9999
MAX_MAP_BYTES = 2 * 1024 * 1024
SOCKET_TIMEOUT = 15.0


def validate_map_bytes(data):
    """途中で切れたJSONや不正な座標で、手元のマップを壊さない。"""
    if not data or len(data) > MAX_MAP_BYTES:
        raise ValueError("マップが空、またはサイズ上限(2MiB)を超えています")
    document = json.loads(data.decode("utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("マップの形式が不正です")
    landmarks = document.get("landmarks")
    if not isinstance(landmarks, dict) or not 1 <= len(landmarks) <= 6:
        raise ValueError("ランドマークが1〜6個のマップを指定してください")
    ids = set()
    for key, point in landmarks.items():
        landmark_id = int(key)
        if landmark_id < 0 or landmark_id in ids:
            raise ValueError("ランドマークIDが不正です")
        ids.add(landmark_id)
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           and math.isfinite(v) for v in point)):
            raise ValueError("ランドマークの座標が不正です")
    return document


def _save_atomically(path, data):
    """検証済みの全データを書き終えてから保存先を置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = f.name
            f.write(data)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def start_sender(filename=None, port=PORT):
    """送信側: 各接続時点で保存済みのマップを読み、接続終了を終端として送る。"""
    path = Path(filename or FILE_TO_SYNC).expanduser().resolve()
    if not path.is_file():
        print(f"❌ マップが見つかりません: {path}。SLAMを終了して保存してください。")
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen(2)
            print(f"📡 送信待機中: {path} (TCP {port})")
            print("受信側では送信PCのLAN側IPv4アドレスを指定してください。")
            while True:
                client, address = server.accept()
                with client:
                    client.settimeout(SOCKET_TIMEOUT)
                    try:
                        with path.open("rb") as f:
                            data = f.read(MAX_MAP_BYTES + 1)
                        document = validate_map_bytes(data)
                        client.sendall(data)
                        print(f"✅ {address[0]}へ駒{len(document['landmarks'])}個のマップを送信しました")
                    except (OSError, ValueError) as error:
                        print(f"❌ 送信失敗: {error}")
    except KeyboardInterrupt:
        print("\n⏹️ 送信を終了しました")
        return True
    except OSError as error:
        print(f"❌ 送信サーバーを起動できません: {error}")
        return False


def start_receiver(server_ip, filename=None, port=PORT):
    """受信側: 全データ受信・JSON検証が成功したときだけマップを更新する。"""
    path = Path(filename or FILE_TO_SYNC).expanduser().resolve()
    try:
        data = bytearray()
        with socket.create_connection((server_ip.strip(), port), timeout=SOCKET_TIMEOUT) as client:
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > MAX_MAP_BYTES:
                    raise ValueError("受信サイズが上限(2MiB)を超えました")
        document = validate_map_bytes(data)
        _save_atomically(path, data)
        print(f"✅ 駒{len(document['landmarks'])}個のマップを保存しました: {path}")
        print("use_cached_map=True と、同じ json_filename を指定してSLAMを起動してください。")
        print("保存時と同じ原点・開始位置・向きが必要です。PC間の座標合わせはこの転送では行いません。")
        return True
    except (OSError, ValueError) as error:
        print(f"❌ マップ受信失敗: {error}。既存のマップは変更していません。")
        return False


def main():
    parser = argparse.ArgumentParser(description="SLAMマップの送受信")
    parser.add_argument("--file", default=FILE_TO_SYNC, help="SLAMと同じJSONパス")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    print("=== SLAMマップ データ同期ツール ===\n1: 送信\n2: 受信")
    choice = input("選択 (1 または 2): ").strip()
    if choice == "1":
        start_sender(args.file, args.port)
    elif choice == "2":
        start_receiver(input("送信側PCのIPアドレス: "), args.file, args.port)
    else:
        print("無効な入力です")


if __name__ == "__main__":
    main()
