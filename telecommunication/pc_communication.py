# telecommunication/pc_communication.py
import ctypes
from ctypes import wintypes
import json
import socket
import threading
import time


# =========================================================
# Python 3.9 (Windows) 向け Winsock2 Bluetooth (AF_BTH) ラッパー
# 標準の socket に AF_BTH が無くても ctypes 経由で直接 Bluetooth 通信を行う
# =========================================================
AF_BTH = 32
BTHPROTO_RFCOMM = 3
INVALID_SOCKET = (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1
FIONBIO = 0x8004667E


class SOCKADDR_BTH(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("addressFamily", ctypes.c_ushort),
        ("btAddr", ctypes.c_ulonglong),
        ("serviceClassId", ctypes.c_byte * 16),
        ("port", ctypes.c_ulong),
    ]


class _WinBluetoothHelper:
    """Winsock2 (ws2_32.dll) を使って Python 3.9 でも RFCOMM ソケットを扱うクラス"""

    def __init__(self):
        self.ws2 = ctypes.WinDLL("ws2_32", use_last_error=True)

        # 関数シグネチャ定義 (64bit/32bit Windows 両対応)
        self.ws2.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.ws2.socket.restype = ctypes.c_size_t

        self.ws2.bind.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(SOCKADDR_BTH),
            ctypes.c_int,
        ]
        self.ws2.bind.restype = ctypes.c_int

        self.ws2.listen.argtypes = [ctypes.c_size_t, ctypes.c_int]
        self.ws2.listen.restype = ctypes.c_int

        self.ws2.accept.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(SOCKADDR_BTH),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.ws2.accept.restype = ctypes.c_size_t

        self.ws2.connect.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(SOCKADDR_BTH),
            ctypes.c_int,
        ]
        self.ws2.connect.restype = ctypes.c_int

        self.ws2.getsockname.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(SOCKADDR_BTH),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.ws2.getsockname.restype = ctypes.c_int

        self.ws2.closesocket.argtypes = [ctypes.c_size_t]
        self.ws2.closesocket.restype = ctypes.c_int

        self.ws2.ioctlsocket.argtypes = [
            ctypes.c_size_t,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        self.ws2.ioctlsocket.restype = ctypes.c_int

    @staticmethod
    def str_to_btaddr(addr_str):
        clean = addr_str.strip().replace("-", ":").replace(":", "")
        return int(clean, 16)

    @staticmethod
    def btaddr_to_str(bt_int):
        hex_str = f"{bt_int:012X}"
        return ":".join(hex_str[i : i + 2] for i in range(0, 12, 2))

    def get_local_bt_mac_from_os(self):
        """ソケットが 00:00:00:00:00:00 を返した場合の補完用"""
        try:
            import subprocess
            res = subprocess.run(
                ["getmac", "/v", "/fo", "csv"],
                capture_output=True,
                text=True,
                encoding="cp932",
                errors="ignore",
            )
            for line in res.stdout.splitlines():
                if "Bluetooth" in line:
                    for part in line.replace('"', "").split(","):
                        p = part.strip()
                        if len(p) == 17 and p.count("-") == 5:
                            return p.replace("-", ":").upper()
        except Exception:
            pass
        return "00:00:00:00:00:00"

    def create_server(self, channel):
        handle = self.ws2.socket(AF_BTH, socket.SOCK_STREAM, BTHPROTO_RFCOMM)
        if handle == INVALID_SOCKET:
            err = self.ws2.WSAGetLastError()
            raise OSError(f"Bluetooth socket作成失敗 (WSAError: {err})")

        addr = SOCKADDR_BTH()
        addr.addressFamily = AF_BTH
        addr.btAddr = 0  # BDADDR_ANY
        addr.port = channel

        if self.ws2.bind(handle, ctypes.byref(addr), ctypes.sizeof(addr)) != 0:
            err = self.ws2.WSAGetLastError()
            self.ws2.closesocket(handle)
            raise OSError(f"Bluetooth bind失敗 (Ch:{channel}, WSAError: {err})")

        if self.ws2.listen(handle, 5) != 0:
            err = self.ws2.WSAGetLastError()
            self.ws2.closesocket(handle)
            raise OSError(f"Bluetooth listen失敗 (WSAError: {err})")

        # ノンブロッキングモードに設定（1秒タイムアウトポーリング用）
        mode = ctypes.c_ulong(1)
        self.ws2.ioctlsocket(handle, FIONBIO, ctypes.byref(mode))

        namelen = ctypes.c_int(ctypes.sizeof(addr))
        self.ws2.getsockname(handle, ctypes.byref(addr), ctypes.byref(namelen))
        bt_str = self.btaddr_to_str(addr.btAddr)
        if bt_str == "00:00:00:00:00:00":
            bt_str = self.get_local_bt_mac_from_os()

        return handle, bt_str

    def accept_nonblocking(self, server_handle):
        client_addr = SOCKADDR_BTH()
        addr_len = ctypes.c_int(ctypes.sizeof(client_addr))
        client_handle = self.ws2.accept(
            server_handle, ctypes.byref(client_addr), ctypes.byref(addr_len)
        )
        if client_handle == INVALID_SOCKET:
            err = self.ws2.WSAGetLastError()
            if err == 10035:  # WSAEWOULDBLOCK (着信なし)
                return None, None
            raise OSError(f"Bluetooth acceptエラー (WSAError: {err})")

        # 受け入れたクライアントソケットをブロッキングに戻し、Python標準socketに変換
        mode = ctypes.c_ulong(0)
        self.ws2.ioctlsocket(client_handle, FIONBIO, ctypes.byref(mode))
        py_sock = socket.socket(fileno=client_handle)
        sender_bt = self.btaddr_to_str(client_addr.btAddr)
        return py_sock, sender_bt

    def connect_client(self, target_bt_str, channel, timeout=2.5):
        handle = self.ws2.socket(AF_BTH, socket.SOCK_STREAM, BTHPROTO_RFCOMM)
        if handle == INVALID_SOCKET:
            err = self.ws2.WSAGetLastError()
            raise OSError(f"Bluetooth socket作成失敗 (WSAError: {err})")

        addr = SOCKADDR_BTH()
        addr.addressFamily = AF_BTH
        addr.btAddr = self.str_to_btaddr(target_bt_str)
        addr.port = channel

        if self.ws2.connect(handle, ctypes.byref(addr), ctypes.sizeof(addr)) != 0:
            err = self.ws2.WSAGetLastError()
            self.ws2.closesocket(handle)
            raise OSError(f"Bluetooth接続失敗 -> {target_bt_str} (WSAError: {err})")

        py_sock = socket.socket(fileno=handle)
        py_sock.settimeout(timeout)
        return py_sock

    def close_handle(self, handle):
        if handle and handle != INVALID_SOCKET:
            self.ws2.closesocket(handle)


_BT_HELPER = _WinBluetoothHelper()


class CoordCommunication:
    _shared_cli = None
    _shared_local_info_cb = None
    _shared_robot_id = None
    _local_bt_addr = None

    @classmethod
    def register_robot_context(cls, cli=None, robot_id=None, get_local_info_cb=None):
        if cli is not None:
            cls._shared_cli = cli
        if robot_id and robot_id != "Cozmo-01":
            cls._shared_robot_id = robot_id
        if get_local_info_cb is not None:
            cls._shared_local_info_cb = get_local_info_cb

    def __init__(
        self,
        port=50007,
        bt_channel=22,
        robot_id="Cozmo-01",
        get_local_info_cb=None,
        on_receive_info_cb=None,
    ):
        self.port = port
        self.bt_channel = bt_channel
        self.robot_id = robot_id
        self.get_local_info_cb = get_local_info_cb
        self.on_receive_info_cb = on_receive_info_cb
        self.running = True

        self.register_robot_context(
            robot_id=robot_id, get_local_info_cb=get_local_info_cb
        )

        self.peer_states = {}
        self._lock = threading.Lock()

    @staticmethod
    def _is_bluetooth_addr(addr_str):
        if not isinstance(addr_str, str):
            return False
        parts = addr_str.strip().replace("-", ":").split(":")
        return len(parts) == 6 and all(len(p) == 2 for p in parts)

    @staticmethod
    def _normalize_bt_addr(addr_str):
        return addr_str.strip().replace("-", ":").upper()

    def get_local_ip(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def _get_current_xy(self):
        cb = self.get_local_info_cb or CoordCommunication._shared_local_info_cb
        if callable(cb):
            try:
                cb_data = cb()
                if isinstance(cb_data, dict) and "x" in cb_data and "y" in cb_data:
                    return float(cb_data["x"]), float(cb_data["y"]), cb_data
            except Exception as e:
                print(f"[通信] 自機データ取得エラー: {e}")

        cli = CoordCommunication._shared_cli
        if cli is not None and getattr(cli, "pose", None) is not None:
            try:
                return float(cli.pose.position.x), float(cli.pose.position.y), {}
            except Exception:
                pass

        return 0.0, 0.0, {}

    def _build_my_packet(self, x=None, y=None, extra_data=None, mode="encounter"):
        effective_id = (
            self.robot_id
            if self.robot_id != "Cozmo-01"
            else (CoordCommunication._shared_robot_id or self.robot_id)
        )

        live_x, live_y, cb_dict = self._get_current_xy()
        final_x = float(x) if x is not None else live_x
        final_y = float(y) if y is not None else live_y

        return {
            "id": cb_dict.get("id", effective_id),
            "mode": mode,
            "x": final_x,
            "y": final_y,
            "timestamp": time.time(),
            "extra": extra_data or cb_dict.get("extra", {}),
        }

    def _record_peer_packet(self, msg, addr_label, via_label="TCP"):
        peer_id = msg.get("id", f"Unknown({addr_label})")
        mode = msg.get("mode", "encounter")

        with self._lock:
            self.peer_states[peer_id] = msg

        if mode != "encounter":
            return

        x = float(msg.get("x", 0.0))
        y = float(msg.get("y", 0.0))
        extra = msg.get("extra", {})
        print(
            f"🔄 [{time.strftime('%H:%M:%S')}] すれ違い相互通信成立({via_label})！ "
            f"相手: {peer_id} ({addr_label}) | 受信座標: X={x:>6.1f}, Y={y:>6.1f}"
            + (f" | 追加情報: {extra}" if extra else "")
        )

        if callable(self.on_receive_info_cb):
            try:
                self.on_receive_info_cb(peer_id, msg)
            except Exception as e:
                print(f"[通信] 受信コールバックエラー: {e}")

    def start_server(self):
        local_ip = self.get_local_ip()
        print(f"\n=== 📡 すれ違い相互通信サーバー起動 (Bluetooth & TCP両対応) ===")
        print(
            f"自機ID: {self.robot_id} / TCP: {local_ip}:{self.port} / BT Channel: {self.bt_channel}"
        )

        # ctypes Winsock2 ラッパー経由で Bluetooth サーバーを別スレッド起動
        threading.Thread(target=self._start_bluetooth_server, daemon=True).start()
        time.sleep(0.2)
        print("----------------------------------------")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", self.port))
            s.listen(5)

            while self.running:
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                    threading.Thread(
                        target=self._handle_client, args=(conn, addr[0]), daemon=True
                    ).start()
                except socket.timeout:
                    continue
                except KeyboardInterrupt:
                    print("\nサーバーを停止します。")
                    break
                except Exception as e:
                    if self.running:
                        print(f"TCP受信エラー: {e}")

    def _start_bluetooth_server(self):
        server_handle = None
        try:
            server_handle, local_bt = _BT_HELPER.create_server(self.bt_channel)
            CoordCommunication._local_bt_addr = local_bt
            print(
                f"🔵 Bluetooth待ち受け開始！ このPCのBTアドレス: {local_bt} (Ch:{self.bt_channel})"
            )

            while self.running:
                conn_sock, sender_bt = _BT_HELPER.accept_nonblocking(server_handle)
                if conn_sock is not None:
                    threading.Thread(
                        target=self._handle_client,
                        args=(conn_sock, sender_bt),
                        daemon=True,
                    ).start()
                else:
                    time.sleep(0.05)
        except Exception as e:
            print(
                f"⚠️ Bluetoothサーバーを起動できませんでした (BluetoothがONか確認してください): {e}"
            )
        finally:
            _BT_HELPER.close_handle(server_handle)

    def _handle_client(self, conn, sender_addr):
        with conn:
            conn.settimeout(1.5)
            try:
                data = conn.recv(4096)
                if not data:
                    return
                msg = json.loads(data.decode("utf-8"))

                via = "Bluetooth" if self._is_bluetooth_addr(sender_addr) else "TCP"
                if sender_addr != "127.0.0.1":
                    self._record_peer_packet(msg, sender_addr, via_label=via)
                else:
                    peer_id = msg.get("id", f"Unknown({sender_addr})")
                    with self._lock:
                        self.peer_states[peer_id] = msg

                if msg.get("mode", "encounter") == "encounter":
                    my_reply = self._build_my_packet(mode="encounter")
                    if (
                        sender_addr == "127.0.0.1"
                        and my_reply["x"] == 0.0
                        and my_reply["y"] == 0.0
                    ):
                        my_reply["x"] = float(msg.get("x", 0.0))
                        my_reply["y"] = float(msg.get("y", 0.0))
                    conn.sendall(json.dumps(my_reply).encode("utf-8"))
            except Exception as e:
                print(f"応答処理エラー ({sender_addr}): {e}")

    def exchange_data(self, target_ip, x, y, robot_id=None, extra_data=None):
        send_id = robot_id or self.robot_id
        my_packet = self._build_my_packet(
            x=x, y=y, extra_data=extra_data, mode="encounter"
        )
        my_packet["id"] = send_id

        is_bt = self._is_bluetooth_addr(target_ip)
        bt_addr = self._normalize_bt_addr(target_ip) if is_bt else ""

        # 自分自身のBTアドレスが指定されている場合（1台テスト時）は内部ループバックへ自動切り替え
        is_self_bt = is_bt and (bt_addr == CoordCommunication._local_bt_addr)

        try:
            if is_bt and not is_self_bt:
                s = _BT_HELPER.connect_client(bt_addr, self.bt_channel, timeout=2.5)
                label_addr = bt_addr
                via_label = "Bluetooth"
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.8)
                s.connect(("127.0.0.1" if is_self_bt else target_ip, self.port))
                label_addr = bt_addr if is_self_bt else target_ip
                via_label = "BT自己ループバック" if is_self_bt else "TCP"

            with s:
                s.sendall(json.dumps(my_packet).encode("utf-8"))
                reply_raw = s.recv(4096)
                if reply_raw:
                    peer_msg = json.loads(reply_raw.decode("utf-8"))
                    if (
                        (label_addr == "127.0.0.1" or is_self_bt)
                        and float(peer_msg.get("x", 0.0)) == 0.0
                        and float(peer_msg.get("y", 0.0)) == 0.0
                    ):
                        peer_msg["x"] = float(x)
                        peer_msg["y"] = float(y)
                    self._record_peer_packet(peer_msg, label_addr, via_label=via_label)
                    return peer_msg
                return True
        except Exception as e:
            if is_bt:
                print(f"⚠️ Bluetooth通信失敗 ({target_ip}): {e}")
            return None

    def send_data(self, target_ip, x, y, robot_id="Cozmo-01"):
        packet = {
            "id": robot_id,
            "mode": "periodic",
            "x": float(x),
            "y": float(y),
            "timestamp": time.time(),
        }
        is_bt = self._is_bluetooth_addr(target_ip)
        bt_addr = self._normalize_bt_addr(target_ip) if is_bt else ""
        is_self_bt = is_bt and (bt_addr == CoordCommunication._local_bt_addr)

        try:
            if is_bt and not is_self_bt:
                s = _BT_HELPER.connect_client(bt_addr, self.bt_channel, timeout=1.0)
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.3)
                s.connect(("127.0.0.1" if is_self_bt else target_ip, self.port))

            with s:
                s.sendall(json.dumps(packet).encode("utf-8"))
                return True
        except Exception:
            return False