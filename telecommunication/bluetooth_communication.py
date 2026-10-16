"""Bluetooth SPPの仮想COMポート。TCP/IPは使わない。

OSでペアリング・SPPポート作成を済ませておく。1組につき双方向リンク1本。
読取と書込を分け、大きい双方向送信の相互待ちを避ける。
"""
import json
import queue
import threading
import time

MAX_MESSAGE_BYTES = 1024 * 1024


def require_pyserial():
    """Bluetooth SPP に必要な pyserial を確認して返す。

    Python ではパッケージ名は ``pyserial`` ですが、コードからは
    ``serial`` という名前で読み込みます。未導入のまま実験を始めると
    分かりにくい ModuleNotFoundError になるため、ここで説明付きで止めます。
    """
    try:
        import serial
    except ModuleNotFoundError as error:
        if error.name != "serial":
            raise
        raise RuntimeError(
            "Bluetooth通信に必要な pyserial がインストールされていません。\n"
            "コマンドプロンプトで次を実行してください。\n"
            "  python -m pip install pyserial\n"
            "インストール後、もう一度 python main.py を実行してください。"
        ) from None

    # PyPI には pyserial とは別に 'serial' という名前のパッケージもある。
    # 間違った方が入っている場合も、実験開始前に分かるようにする。
    if not hasattr(serial, "Serial") or not hasattr(serial, "SerialException"):
        raise RuntimeError(
            "pyserial ではない 'serial' パッケージが読み込まれています。\n"
            "次を順番に実行して修正してください。\n"
            "  python -m pip uninstall serial\n"
            "  python -m pip install --upgrade pyserial"
        )

    return serial


class LineDecoder:
    def __init__(self):
        self.buffer = bytearray()
        self.discard = False

    def feed(self, chunk):
        result = []
        for part in chunk.splitlines(keepends=True):
            if not self.discard:
                self.buffer.extend(part)
                if len(self.buffer)>MAX_MESSAGE_BYTES:
                    self.buffer.clear()
                    self.discard = True
            if part.endswith(b'\n'):
                if not self.discard:
                    try:
                        obj = json.loads(self.buffer.decode('utf-8'))
                        if isinstance(obj,dict):
                            result.append(obj)
                    except (ValueError,UnicodeError):
                        pass
                self.buffer.clear()
                self.discard = False
        return result


class SerialLink:
    def __init__(self, peer, port, incoming, status):
        self.peer, self.port = peer,port
        self.incoming, self.status = incoming,status
        self.outgoing = queue.Queue(maxsize=8)
        self.stop_event = threading.Event()
        self.connection = None
        self.threads = []

    def start(self):
        # 実機を使うときだけ pyserial を読み込む。
        # 未導入なら、インストール方法を含む分かりやすいエラーを出す。
        self.serial = require_pyserial()
        for fn in (self._reader,self._writer):
            t = threading.Thread(target=fn,daemon=True)
            self.threads.append(t)
            t.start()

    def send(self, message):
        data = json.dumps(message,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(data)>MAX_MESSAGE_BYTES:
            raise ValueError('Bluetooth送信データが1MiBを超えています')
        try:
            self.outgoing.put_nowait((time.monotonic(),data))
            return True
        except queue.Full:
            return False

    def _reader(self):
        while not self.stop_event.is_set():
            conn = None
            try:
                conn = self.serial.Serial(self.port,115200,timeout=0.1,write_timeout=5)
                self.connection = conn
                self.status[self.peer] = 'connected'
                decoder = LineDecoder()
                while not self.stop_event.is_set() and self.connection is conn:
                    for msg in decoder.feed(conn.read(4096)):
                        try:
                            self.incoming.put_nowait((self.peer,msg,time.monotonic()))
                        except queue.Full:
                            self.status[self.peer] = 'receive queue full'
            except (OSError,self.serial.SerialException) as error:
                self.status[self.peer] = str(error)
            finally:
                if self.connection is conn:
                    self.connection = None
                if conn is not None:
                    conn.close()
            self.stop_event.wait(1.0)

    def _writer(self):
        while not self.stop_event.is_set():
            if self.connection is None:
                self.stop_event.wait(0.1)
                continue
            try:
                created,data = self.outgoing.get(timeout=0.1)
            except queue.Empty:
                continue
            if time.monotonic()-created > 8:
                continue  # 切断中に古くなった遭遇データは送らない
            conn = self.connection
            if conn is None:
                continue
            try:
                # 前の送信が途中切断していた場合も、先頭改行で同期を戻す。
                payload = b'\n'+data
                while payload and not self.stop_event.is_set():
                    n = conn.write(payload)
                    if not n:
                        raise OSError('serial write returned zero')
                    payload = payload[n:]
            except (OSError,self.serial.SerialException) as error:
                self.status[self.peer] = str(error)
                if self.connection is conn:
                    self.connection = None
                conn.close()

    def stop(self):
        self.stop_event.set()
        if self.connection is not None:
            self.connection.close()
        for t in self.threads:
            t.join(timeout=1)


class BluetoothNetwork:
    def __init__(self, ports):
        self.incoming = queue.Queue(maxsize=64)
        self.status = {}
        self.links = {peer:SerialLink(peer,port,self.incoming,self.status) for peer,port in ports.items()}

    def start(self):
        for link in self.links.values():
            link.start()

    def send(self, peer, message):
        return peer in self.links and self.links[peer].send(message)

    def receive(self):
        messages = []
        while len(messages)<32:
            try:
                messages.append(self.incoming.get_nowait())
            except queue.Empty:
                break
        return messages

    def stop(self):
        for link in self.links.values():
            link.stop()
