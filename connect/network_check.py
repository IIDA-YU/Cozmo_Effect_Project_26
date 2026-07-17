#----------------------------------------------------------------------
# network_check.py
# CozmoのWi-Fi接続と本体通信を管理するためのモジュールクラス
#----------------------------------------------------------------------

import time
import subprocess
import contextlib

import pycozmo


# ============================================================
# Wi-Fi 切り替え（Windows / netsh）
# ============================================================

def get_current_ssid():
    """現在PCが接続しているWi-Fiの名前(SSID)を返す。未接続ならNone。"""
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as e:
        print(f"接続エラー：Wi-Fiの確認中に問題が発生しました（{e}）")
        return None

    if result.returncode != 0 or not result.stdout:
        print("接続エラー：Wi-Fi情報を取得できませんでした。Wi-Fi機能がオンか確認してください。")
        return None

    # "SSID" を含み "BSSID" を含まない行が、接続中のSSID
    for line in result.stdout.splitlines():
        if "SSID" in line and "BSSID" not in line and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _connect_to_wifi(ssid):
    """指定SSIDへの接続をPCに命令する。致命的なエラー時は"FATAL"を返す。"""
    print(f"接続中：[{ssid}] へ接続しています...")
    try:
        result = subprocess.run(
            ["netsh", "wlan", "connect", f"name={ssid}"],
            capture_output=True,
            text=True,
            encoding="shift_jis",
            errors="ignore",
        )
    except Exception as e:
        print(f"接続エラー：Wi-Fi接続の処理中に問題が発生しました（{e}）")
        return False

    if result.returncode != 0:
        print("接続エラー：接続コマンドに失敗しました。")
        print("通知：このWi-Fiのパスワードが保存されていない可能性があります。一度手動で接続してください。")
        # パスワード未保存など、リトライしても解決しないエラーの場合は専用の文字列を返す
        return "FATAL"
    return True


def _wait_for_wifi(target_ssid, timeout=5, settle_time=8):
    """target_ssid に繋がるまで最大 timeout 秒待つ。繋がれば settle_time 秒だけ安定待ち。"""
    deadline = time.time() + timeout

    while time.time() < deadline:
        if get_current_ssid() == target_ssid:
            print(f"接続完了：[{target_ssid}] へ接続しました。次のプロセスへ移行中。")
            time.sleep(settle_time)
            return True
        time.sleep(2)

    print(f"失敗：時間内に [{target_ssid}] へ接続できませんでした。")
    return False


def switch_wifi(target_ssid, max_retries=3):
    """
    PCのWi-Fiを target_ssid へ切り替える。
    既に接続済みなら何もしない。成功でTrue、失敗でFalseを返す。
    """
    for attempt in range(1, max_retries + 1):
        current = get_current_ssid()
        if current == target_ssid:
            print(f"接続完了：すでに [{target_ssid}] に接続されています。")
            return True

        # 接続コマンドの発行
        connect_status = _connect_to_wifi(target_ssid)
        
        # 致命的なエラー（パスワード未保存など）の場合はリトライせずに即終了
        if connect_status == "FATAL":
            return False

        if connect_status is True and _wait_for_wifi(target_ssid):
            return True

        if attempt < max_retries:
            print("接続再試行：切り替えに失敗しました。少し待って再試行します...")

    print(f"失敗：[{target_ssid}] への切り替えができませんでした（最大再試行回数に到達）。")
    return False


# ============================================================
# Cozmo本体との通信（pycozmo）
# ============================================================

def _connect_robot(max_retries=3):
    """pycozmoでCozmo本体へ接続し、接続済みクライアントを返す。失敗し続けた場合は例外を送出。"""
    for attempt in range(1, max_retries + 1):
        cli = None
        try:
            cli = pycozmo.Client()
            cli.start()
            cli.connect()
            cli.wait_for_robot()
            return cli
        except Exception as e:
            if cli is not None:
                cli.disconnect()
            print(f"通信エラー（試行 {attempt}/{max_retries}）：{e}")
            if attempt < max_retries:
                print("Wi-Fiが不安定な可能性があります。5秒後に再接続します...")
                time.sleep(5)

    raise RuntimeError("複数回試みましたが、Cozmo本体と通信できませんでした。")


# ============================================================
# 一連の流れをまとめた高レベルAPI
# ============================================================

@contextlib.contextmanager
def cozmo_session(cozmo_ssid, return_ssid):
    """
    Cozmoを使うための一連の流れをまとめたコンテキストマネージャ。

      1. PCのWi-Fiを cozmo_ssid へ切り替える
      2. pycozmoでCozmo本体へ接続する
      3. 接続済みクライアント(cli)を with ブロックへ渡す
      4. 終了時：本体を切断し、Wi-Fiを return_ssid へ戻す（エラーが起きても必ず実行）

    使い方:
        with cozmo_session("Cozmo_XXXX", "教室のSSID") as cli:
            ...  # cli を使った処理

    CozmoのWi-Fiまたは本体への接続に失敗した場合は RuntimeError を送出します。

    ※事前準備：一度PCから手動でCozmoのWi-Fiへ接続し、パスワードを保存しておいてください。
    """
    print("🛜   Cozmoに接続しています")
    # 1. CozmoのWi-Fiへ切り替え
    if not switch_wifi(cozmo_ssid):
        raise RuntimeError(f"CozmoのWi-Fi [{cozmo_ssid}] へ接続できませんでした。")

    # 2. 本体へ接続
    cli = _connect_robot()

    try:
        # 3. 処理本体へ cli を渡す
        yield cli
    finally:
        # 4. 終了処理（本体切断 → Wi-Fi復帰）
        print("\n--- 🛜  元のWi-Fiへ接続しています ---")
        try:
            cli.disconnect()
        except Exception as e:
            print(f"警告：Cozmo本体の切断中に問題が発生しました（{e}）")
        switch_wifi(return_ssid)