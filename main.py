"""Cozmoアプリの起動用ファイル。設定はapplication/config.pyで変更する。"""

from application.controller import run_application


# このファイルを直接実行したときだけ、アプリを起動する。
if __name__ == "__main__":
    run_application()
