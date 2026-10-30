"""Cozmoアプリの起動用ファイル。設定はapplication/config.pyで変更する。"""

import sys

# Python 3.9 (推奨: 3.9.13) 統一チェック
if sys.version_info[:2] != (3, 9):
    print(
        f"[エラー] 本プロジェクトは Python 3.9 (推奨: 3.9.13) に統一されています。\n"
        f"現在の実行環境: Python {sys.version.split()[0]} ({sys.executable})\n"
        "Python 3.9 環境または 'py -3.9 main.py' で実行してください。"
    )
    sys.exit(1)

from application.controller import run_application


# このファイルを直接実行したときだけ、アプリを起動する。
if __name__ == "__main__":
    run_application()
