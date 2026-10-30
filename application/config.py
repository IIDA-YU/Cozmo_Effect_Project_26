"""接続先・機体名・カメラなど、アプリ全体で共有する設定。"""

from dataclasses import dataclass, field
from pathlib import Path

from algorithm.mapping.visual_slam import DEFAULT_CHESS_MODEL_PATH
from camera.camera_movie import CameraSettings


@dataclass
class ApplicationConfig:
    """設定を1つにまとめ、各機能へ渡すためのクラス。"""

    cozmo_ssid: str
    return_ssid: str
    robot_id: str = "Cozmo-A"
    target_pc_ip: str = "172.19.16.1"
    listen_port: int = 50007

    # モデルの保存場所はvisual_slam.pyの定義を共用する。
    chess_model_path: Path = DEFAULT_CHESS_MODEL_PATH

    # 設定を作るたびに、独立したカメラ設定と辞書を用意する。
    camera_settings: CameraSettings = field(
        default_factory=lambda: CameraSettings(
            exposure_ms=5,
            gain=1.0,
            auto_exposure=False,
        )
    )
    target_ip_dict: dict = field(default_factory=dict)


# ---------------------------------------------------------
# 機体・PCごとに変更する設定
# ---------------------------------------------------------
CONFIG = ApplicationConfig(
    # cozmo_ssid="Cozmo_613F59",  # 接続するCozmoのWi-Fi名
    # return_ssid="ctc-g-171bde",       # 終了後に戻る学校・自宅のWi-Fi名
    cozmo_ssid="Cozmo_613F59",  # 接続するCozmoのWi-Fi名
    return_ssid="R0347",       # 終了後に戻る学校・自宅のWi-Fi名
    robot_id="Cozmo-A",        # 通信で使用する自分の機体名
    target_pc_ip="127.0.0.1",  # 自分のPCへのテスト用。他PCへ送る場合は変更する
    listen_port=50007,         # 座標を受信するポート番号

    # SLAMと自律走行で同じカメラ設定を使う。
    # 初期値は、白飛びを抑えるための手動露出5ms・ゲイン1.0。
    camera_settings=CameraSettings(
        exposure_ms=5,
        gain=1.0,
        auto_exposure=False,
    ),

    # ArUcoマーカーID → その機体を操作するPCのIPアドレス
    target_ip_dict={
        1: "127.0.0.1",
        2: "172.19.16.1",
        3: "192.168.1.12",
        4: "192.168.1.13",
        5: "192.168.1.14",
    },
)
