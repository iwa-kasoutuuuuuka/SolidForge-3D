"""
SolidForge 3D - アプリケーション起動エントリポイント
Windows Desktop Application for Photogrammetry & 3D Printing
"""

import sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from solidforge.config import CONFIG
from solidforge.ui.main_window import MainWindow


def main():
    # 高DPIスケーリングの有効化
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Windows タスクバーでのアプリアイコン個別表示 (AppUserModelID)
    try:
        import ctypes
        myappid = "solidforge.3d.photogrammetry.v1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(CONFIG.app_name)
    app.setApplicationVersion(CONFIG.version)

    # アプリアイコンの設定
    icon_path = Path(__file__).parent / "assets" / "icon.png"
    if icon_path.exists():
        app_icon = QIcon(str(icon_path))
        app.setWindowIcon(app_icon)

    window = MainWindow(config=CONFIG)
    if icon_path.exists():
        window.setWindowIcon(QIcon(str(icon_path)))
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
