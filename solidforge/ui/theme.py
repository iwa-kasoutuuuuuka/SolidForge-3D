"""
SolidForge 3D - UI Theme & Modern Dark Styling
プロフェッショナルなインダストリアル・サイバーテイストのQSSスタイルシート
"""

DARK_THEME_QSS = """
/* 全体基本スタイル */
QWidget {
    background-color: #12141c;
    color: #e2e8f0;
    font-family: "Segoe UI", "Meiryo UI", "Yu Gothic UI", sans-serif;
    font-size: 13px;
    selection-background-color: #00e676;
    selection-color: #000000;
}

/* メインウィンドウ & ダイアログ */
QMainWindow, QDialog {
    background-color: #0f1118;
}

/* グループボックス & カードコンテナ */
QGroupBox {
    background-color: #181b26;
    border: 1px solid #272c3d;
    border-radius: 8px;
    margin-top: 24px;
    padding-top: 14px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 4px;
    padding: 2px 8px;
    color: #00e5ff;
    background-color: #1f2433;
    border-radius: 4px;
    font-size: 12px;
}

/* タブウィジェット */
QTabWidget::pane {
    border: 1px solid #272c3d;
    background-color: #161922;
    border-radius: 6px;
    top: -1px;
}

QTabBar::tab {
    background-color: #13151f;
    color: #94a3b8;
    border: 1px solid #272c3d;
    border-bottom: none;
    padding: 8px 18px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
}

QTabBar::tab:selected {
    background-color: #1e2330;
    color: #00e676;
    border-bottom: 2px solid #00e676;
}

QTabBar::tab:hover:!selected {
    background-color: #1a1e2b;
    color: #e2e8f0;
}

/* ボタン */
QPushButton {
    background-color: #1e2433;
    color: #f1f5f9;
    border: 1px solid #333b52;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #272f44;
    border-color: #00e5ff;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #151a26;
}

QPushButton:disabled {
    background-color: #151720;
    color: #52586b;
    border-color: #1f2330;
}

/* プライマリアクションボタン (Forge 3D Mesh) */
QPushButton#primaryActionBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00c853, stop:1 #00e5ff);
    color: #050b14;
    font-size: 15px;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 12px 24px;
}

QPushButton#primaryActionBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00e676, stop:1 #18ffff);
}

QPushButton#primaryActionBtn:pressed {
    background: #009624;
}

QPushButton#primaryActionBtn:disabled {
    background: #25332e;
    color: #5c786c;
}

/* シャッター撮影ボタン */
QPushButton#shutterBtn {
    background-color: #d50000;
    color: #ffffff;
    font-weight: bold;
    border: 2px solid #ff5252;
    border-radius: 6px;
    padding: 8px 16px;
}

QPushButton#shutterBtn:hover {
    background-color: #ff1744;
    border-color: #ff8a80;
}

/* スライダー */
QSlider::groove:horizontal {
    border: 1px solid #272c3d;
    height: 6px;
    background: #181b26;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #00e676;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #272c3d;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00e5ff;
}

/* ドロップダウン (QComboBox) */
QComboBox {
    background-color: #1a1e2b;
    border: 1px solid #2d3447;
    border-radius: 6px;
    padding: 6px 12px;
    color: #f1f5f9;
    font-weight: 500;
}

QComboBox:hover {
    border-color: #00e5ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #1a1e2b;
    border: 1px solid #2d3447;
    selection-background-color: #00e676;
    selection-color: #000000;
    color: #f1f5f9;
}

/* 入力フィールド */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #161924;
    border: 1px solid #2d3447;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f1f5f9;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #00e676;
}

/* プログレスバー */
QProgressBar {
    background-color: #141721;
    border: 1px solid #272c3d;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    height: 18px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00c853, stop:1 #00e5ff);
    border-radius: 5px;
}

/* スクロールバー */
QScrollBar:vertical {
    background: #10121a;
    width: 10px;
    margin: 0px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #2a3144;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #3f4a66;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ログテキストエリア */
QPlainTextEdit, QTextBrowser {
    background-color: #0b0d12;
    color: #38ef7d;
    border: 1px solid #1f2433;
    border-radius: 6px;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
}

/* ガイダンスバナー */
#guidanceBanner {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a2736, stop:1 #141e2b);
    border: 1px solid #00e5ff;
    border-radius: 8px;
    padding: 10px 16px;
}
"""
