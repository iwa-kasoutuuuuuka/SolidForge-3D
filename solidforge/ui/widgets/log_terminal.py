"""
SolidForge 3D - Log Terminal Widget (処理ログ & 進捗ターミナル)
リアルタイムログストリーミング、フェーズ別プログレスバー、CUDA / RTX 5080 ステータス表示
"""

import time
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from solidforge.config import CONFIG
from solidforge.core.pipeline import PIPELINE


class LogTerminalWidget(QWidget):
    """
    再構築処理ログ & ステージ進捗表示用ウィジェット
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # トップヘッダー
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("💻 パイプライン実行ログ & ハードウェア状態")
        title.setStyleSheet("font-weight: bold; color: #00e5ff;")
        header.addWidget(title)

        # リアルタイム GPU/VRAM テレメトリバッジ
        self.gpu_badge = QLabel("⚡ GPU: 初期化中...")
        self.gpu_badge.setStyleSheet(
            "background-color: #1a2f23; color: #00e676; border: 1px solid #00c853; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        header.addWidget(self.gpu_badge)

        self.vram_meter = QLabel("VRAM: -- / -- MB")
        self.vram_meter.setStyleSheet(
            "background-color: #1a2333; color: #00e5ff; border: 1px solid #0091ea; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        header.addWidget(self.vram_meter)

        header.addStretch()

        self.btn_copy = QPushButton("ログコピー")
        self.btn_copy.clicked.connect(self._copy_log)
        header.addWidget(self.btn_copy)

        self.btn_clear = QPushButton("クリア")
        self.btn_clear.clicked.connect(self.clear_log)
        header.addWidget(self.btn_clear)

        layout.addLayout(header)

        # プログレスバー & ステージラベル
        prog_box = QHBoxLayout()
        self.stage_label = QLabel("待機中 (Ready)")
        self.stage_label.setStyleSheet("color: #cbd5e1; font-weight: bold; min-width: 220px;")
        prog_box.addWidget(self.stage_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        prog_box.addWidget(self.progress_bar, stretch=1)

        layout.addLayout(prog_box)

        # ログコンソールテキストエリア
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        self.console.setMinimumHeight(140)
        self.console.setStyleSheet("""
            QPlainTextEdit {
                background-color: #080a0f;
                color: #4ade80;
                border: 1px solid #1e2433;
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        layout.addWidget(self.console, stretch=1)

        # テレメトリ定期更新タイマー (2秒間隔)
        from PySide6.QtCore import QTimer
        from solidforge.core.hardware_optimizer import HARDWARE_OPTIMIZER
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._update_telemetry)
        self._telemetry_timer.start(2000)
        self._update_telemetry()

        self.append_log("SolidForge 3D 極限GPUアクセラレーション初期化完了。パイプライン待機中。")

    def _update_telemetry(self):
        """GPUおよびVRAMテレメトリを更新 (Multi-GPU対応)"""
        try:
            from solidforge.core.hardware_optimizer import HARDWARE_OPTIMIZER
            all_gpus = HARDWARE_OPTIMIZER.list_all_gpus()
            if len(all_gpus) > 1:
                self.gpu_badge.setText(f"⚡ Multi-GPU クラスタ ({len(all_gpus)}基 稼働中)")
                vram_strs = [f"GPU{g.device_id}: {g.vram_used_mb/1024:.1f}/{g.vram_total_mb/1024:.0f}GB ({g.vram_usage_percent}%)" for g in all_gpus]
                self.vram_meter.setText(" | ".join(vram_strs))
            else:
                t = all_gpus[0]
                self.gpu_badge.setText(f"⚡ {t.gpu_name} ({t.cuda_compute_capability})")
                self.vram_meter.setText(f"VRAM: {t.vram_used_mb:,} / {t.vram_total_mb:,} MB ({t.vram_usage_percent}%)")
        except Exception:
            pass

    def _connect_signals(self):
        PIPELINE.log_emitted.connect(self.append_log)
        PIPELINE.progress_updated.connect(self.update_progress)

    def append_log(self, text: str):
        timestamp = time.strftime("[%H:%M:%S]")
        self.console.appendPlainText(f"{timestamp} {text}")
        self.console.moveCursor(QTextCursor.End)

    def update_progress(self, percentage: int, stage_text: str):
        self.progress_bar.setValue(percentage)
        self.stage_label.setText(stage_text)

    def clear_log(self):
        self.console.clear()
        self.progress_bar.setValue(0)
        self.stage_label.setText("待機中 (Ready)")

    def _copy_log(self):
        from PySide6.QtGui import QGuiApplication
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.console.toPlainText())
