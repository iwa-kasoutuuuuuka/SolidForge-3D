"""
SolidForge 3D - Live View Widget
SONY ZV-E10 Live View ストリーミング、リアルタイム特徴点オーバーレイ、構図ガイド、撮影コントロール
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from solidforge.core.camera_manager import CAMERA_MANAGER
from solidforge.core.quality_gate import QUALITY_GATE, QualityEvaluationResult


class LiveViewWidget(QWidget):
    """
    Live View 映像描画 & 撮影コントロール用ウィジェット
    """

    snapshot_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_raw_frame: Optional[np.ndarray] = None
        self._last_eval: Optional[QualityEvaluationResult] = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 映像表示ラベル
        self.video_label = QLabel("カメラ未接続 / 待機中...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #0b0d12; border: 2px dashed #272c3d; border-radius: 8px; color: #64748b; font-size: 14px;"
        )
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(640, 380)
        layout.addWidget(self.video_label, stretch=1)

        # コントロールバー
        control_bar = QHBoxLayout()
        control_bar.setSpacing(10)

        # 特徴点トグル
        self.chk_feature_points = QCheckBox("特徴点 (SIFT/ORB)")
        self.chk_feature_points.setChecked(True)
        control_bar.addWidget(self.chk_feature_points)

        self.chk_grid_guide = QCheckBox("構図グリッド")
        self.chk_grid_guide.setChecked(True)
        control_bar.addWidget(self.chk_grid_guide)

        # オニオンスキン (Ghost Overlay) トグル & 透明度
        self.chk_ghost = QCheckBox("👻 オニオンスキン")
        self.chk_ghost.setChecked(True)
        control_bar.addWidget(self.chk_ghost)

        self.slider_ghost_alpha = QSlider(Qt.Horizontal)
        self.slider_ghost_alpha.setRange(10, 80)
        self.slider_ghost_alpha.setValue(40)
        self.slider_ghost_alpha.setFixedWidth(70)
        self.slider_ghost_alpha.setToolTip("ゴースト透過度 (40% 推奨)")
        control_bar.addWidget(self.slider_ghost_alpha)

        control_bar.addStretch()

        # カメラ設定 & スマホ接続ボタン
        self.btn_camera_select = QPushButton("⚙️ カメラ選択 / スマホ接続")
        self.btn_camera_select.clicked.connect(self._open_camera_select_dialog)
        control_bar.addWidget(self.btn_camera_select)

        # カメラ接続/切断トグルボタン
        self.btn_camera_toggle = QPushButton("接続切断")
        self.btn_camera_toggle.clicked.connect(self._toggle_camera_connection)
        control_bar.addWidget(self.btn_camera_toggle)

        # シャッターボタン
        self.btn_shutter = QPushButton("📷 1枚撮影 (Quality Gate判定)")
        self.btn_shutter.setObjectName("shutterBtn")
        self.btn_shutter.clicked.connect(self._on_shutter_clicked)
        control_bar.addWidget(self.btn_shutter)

        layout.addLayout(control_bar)

    def _connect_signals(self):
        CAMERA_MANAGER.frame_ready.connect(self._on_frame_received)
        CAMERA_MANAGER.connection_status_changed.connect(self._on_camera_status_changed)

    def _open_camera_select_dialog(self):
        from solidforge.ui.widgets.camera_select_dialog import CameraSelectDialog
        dialog = CameraSelectDialog(self)
        dialog.exec()

    def _toggle_camera_connection(self):
        if CAMERA_MANAGER.is_connected:
            CAMERA_MANAGER.disconnect()
        else:
            CAMERA_MANAGER.connect_sony_sdk()

    def _on_camera_status_changed(self, connected: bool, device_name: str):
        if connected:
            self.btn_camera_toggle.setText("切断")
            self.btn_camera_toggle.setStyleSheet("border-color: #ff5252; color: #ff5252;")
            self.btn_shutter.setEnabled(True)
        else:
            self.btn_camera_toggle.setText("接続")
            self.btn_camera_toggle.setStyleSheet("")
            self.btn_shutter.setEnabled(False)
            self.video_label.setText("カメラ未接続 / 待機中...")
            self.video_label.setPixmap(QPixmap())

    def _on_shutter_clicked(self):
        success, filepath, eval_res = CAMERA_MANAGER.capture_snapshot()
        self.snapshot_requested.emit()

    def _on_frame_received(self, raw_frame: np.ndarray, eval_res: QualityEvaluationResult):
        self._last_raw_frame = raw_frame
        self._last_eval = eval_res

        # オーバーレイ描画
        draw_grid = self.chk_grid_guide.isChecked()
        draw_pts = self.chk_feature_points.isChecked()
        ghost_frame = CAMERA_MANAGER.get_last_captured_frame() if self.chk_ghost.isChecked() else None
        ghost_alpha = float(self.slider_ghost_alpha.value()) / 100.0

        display_frame = QUALITY_GATE.draw_quality_overlay(
            raw_frame,
            eval_result=eval_res,
            draw_grid=draw_grid,
            draw_points=draw_pts,
            ghost_frame=ghost_frame,
            ghost_alpha=ghost_alpha,
        )

        # BGR -> RGB変換 & QImage化
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # アスペクト比を維持して拡大縮小
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
