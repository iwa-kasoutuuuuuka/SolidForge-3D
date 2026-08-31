"""
SolidForge 3D - 3D Camera Trajectory & Coverage Viewport Widget
リアルタイム3Dカメラ姿勢推定・360度カバレッジ可視化 & 未撮影ギャップ（死角）警告
"""

import math
from typing import List, Optional
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from solidforge.core.trajectory_analyzer import (
    TRAJECTORY_ANALYZER,
    CameraPose,
    CoverageAnalysisResult,
)


class RadarCanvas(QWidget):
    """360度カメラ軌跡 & ギャップ可視化キャンバス"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.poses: List[CameraPose] = []
        self.analysis_result: Optional[CoverageAnalysisResult] = None
        self.view_elevation = 45.0  # 俯瞰アングル
        self.view_rotation = 0.0

    def set_data(self, poses: List[CameraPose], result: CoverageAnalysisResult):
        self.poses = poses
        self.analysis_result = result
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2.0, h / 2.0
        radius = min(w, h) * 0.38

        # 背景グラデーション
        bg_grad = QRadialGradient(cx, cy, radius * 1.3)
        bg_grad.setColorAt(0.0, QColor(18, 22, 32))
        bg_grad.setColorAt(1.0, QColor(11, 13, 18))
        painter.fillRect(0, 0, w, h, bg_grad)

        # 1. 360度方位グリッド & 同心円
        grid_pen = QPen(QColor(40, 50, 70), 1, Qt.DashLine)
        painter.setPen(grid_pen)
        for r_factor in [0.33, 0.66, 1.0]:
            painter.drawEllipse(QPointF(cx, cy), radius * r_factor, radius * r_factor * 0.6)

        # 4方位軸
        axis_pen = QPen(QColor(50, 65, 90), 1)
        painter.setPen(axis_pen)
        painter.drawLine(int(cx - radius), int(cy), int(cx + radius), int(cy))
        painter.drawLine(int(cx), int(cy - radius * 0.6), int(cx), int(cy + radius * 0.6))

        # 方位テキスト
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor(100, 120, 160))
        painter.drawText(int(cx + radius + 8), int(cy + 4), "0° (E)")
        painter.drawText(int(cx - radius - 45), int(cy + 4), "180° (W)")
        painter.drawText(int(cx - 15), int(cy - radius * 0.6 - 8), "90° (N)")
        painter.drawText(int(cx - 15), int(cy + radius * 0.6 + 18), "270° (S)")

        # 2. 未撮影ギャップ領域の扇形ハイライト (赤色半透明)
        if self.analysis_result and self.analysis_result.missing_angle_ranges:
            for start_deg, end_deg in self.analysis_result.missing_angle_ranges:
                span = (end_deg - start_deg) % 360.0
                if span <= 0:
                    span = 360.0

                gap_path = QPainterPath()
                gap_path.moveTo(cx, cy)
                gap_rect = QRectF(cx - radius, cy - radius * 0.6, radius * 2, radius * 1.2)
                # QPainterの角度は反時計回り 1/16度単位 (3時の位置が0度)
                gap_path.arcTo(gap_rect, -start_deg, -span)
                gap_path.closeSubpath()

                painter.fillPath(gap_path, QBrush(QColor(255, 50, 50, 50)))
                painter.strokePath(gap_path, QPen(QColor(255, 80, 80, 140), 1.5, Qt.DashDotLine))

        # 3. 中心オブジェクト (被写体)
        obj_grad = QRadialGradient(cx, cy, 18)
        obj_grad.setColorAt(0.0, QColor(0, 229, 255))
        obj_grad.setColorAt(1.0, QColor(0, 120, 180))
        painter.setBrush(QBrush(obj_grad))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(cx, cy), 16, 10)

        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(int(cx - 11), int(cy + 3), "OBJECT")

        # 4. カメラ位置 (緑色ドット & 視線錐台)
        cam_pen = QPen(QColor(0, 230, 118), 2)
        frustum_pen = QPen(QColor(0, 230, 118, 90), 1)

        for p in self.poses:
            rad = math.radians(p.azimuth_deg)
            px = cx + radius * math.cos(rad)
            py = cy + radius * 0.6 * math.sin(rad)

            # 視線ベクトル線 (中心へ)
            painter.setPen(frustum_pen)
            painter.drawLine(int(px), int(py), int(cx), int(cy))

            # カメラインジケータ
            painter.setPen(cam_pen)
            painter.setBrush(QBrush(QColor(0, 230, 118)))
            painter.drawEllipse(QPointF(px, py), 4, 4)


class Trajectory3DWidget(QWidget):
    """
    3Dカメラ軌跡 & カバレッジビューポートウィジェット
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.simulate_demo_trajectory(with_gap=True)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # トップ統計ヘッダー
        top_box = QGroupBox("🎯 360度撮影カバレッジ & ギャップ解析 (Smart Shoot Assist)")
        top_layout = QVBoxLayout(top_box)
        top_layout.setSpacing(6)

        # 進捗バー & カバレッジパーセント
        prog_row = QHBoxLayout()
        self.lbl_coverage_pct = QLabel("カバレッジ率: 0.0%")
        self.lbl_coverage_pct.setStyleSheet("font-weight: bold; color: #00e5ff; min-width: 140px;")
        prog_row.addWidget(self.lbl_coverage_pct)

        self.coverage_bar = QProgressBar()
        self.coverage_bar.setRange(0, 100)
        self.coverage_bar.setValue(0)
        prog_row.addWidget(self.coverage_bar, stretch=1)

        top_layout.addLayout(prog_row)

        # ガイダンス警告カード
        self.guidance_banner = QLabel("撮影データ解析中...")
        self.guidance_banner.setStyleSheet(
            "background-color: #1f2735; color: #ffd600; border: 1px solid #ffab00; border-radius: 6px; padding: 8px; font-weight: bold;"
        )
        self.guidance_banner.setWordWrap(True)
        top_layout.addWidget(self.guidance_banner)

        layout.addWidget(top_box)

        # 360度レーダーキャンバス
        self.canvas = RadarCanvas()
        layout.addWidget(self.canvas, stretch=1)

        # ボトムコントロールバー
        bot_bar = QHBoxLayout()
        bot_bar.setSpacing(8)

        self.lbl_shots_count = QLabel("推定カメラ位置: 0 箇所")
        self.lbl_shots_count.setStyleSheet("color: #94a3b8;")
        bot_bar.addWidget(self.lbl_shots_count)

        bot_bar.addStretch()

        self.btn_sim_gap = QPushButton("⚠️ 欠落ギャップあり軌跡テスト")
        self.btn_sim_gap.clicked.connect(lambda: self.simulate_demo_trajectory(with_gap=True))
        bot_bar.addWidget(self.btn_sim_gap)

        self.btn_sim_full = QPushButton("✨ 360°完全カバレッジテスト")
        self.btn_sim_full.clicked.connect(lambda: self.simulate_demo_trajectory(with_gap=False))
        bot_bar.addWidget(self.btn_sim_full)

        layout.addLayout(bot_bar)

    def update_trajectory(self):
        """現在のTRAJECTORY_ANALYZERデータでUIを更新"""
        result = TRAJECTORY_ANALYZER.analyze_coverage()
        self.canvas.set_data(TRAJECTORY_ANALYZER.poses, result)

        self.lbl_coverage_pct.setText(f"カバレッジ率: {result.coverage_percentage:.1f}%")
        self.coverage_bar.setValue(int(result.coverage_percentage))
        self.lbl_shots_count.setText(f"推定カメラ位置: {result.total_shots} 箇所 (平均角度ステップ: {result.average_angular_step_deg}°)")

        if result.is_coverage_sufficient:
            self.guidance_banner.setText(result.guidance_message)
            self.guidance_banner.setStyleSheet(
                "background-color: #003314; color: #00e676; border: 1px solid #00c853; border-radius: 6px; padding: 8px; font-weight: bold;"
            )
        else:
            self.guidance_banner.setText(result.guidance_message)
            self.guidance_banner.setStyleSheet(
                "background-color: #3b1800; color: #ffab00; border: 1px solid #ffd600; border-radius: 6px; padding: 8px; font-weight: bold;"
            )

    def simulate_demo_trajectory(self, with_gap: bool = True):
        """デモ用のカメラ軌跡を生成して表示"""
        TRAJECTORY_ANALYZER.generate_simulated_trajectory(count=36, create_gap=with_gap)
        self.update_trajectory()
