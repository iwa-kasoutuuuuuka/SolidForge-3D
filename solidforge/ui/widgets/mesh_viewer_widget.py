"""
SolidForge 3D - Mesh Viewer & Diagnostics Widget
生成された3Dメッシュの品質診断、実寸mm寸法、スライサー投入プレビュー、外部ビューア連携
"""

import os
from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from solidforge.core.post_processor import PrintabilityReport


class MeshDiagnosticsWidget(QWidget):
    """
    3Dメッシュの診断レポート表示および外部ビューア/スライサー連携ウィジェット
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mesh_path: Optional[Path] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # 水密ステータスバナー
        self.status_banner = QLabel("3Dモデル未生成")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setStyleSheet(
            "background-color: #1a1e2b; color: #94a3b8; border: 1px solid #2d3447; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px;"
        )
        layout.addWidget(self.status_banner)

        # 診断詳細表示用テキストブラウザ
        self.report_view = QTextBrowser()
        self.report_view.setReadOnly(True)
        self.report_view.setStyleSheet("""
            QTextBrowser {
                background-color: #12151f;
                color: #e2e8f0;
                border: 1px solid #272c3d;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.report_view.setHtml("""
            <p style='color: #64748b; text-align: center; margin-top: 20px;'>
                「Forge 3D Mesh」を実行すると、ここに実寸寸法(mm)や<br>
                水密性(Watertight)の診断レポートが表示されます。
            </p>
        """)
        layout.addWidget(self.report_view, stretch=1)

        # 外部ツール連携ボタン
        btn_box = QHBoxLayout()
        self.btn_open_external = QPushButton("🖨️ 3Dビューア / スライサーで開く")
        self.btn_open_external.setEnabled(False)
        self.btn_open_external.clicked.connect(self._open_in_external_app)
        btn_box.addWidget(self.btn_open_external)

        self.btn_open_folder = QPushButton("📂 出力フォルダを開く")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        btn_box.addWidget(self.btn_open_folder)

        layout.addLayout(btn_box)

    def display_report(self, mesh_path: Path, report: PrintabilityReport):
        """診断レポートとモデル情報を反映"""
        self.current_mesh_path = mesh_path
        self.btn_open_external.setEnabled(True)
        self.btn_open_folder.setEnabled(True)

        if report.is_watertight:
            self.status_banner.setText("✨ 【水密性 合格】 3Dスライサーへ直行可能 (Watertight)")
            self.status_banner.setStyleSheet(
                "background-color: #003314; color: #00e676; border: 1px solid #00c853; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px;"
            )
        else:
            self.status_banner.setText("⚠️ 【非水密】 スライサーの穴埋め機能を使用してください")
            self.status_banner.setStyleSheet(
                "background-color: #3b1800; color: #ffab00; border: 1px solid #ffd600; border-radius: 6px; padding: 10px; font-weight: bold; font-size: 13px;"
            )

        dim_x, dim_y, dim_z = report.dimensions_mm
        html = f"""
        <table style='width: 100%; border-collapse: collapse; font-size: 12px;'>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>出力ファイル</td>
                <td style='color: #f1f5f9; padding: 4px;'><b>{mesh_path.name}</b></td>
            </tr>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>実寸外形寸法 (XYZ)</td>
                <td style='color: #00e676; font-weight: bold; padding: 4px;'>{dim_x:.1f} x {dim_y:.1f} x {dim_z:.1f} mm</td>
            </tr>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>モデル体積</td>
                <td style='color: #f1f5f9; padding: 4px;'>{report.volume_cm3:.2f} cm³</td>
            </tr>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>表面積</td>
                <td style='color: #f1f5f9; padding: 4px;'>{report.surface_area_cm2:.2f} cm²</td>
            </tr>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>推定PLA重量 (100%充填)</td>
                <td style='color: #f1f5f9; padding: 4px;'>約 {report.estimated_weight_pla_g:.1f} g</td>
            </tr>
            <tr style='border-bottom: 1px solid #2d3447;'>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>ポリゴン面数 / 頂点数</td>
                <td style='color: #f1f5f9; padding: 4px;'>{report.face_count:,} 面 / {report.vertex_count:,} 頂点</td>
            </tr>
            <tr>
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>ArUco校正倍率</td>
                <td style='color: #f1f5f9; padding: 4px;'>{report.scale_factor_applied:.4f}x (1:1 mm)</td>
            </tr>
        </table>
        """
        self.report_view.setHtml(html)

    def _open_in_external_app(self):
        if self.current_mesh_path and self.current_mesh_path.exists():
            from solidforge.core.exporter import ModelExporter
            ModelExporter.launch_in_orcaslicer(self.current_mesh_path)

    def _open_output_folder(self):
        if self.current_mesh_path and self.current_mesh_path.exists():
            import os
            os.startfile(str(self.current_mesh_path.parent))
