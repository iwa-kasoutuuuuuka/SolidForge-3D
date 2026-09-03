# -*- coding: utf-8 -*-
"""
SolidForge 3D - Mesh Viewer & Diagnostics Widget
スタジオ内蔵 インタラクティブ 3D OpenGL ビューア (360度オービット・ズーム・パン・ワイヤーフレーム・Z=0グリッド・オーバーハング可視化)
および 3Dプリント適性診断レポート
"""

import math
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import trimesh

from PySide6.QtCore import Qt, QPointF
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from OpenGL import GL as gl
from OpenGL import GLU as glu

from solidforge.core.post_processor import PrintabilityReport
from solidforge.core.geometry_prep import GEOMETRY_PREP
from solidforge.core.project_manager import PROJECT_MANAGER


class InteractiveMeshCanvas(QOpenGLWidget):
    """
    スタジオ内蔵の高速インタラクティブ 3D OpenGL ビューア
    - マウス左ドラッグ: 360° オービット回転
    - マウス右/中ドラッグ: パン移動
    - マウスホイール: スムーズズーム
    - ビルドプレート密着グリッド (Z=0)
    - オーバーハング傾斜角 (>45°) ヒートマップ表示
    - ワイヤーフレーム表示切替
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh: Optional[trimesh.Trimesh] = None
        self.wireframe: bool = False
        self.show_grid: bool = True
        self.show_overhang: bool = False

        # カメラ制御パラメータ
        self.rot_x: float = 20.0
        self.rot_y: float = -45.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.zoom: float = 1.0
        self.last_pos: Optional[QPointF] = None

        # 頂点バッファキャッシュ
        self._vertices: Optional[np.ndarray] = None
        self._normals: Optional[np.ndarray] = None
        self._colors: Optional[np.ndarray] = None
        self._scale: float = 1.0
        self._center: np.ndarray = np.array([0.0, 0.0, 0.0])

        self.setMinimumSize(280, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def load_mesh(self, mesh: trimesh.Trimesh):
        """3Dメッシュを読み込み、OpenGL描画データをキャッシュ"""
        if mesh is None or len(mesh.faces) == 0:
            return

        self.mesh = mesh
        # メッシュのバウンディングボックスを正規化
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        max_dim = max(float(np.max(extents)), 1e-4)
        self._scale = 2.0 / max_dim
        # 底面を Z=0 に、中心を (0,0) に配置
        self._center = np.array([
            (bounds[0, 0] + bounds[1, 0]) / 2.0,
            (bounds[0, 1] + bounds[1, 1]) / 2.0,
            bounds[0, 2],
        ])

        self._update_buffers()
        self.update()

    def _update_buffers(self):
        if self.mesh is None:
            return

        faces = self.mesh.faces
        verts = self.mesh.vertices
        normals = self.mesh.face_normals

        # 三角形ごとの展開頂点
        flat_verts = verts[faces].reshape(-1, 3)
        # 正規化座標へ変換
        norm_verts = (flat_verts - self._center) * self._scale
        self._vertices = norm_verts.astype(np.float32)

        # 面法線の展開
        flat_normals = np.repeat(normals, 3, axis=0).astype(np.float32)
        self._normals = flat_normals

        # カラー配列生成
        n_pts = len(norm_verts)
        colors = np.zeros((n_pts, 3), dtype=np.float32)

        if self.show_overhang:
            # オーバーハング面 (下向き45度以上: Nz < -0.7071)
            overhang_mask = normals[:, 2] < -0.7071
            flat_mask = np.repeat(overhang_mask, 3)
            # オーバーハング面は警告色（赤橙色）
            colors[flat_mask] = [1.0, 0.25, 0.1]
            # 通常面はシックなシアン
            colors[~flat_mask] = [0.2, 0.7, 0.85]
        else:
            # 通常のサイバーブルーメタリック
            colors[:] = [0.15, 0.75, 0.9]

        self._colors = colors

    def reset_view(self):
        """視点を初期位置にリセット"""
        self.rot_x = 20.0
        self.rot_y = -45.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = 1.0
        self.update()

    def set_wireframe(self, enabled: bool):
        self.wireframe = enabled
        self.update()

    def set_show_grid(self, enabled: bool):
        self.show_grid = enabled
        self.update()

    def set_show_overhang(self, enabled: bool):
        self.show_overhang = enabled
        self._update_buffers()
        self.update()

    # --- OpenGL ライフサイクル ---

    def initializeGL(self):
        gl.glClearColor(0.06, 0.08, 0.12, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        # ライティング設定
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, [1.0, 2.0, 3.0, 0.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [0.9, 0.9, 0.95, 1.0])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [0.35, 0.35, 0.4, 1.0])

        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glShadeModel(gl.GL_SMOOTH)

    def resizeGL(self, w: int, h: int):
        gl.glViewport(0, 0, w, h)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect = w / max(h, 1)
        # 透視投影
        glu.gluPerspective(45.0, aspect, 0.1, 100.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # カメラ視点変換
        gl.glTranslatef(self.pan_x, self.pan_y, -3.8 / max(self.zoom, 0.05))
        gl.glRotatef(self.rot_x, 1.0, 0.0, 0.0)
        gl.glRotatef(self.rot_y, 0.0, 0.0, 1.0)

        # 1. ビルドプレート密着グリッド (Z=0)
        if self.show_grid:
            self._draw_build_plate_grid()

        # 2. 3Dメッシュ描画
        if self._vertices is not None and len(self._vertices) > 0:
            if self.wireframe:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
            else:
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            gl.glEnableClientState(gl.GL_NORMAL_ARRAY)
            gl.glEnableClientState(gl.GL_COLOR_ARRAY)

            gl.glVertexPointer(3, gl.GL_FLOAT, 0, self._vertices)
            gl.glNormalPointer(gl.GL_FLOAT, 0, self._normals)
            gl.glColorPointer(3, gl.GL_FLOAT, 0, self._colors)

            gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(self._vertices))

            gl.glDisableClientState(gl.GL_COLOR_ARRAY)
            gl.glDisableClientState(gl.GL_NORMAL_ARRAY)
            gl.glDisableClientState(gl.GL_VERTEX_ARRAY)

    def _draw_build_plate_grid(self):
        """Z=0 の 3Dプリンタ風ベッドグリッドを描画"""
        gl.glDisable(gl.GL_LIGHTING)
        gl.glLineWidth(1.0)
        gl.glColor4f(0.18, 0.24, 0.35, 0.8)

        grid_size = 1.6
        step = 0.2
        n_lines = int(grid_size / step)

        gl.glBegin(gl.GL_LINES)
        for i in range(-n_lines, n_lines + 1):
            coord = i * step
            # X方向ライン
            gl.glVertex3f(-grid_size, coord, 0.0)
            gl.glVertex3f(grid_size, coord, 0.0)
            # Y方向ライン
            gl.glVertex3f(coord, -grid_size, 0.0)
            gl.glVertex3f(coord, grid_size, 0.0)
        gl.glEnd()

        # ベッド中央の原点マーカー
        gl.glLineWidth(2.0)
        gl.glBegin(gl.GL_LINES)
        gl.glColor3f(1.0, 0.2, 0.2)  # X軸: 赤
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.3, 0.0, 0.0)
        gl.glColor3f(0.2, 1.0, 0.2)  # Y軸: 緑
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.3, 0.0)
        gl.glColor3f(0.2, 0.6, 1.0)  # Z軸: 青
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.3)
        gl.glEnd()

        gl.glEnable(gl.GL_LIGHTING)

    # --- マウスインタラクション ---

    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return
        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()

        if event.buttons() & Qt.LeftButton:
            # オービット回転
            self.rot_x += dy * 0.45
            self.rot_y += dx * 0.45
            self.rot_x = max(-89.0, min(89.0, self.rot_x))
            self.update()
        elif event.buttons() & (Qt.RightButton | Qt.MiddleButton):
            # パン移動
            factor = 0.003 / max(self.zoom, 0.1)
            self.pan_x += dx * factor
            self.pan_y -= dy * factor
            self.update()

        self.last_pos = event.position()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.15
        else:
            self.zoom /= 1.15
        self.zoom = max(0.1, min(20.0, self.zoom))
        self.update()


class MeshDiagnosticsWidget(QWidget):
    """
    3Dメッシュの診断レポート表示およびスタジオ内蔵 3D ビューア
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_mesh_path: Optional[Path] = None
        self.current_mesh: Optional[trimesh.Trimesh] = None
        self.current_report: Optional[PrintabilityReport] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # 水密ステータスバナー
        self.status_banner = QLabel("3Dモデル未生成")
        self.status_banner.setAlignment(Qt.AlignCenter)
        self.status_banner.setStyleSheet(
            "background-color: #1a1e2b; color: #94a3b8; border: 1px solid #2d3447; border-radius: 6px; padding: 8px; font-weight: bold; font-size: 13px;"
        )
        main_layout.addWidget(self.status_banner)

        # 2面切り替えタブ (3Dビューア / 診断レポート)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #272c3d; background: #0f1219; border-radius: 6px; }
            QTabBar::tab { background: #1a202c; color: #94a3b8; padding: 6px 14px; font-weight: bold; }
            QTabBar::tab:selected { background: #00e5ff; color: #000; }
        """)

        # Tab 1: 3D インタラクティブビューア
        viewer_page = QWidget()
        viewer_layout = QVBoxLayout(viewer_page)
        viewer_layout.setContentsMargins(4, 4, 4, 4)
        viewer_layout.setSpacing(6)

        # ビューア操作ツールバー
        tool_bar = QHBoxLayout()
        tool_bar.setSpacing(8)

        self.chk_wire = QCheckBox("ワイヤーフレーム")
        self.chk_wire.toggled.connect(self._on_wireframe_toggled)
        tool_bar.addWidget(self.chk_wire)

        self.chk_grid = QCheckBox("Z=0 グリッド")
        self.chk_grid.setChecked(True)
        self.chk_grid.toggled.connect(self._on_grid_toggled)
        tool_bar.addWidget(self.chk_grid)

        self.chk_overhang = QCheckBox("⚠️ オーバーハング警告")
        self.chk_overhang.toggled.connect(self._on_overhang_toggled)
        self.chk_overhang.setToolTip("サポート材が必要な45度以上の急傾斜面を赤橙色でハイライト表示します。")
        tool_bar.addWidget(self.chk_overhang)

        self.btn_reset_view = QPushButton("🔄 視点リセット")
        self.btn_reset_view.setStyleSheet("background-color: #1e293b; color: #cbd5e1; border: 1px solid #334155; padding: 3px 8px;")
        self.btn_reset_view.clicked.connect(self._on_reset_view_clicked)
        tool_bar.addWidget(self.btn_reset_view)

        tool_bar.addStretch()
        viewer_layout.addLayout(tool_bar)

        # OpenGL Canvas
        self.canvas = InteractiveMeshCanvas(self)
        viewer_layout.addWidget(self.canvas, stretch=1)

        # 3D造形アシストツールバー (台座付加・デシメーション)
        assist_bar = QHBoxLayout()
        self.btn_add_pedestal = QPushButton("🧱 転倒防止台座を付加")
        self.btn_add_pedestal.setEnabled(False)
        self.btn_add_pedestal.setToolTip("自立しにくいモデルの底面に厚さ2.5mmの楕円台座を結合します。")
        self.btn_add_pedestal.clicked.connect(self._on_add_pedestal_clicked)
        assist_bar.addWidget(self.btn_add_pedestal)

        self.btn_decimate = QPushButton("📉 ポリゴン軽量化 (50%)")
        self.btn_decimate.setEnabled(False)
        self.btn_decimate.setToolTip("幾何形状を維持したままポリゴン面数を半分に軽量化します。")
        self.btn_decimate.clicked.connect(self._on_decimate_clicked)
        assist_bar.addWidget(self.btn_decimate)

        viewer_layout.addLayout(assist_bar)
        self.tabs.addTab(viewer_page, "🎨 3D ビューア")

        # Tab 2: 診断レポート
        report_page = QWidget()
        report_layout = QVBoxLayout(report_page)
        report_layout.setContentsMargins(4, 4, 4, 4)

        self.report_view = QTextBrowser()
        self.report_view.setReadOnly(True)
        self.report_view.setStyleSheet("""
            QTextBrowser {
                background-color: #12151f;
                color: #e2e8f0;
                border: none;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        self.report_view.setHtml("""
            <p style='color: #64748b; text-align: center; margin-top: 20px;'>
                「Forge 3D Mesh」を実行すると、ここに実寸寸法(mm)や<br>
                水密性(Watertight)、オーバーハング角の診断レポートが表示されます。
            </p>
        """)
        report_layout.addWidget(self.report_view)
        self.tabs.addTab(report_page, "📊 診断レポート")

        main_layout.addWidget(self.tabs, stretch=1)

        # 外部連携 & プロジェクト保存ボタン
        btn_box = QHBoxLayout()
        self.btn_open_external = QPushButton("🖨️ スライサーで開く")
        self.btn_open_external.setEnabled(False)
        self.btn_open_external.clicked.connect(self._open_in_external_app)
        btn_box.addWidget(self.btn_open_external)

        self.btn_open_folder = QPushButton("📂 出力フォルダ")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self._open_output_folder)
        btn_box.addWidget(self.btn_open_folder)

        self.btn_save_project = QPushButton("💾 プロジェクト保存")
        self.btn_save_project.setEnabled(False)
        self.btn_save_project.clicked.connect(self._on_save_project_clicked)
        btn_box.addWidget(self.btn_save_project)

        main_layout.addLayout(btn_box)

    def display_report(self, mesh_path: Path, report: PrintabilityReport):
        """診断レポートと 3D メッシュを反映"""
        self.current_mesh_path = mesh_path
        self.current_report = report
        self.btn_open_external.setEnabled(True)
        self.btn_open_folder.setEnabled(True)
        self.btn_save_project.setEnabled(True)
        self.btn_add_pedestal.setEnabled(True)
        self.btn_decimate.setEnabled(True)

        # 3Dモデルをキャンバスにロード
        try:
            mesh = trimesh.load(str(mesh_path))
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            self.current_mesh = mesh
            self.canvas.load_mesh(mesh)
        except Exception as e:
            print(f"[MeshViewer] 3D表示エラー: {e}")

        # 水密バナー更新
        if report.is_watertight:
            self.status_banner.setText("✨ 【水密性 合格】 3Dスライサーへ直行可能 (Watertight)")
            self.status_banner.setStyleSheet(
                "background-color: #003314; color: #00e676; border: 1px solid #00c853; border-radius: 6px; padding: 8px; font-weight: bold; font-size: 13px;"
            )
        else:
            self.status_banner.setText("⚠️ 【非水密】 スライサーの穴埋め機能を使用してください")
            self.status_banner.setStyleSheet(
                "background-color: #3b1800; color: #ffab00; border: 1px solid #ffd600; border-radius: 6px; padding: 8px; font-weight: bold; font-size: 13px;"
            )

        dim_x, dim_y, dim_z = report.dimensions_mm
        overhang_txt = f"<span style='color: #ff5722;'>⚠️ 要サポート材 (急傾斜 {report.overhang_area_ratio_pct:.1f}%)</span>" if report.has_critical_overhang else f"<span style='color: #00e676;'>良好 (急傾斜 {report.overhang_area_ratio_pct:.1f}%)</span>"

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
                <td style='color: #00e5ff; font-weight: bold; padding: 4px;'>オーバーハング傾斜 (>45°)</td>
                <td style='padding: 4px;'>{overhang_txt}</td>
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

    # --- アクションハンドラ ---

    def _on_wireframe_toggled(self, checked: bool):
        self.canvas.set_wireframe(checked)

    def _on_grid_toggled(self, checked: bool):
        self.canvas.set_show_grid(checked)

    def _on_overhang_toggled(self, checked: bool):
        self.canvas.set_show_overhang(checked)

    def _on_reset_view_clicked(self):
        self.canvas.reset_view()

    def _on_add_pedestal_clicked(self):
        if self.current_mesh is None or self.current_mesh_path is None:
            return
        modified = GEOMETRY_PREP.add_base_pedestal(self.current_mesh, thickness_mm=2.5, margin_mm=5.0)
        out_path = self.current_mesh_path.parent / f"{self.current_mesh_path.stem}_with_pedestal.stl"
        modified.export(str(out_path))
        self.current_mesh = modified
        self.current_mesh_path = out_path
        self.canvas.load_mesh(modified)
        QMessageBox.information(self, "台座付加完了", f"転倒防止台座（厚さ2.5mm）を合成しました！\n保存先: {out_path.name}")

    def _on_decimate_clicked(self):
        if self.current_mesh is None or self.current_mesh_path is None:
            return
        orig_faces = len(self.current_mesh.faces)
        target_faces = max(1000, orig_faces // 2)
        decimated = GEOMETRY_PREP.decimate_mesh(self.current_mesh, target_face_count=target_faces)
        out_path = self.current_mesh_path.parent / f"{self.current_mesh_path.stem}_decimated.stl"
        decimated.export(str(out_path))
        self.current_mesh = decimated
        self.current_mesh_path = out_path
        self.canvas.load_mesh(decimated)
        QMessageBox.information(
            self,
            "軽量化完了",
            f"ポリゴン面数を {orig_faces:,} 面 ➔ {len(decimated.faces):,} 面 に軽量化しました！\n保存先: {out_path.name}"
        )

    def _on_save_project_clicked(self):
        if not self.current_mesh_path:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "プロジェクトを保存 (.sforge)", str(self.current_mesh_path.parent), "SolidForge Project (*.sforge)")
        if save_path:
            p = Path(save_path)
            # 画像フォルダから元写真を取得
            images_dir = self.current_mesh_path.parent / "images"
            imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) if images_dir.exists() else []
            success = PROJECT_MANAGER.export_project(
                dest_file_path=p,
                project_name=p.stem,
                image_paths=imgs,
                model_path=self.current_mesh_path,
                report=self.current_report,
            )
            if success:
                QMessageBox.information(self, "保存完了", f"スキャンプロジェクトを正常に保存しました！\n{p.name}")

    def _open_in_external_app(self):
        if self.current_mesh_path and self.current_mesh_path.exists():
            from solidforge.core.exporter import ModelExporter
            ModelExporter.launch_in_orcaslicer(self.current_mesh_path)

    def _open_output_folder(self):
        if self.current_mesh_path and self.current_mesh_path.exists():
            os.startfile(str(self.current_mesh_path.parent))
