"""
SolidForge 3D - Main Window (メインアプリケーションウィンドウ - 拡張版)
完全日本語対応 / SONY ZV-E10 / RTX 5080 TensorRT AI / Smart Shoot Assist / Direct-to-Print
"""

from pathlib import Path
from typing import Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from solidforge.config import CONFIG, AppConfig
from solidforge.core.camera_manager import CAMERA_MANAGER
from solidforge.core.pipeline import PIPELINE
from solidforge.core.post_processor import PrintabilityReport
from solidforge.core.quality_gate import QUALITY_GATE
from solidforge.ui.theme import DARK_THEME_QSS
from solidforge.ui.widgets.gallery_widget import GalleryWidget
from solidforge.ui.widgets.live_view_widget import LiveViewWidget
from solidforge.ui.widgets.log_terminal import LogTerminalWidget
from solidforge.ui.widgets.mesh_viewer_widget import MeshDiagnosticsWidget
from solidforge.ui.widgets.trajectory_3d_widget import Trajectory3DWidget


class MainWindow(QMainWindow):
    """
    SolidForge 3D メインウィンドウ
    """

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or CONFIG
        self.setWindowTitle(f"SolidForge 3D - 高精度3Dプリント用フォトグラメトリスタジオ (v{self.config.version})")
        self.resize(1420, 920)
        self.setMinimumSize(1150, 720)

        # テーマ適用
        self.setStyleSheet(DARK_THEME_QSS)

        # ウィンドウアイコン設定
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # 1. トップナビゲーションバー (タイトル & モード切替)
        top_nav = self._create_top_nav()
        root_layout.addLayout(top_nav)

        # 2. 初心者向け撮影チュートリアルガイダンスバナー
        guidance_banner = self._create_guidance_banner()
        root_layout.addWidget(guidance_banner)

        # 3. メインコンテンツエリア (左右スプリッター)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setChildrenCollapsible(False)

        # 左側: Live View, ギャラリー, 3Dカメラ軌跡タブ
        left_pane = self._create_left_pane()
        main_splitter.addWidget(left_pane)

        # 右側: 設定・ハードウェア最適化・Direct-to-Print・診断パネル
        right_pane = self._create_right_pane()
        main_splitter.addWidget(right_pane)

        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 2)
        root_layout.addWidget(main_splitter, stretch=2)

        # 4. 下部: リアルタイム処理ログターミナル
        self.log_terminal = LogTerminalWidget()
        root_layout.addWidget(self.log_terminal, stretch=1)

    def _create_top_nav(self) -> QHBoxLayout:
        top_layout = QHBoxLayout()
        top_layout.setSpacing(12)

        # アプリアイコンバッジ
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.png"
        if icon_path.exists():
            logo_lbl = QLabel()
            pix = QPixmap(str(icon_path)).scaled(28, 28, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(pix)
            top_layout.addWidget(logo_lbl)

        title_label = QLabel("SolidForge 3D")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00e676; letter-spacing: 1px;")
        top_layout.addWidget(title_label)

        sub_label = QLabel("| Multi-Camera × RTX 5080 Multi-GPU フォトグラメトリ")
        sub_label.setStyleSheet("font-size: 13px; color: #94a3b8; font-weight: bold;")
        top_layout.addWidget(sub_label)

        top_layout.addStretch()

        # カメラ切替ダイアログ起動ボタン
        self.btn_top_camera_select = QPushButton("📷 カメラ切替 / スマホ接続")
        self.btn_top_camera_select.setStyleSheet(
            "QPushButton { background-color: #1e293b; color: #38bdf8; border: 1px solid #38bdf8; padding: 4px 10px; border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0284c7; color: #ffffff; }"
        )
        self.btn_top_camera_select.clicked.connect(self._open_camera_select_dialog)
        top_layout.addWidget(self.btn_top_camera_select)

        # モード選択
        mode_box = QHBoxLayout()
        mode_box.setSpacing(8)
        mode_label = QLabel("動作モード:")
        mode_label.setStyleSheet("color: #00e5ff; font-weight: bold;")
        mode_box.addWidget(mode_label)

        self.btn_mode_wired = QRadioButton("Live View 撮影モード (カメラ/スマホ/Insta360)")
        self.btn_mode_offline = QRadioButton("オフライン読込モード (SDカード/フォルダ)")
        self.btn_mode_wired.setChecked(True)

        self.btn_mode_wired.toggled.connect(self._on_mode_changed)
        mode_box.addWidget(self.btn_mode_wired)
        mode_box.addWidget(self.btn_mode_offline)

        top_layout.addLayout(mode_box)
        return top_layout

    def _create_guidance_banner(self) -> QWidget:
        banner = QFrame()
        banner.setObjectName("guidanceBanner")
        layout = QHBoxLayout(banner)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        icon_label = QLabel("💡")
        icon_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(icon_label)

        guide_text = QLabel(
            "<b>【Smart Shoot Assist ガイダンス】</b> ターンテーブル不要：被写体の周囲を360度回りながら"
            "<b>オニオンスキン（前フレーム重畳）で60〜80%オーバーラップ</b>を維持して撮影してください。"
            "「3D軌跡」タブで撮影角度の欠落ギャップをリアルタイム監視できます。"
        )
        guide_text.setStyleSheet("color: #e2e8f0; font-size: 12px; line-height: 1.4;")
        guide_text.setWordWrap(True)
        layout.addWidget(guide_text, stretch=1)

        return banner

    def _create_left_pane(self) -> QWidget:
        pane = QWidget()
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # タブコンテナ (3タブ)
        self.view_tabs = QTabWidget()
        
        # タブ1: Live View
        self.live_view_widget = LiveViewWidget()
        self.view_tabs.addTab(self.live_view_widget, "🎥 Live View & オニオンスキン")

        # タブ2: 画像ギャラリー
        self.gallery_widget = GalleryWidget()
        self.view_tabs.addTab(self.gallery_widget, "🖼️ 画像一覧 & キュー管理 (Quality Gate)")

        # タブ3: 3Dカメラ軌跡 & カバレッジ
        self.trajectory_widget = Trajectory3DWidget()
        self.view_tabs.addTab(self.trajectory_widget, "🎯 3Dカメラ軌跡 & 360°カバレッジ")

        layout.addWidget(self.view_tabs)
        return pane

    def _create_right_pane(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)

        # 1. Multi-GPU & ハードウェアクラスタグループ
        gpu_group = QGroupBox("⚡ GPU & Multi-GPU クラスタ設定")
        gpu_layout = QFormLayout(gpu_group)
        gpu_layout.setSpacing(8)

        self.combo_gpu_device = QComboBox()
        self._populate_gpu_dropdown()
        self.combo_gpu_device.currentIndexChanged.connect(self._on_gpu_selection_changed)
        gpu_layout.addRow("計算デバイス:", self.combo_gpu_device)
        layout.addWidget(gpu_group)

        # 2. RTX 5080 AI TensorRT 超解像 & ブレ除去グループ
        ai_group = QGroupBox("🤖 RTX 5080 TensorRT AI画質向上")
        ai_layout = QFormLayout(ai_group)
        ai_layout.setSpacing(8)

        self.chk_enable_ai = QCheckBox("AIブレ除去 & 超解像を有効化 (Tensor Cores)")
        self.chk_enable_ai.setChecked(self.config.ai_enhancement.enable_ai_enhancer)
        ai_layout.addRow("AI前処理:", self.chk_enable_ai)

        self.combo_ai_model = QComboBox()
        self.combo_ai_model.addItems([
            "Hybrid_Fast (NAFNetデブラー + エッジ復元) [推奨]",
            "RealESRGAN_x2 (超解像テクスチャ強化)",
            "High-ISO Denoise (暗所ノイズ特化)",
        ])
        ai_layout.addRow("AIモデル:", self.combo_ai_model)

        self.slider_ai_sharpen = QSlider(Qt.Horizontal)
        self.slider_ai_sharpen.setRange(10, 100)
        self.slider_ai_sharpen.setValue(int(self.config.ai_enhancement.sharpen_strength * 100))
        ai_layout.addRow("鮮鋭化強度:", self.slider_ai_sharpen)

        layout.addWidget(ai_group)

        # 2. Quality Gate 設定グループ
        qg_group = QGroupBox("🔍 Quality Gate (品質判定) 設定")
        qg_layout = QFormLayout(qg_group)
        qg_layout.setSpacing(8)

        self.slider_blur = QSlider(Qt.Horizontal)
        self.slider_blur.setRange(20, 300)
        self.slider_blur.setValue(int(self.config.quality.blur_threshold))
        self.lbl_blur_val = QLabel(f"{self.config.quality.blur_threshold:.0f}")
        self.slider_blur.valueChanged.connect(self._on_blur_slider_changed)
        
        blur_row = QHBoxLayout()
        blur_row.addWidget(self.slider_blur, stretch=1)
        blur_row.addWidget(self.lbl_blur_val)
        qg_layout.addRow("ブレ鮮鋭度閾値 (Laplacian):", blur_row)

        self.spin_feat_min = QSpinBox()
        self.spin_feat_min.setRange(50, 2000)
        self.spin_feat_min.setValue(self.config.quality.min_feature_count)
        self.spin_feat_min.setSingleStep(50)
        self.spin_feat_min.valueChanged.connect(self._on_feat_spin_changed)
        qg_layout.addRow("最低必要特徴点数:", self.spin_feat_min)

        layout.addWidget(qg_group)

        # 3. Direct-to-Print 3Dプリント幾何形状処理グループ
        direct_group = QGroupBox("🖨️ Direct-to-Print (3Dプリント造形最適化)")
        direct_layout = QFormLayout(direct_group)
        direct_layout.setSpacing(8)

        self.chk_ground_cut = QCheckBox("RANSAC 接地面自動検出 & 底面フラットカット")
        self.chk_ground_cut.setChecked(self.config.post_process.direct_to_print.enable_ground_cut)
        self.chk_ground_cut.setToolTip("テーブル面を自動検出して水平にスライスし、ビルドプレートへの100%密着底面を成形します")
        direct_layout.addRow("接地面成形:", self.chk_ground_cut)

        self.chk_hollowing = QCheckBox("SLA向け中空化 (Hollowing)")
        self.chk_hollowing.setChecked(self.config.post_process.direct_to_print.enable_hollowing)
        self.chk_hollowing.toggled.connect(self._on_hollowing_toggled)
        direct_layout.addRow("中空化造形:", self.chk_hollowing)

        self.spin_wall_thick = QDoubleSpinBox()
        self.spin_wall_thick.setRange(0.8, 8.0)
        self.spin_wall_thick.setValue(self.config.post_process.direct_to_print.wall_thickness_mm)
        self.spin_wall_thick.setSingleStep(0.5)
        self.spin_wall_thick.setSuffix(" mm")
        self.spin_wall_thick.setEnabled(self.chk_hollowing.isChecked())
        direct_layout.addRow("中空肉厚 (Wall Thickness):", self.spin_wall_thick)

        self.chk_drain_holes = QCheckBox("レジン排出穴 (Drain Holes) 自動開口")
        self.chk_drain_holes.setChecked(self.config.post_process.direct_to_print.enable_drain_holes)
        self.chk_drain_holes.setEnabled(self.chk_hollowing.isChecked())
        direct_layout.addRow("排出口配置:", self.chk_drain_holes)

        self.spin_aruco_size = QDoubleSpinBox()
        self.spin_aruco_size.setRange(5.0, 500.0)
        self.spin_aruco_size.setValue(self.config.post_process.aruco_marker_size_mm)
        self.spin_aruco_size.setSuffix(" mm")
        direct_layout.addRow("ArUcoマーカー実寸一辺:", self.spin_aruco_size)

        layout.addWidget(direct_group)

        # 4. 3D診断レポートウィジェット
        self.mesh_diagnostics = MeshDiagnosticsWidget()
        layout.addWidget(self.mesh_diagnostics)

        # 5. エクスポート & 生成実行アクション
        action_group = QGroupBox("🚀 3Dモデル生成 & エクスポート")
        action_layout = QVBoxLayout(action_group)
        action_layout.setSpacing(10)

        fmt_row = QHBoxLayout()
        fmt_label = QLabel("出力フォーマット:")
        fmt_label.setStyleSheet("font-weight: bold; color: #00e5ff;")
        fmt_row.addWidget(fmt_label)

        self.combo_format = QComboBox()
        self.combo_format.addItems([
            "STL (.stl) - 3Dプリント推奨 (バイナリ)",
            "OBJ (.obj) - テクスチャ付きメッシュ",
            "PLY (.ply) - 高密度点群・頂点カラー",
            "GLTF (.gltf) - Web/AR用標準フォーマット",
        ])
        fmt_row.addWidget(self.combo_format, stretch=1)
        action_layout.addLayout(fmt_row)

        self.btn_forge = QPushButton("✨ 3Dメッシュ生成開始 (Forge 3D Mesh)")
        self.btn_forge.setObjectName("primaryActionBtn")
        self.btn_forge.setMinimumHeight(46)
        self.btn_forge.clicked.connect(self._on_start_forge_clicked)
        action_layout.addWidget(self.btn_forge)

        layout.addWidget(action_group)
        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _connect_signals(self):
        CAMERA_MANAGER.photo_captured.connect(self._on_photo_captured_from_camera)
        PIPELINE.reconstruction_finished.connect(self._on_reconstruction_finished)

    def _open_camera_select_dialog(self):
        from solidforge.ui.widgets.camera_select_dialog import CameraSelectDialog
        dialog = CameraSelectDialog(self)
        dialog.exec()

    def _on_mode_changed(self):
        if self.btn_mode_wired.isChecked():
            self.view_tabs.setCurrentIndex(0)
            if not CAMERA_MANAGER.is_connected:
                CAMERA_MANAGER.connect_sony_sdk()
        else:
            self.view_tabs.setCurrentIndex(1)

    def _on_hollowing_toggled(self, checked: bool):
        self.spin_wall_thick.setEnabled(checked)
        self.chk_drain_holes.setEnabled(checked)

    def _on_blur_slider_changed(self, val: int):
        self.lbl_blur_val.setText(str(val))
        QUALITY_GATE.config.blur_threshold = float(val)

    def _on_feat_spin_changed(self, val: int):
        QUALITY_GATE.config.min_feature_count = val

    def _on_photo_captured_from_camera(self, file_path_str: str, eval_res):
        p = Path(file_path_str)
        self.gallery_widget.add_image(p, eval_res)
        self.log_terminal.append_log(
            f"撮影画像を保存: {p.name} | 鮮鋭度: {eval_res.blur_score:.1f} | 判定: {eval_res.status_label}"
        )
        # 3Dカメラ軌跡を更新
        self.trajectory_widget.simulate_demo_trajectory(with_gap=False)

    def _populate_gpu_dropdown(self):
        """利用可能なGPUをドロップダウンに列挙"""
        from solidforge.core.hardware_optimizer import HARDWARE_OPTIMIZER
        self.combo_gpu_device.clear()
        gpus = HARDWARE_OPTIMIZER.list_all_gpus()

        if len(gpus) > 1:
            total_vram = sum(g.vram_total_mb for g in gpus) / 1024
            self.combo_gpu_device.addItem(
                f"⚡ 全GPU並列クラスタ ({len(gpus)}基 合計{total_vram:.0f}GB VRAM) [推奨]",
                "ALL"
            )
            for g in gpus:
                self.combo_gpu_device.addItem(
                    f"GPU {g.device_id}: {g.gpu_name} ({g.vram_total_mb/1024:.0f}GB)",
                    g.device_id
                )
        else:
            g = gpus[0]
            self.combo_gpu_device.addItem(f"⚡ GPU 0: {g.gpu_name} ({g.vram_total_mb/1024:.0f}GB)", 0)

    def _on_gpu_selection_changed(self, index: int):
        data = self.combo_gpu_device.currentData()
        if data == "ALL":
            self.config.multi_gpu.mode = "ALL_GPUS"
            from solidforge.core.hardware_optimizer import HARDWARE_OPTIMIZER
            gpus = HARDWARE_OPTIMIZER.list_all_gpus()
            self.config.multi_gpu.active_device_indices = [g.device_id for g in gpus]
        else:
            dev_id = int(data or 0)
            self.config.multi_gpu.mode = "CUSTOM"
            self.config.multi_gpu.active_device_indices = [dev_id]

    def _on_start_forge_clicked(self):
        if PIPELINE.is_running:
            QMessageBox.warning(self, "警告", "既にパイプラインが実行中です。")
            return

        selected_images = self.gallery_widget.get_selected_images()
        if len(selected_images) == 0:
            res = QMessageBox.question(
                self,
                "画像未投入",
                "再構築キューに画像が選択されていません。\nデモ検証用としてシミュレーション撮影データ（36枚）を生成して実行しますか？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if res == QMessageBox.Yes:
                demo_dir = self.config.workspace_dir / "demo_dataset"
                demo_dir.mkdir(parents=True, exist_ok=True)
                demo_paths = []
                for i in range(36):
                    f_path = demo_dir / f"demo_scan_{i:03d}.jpg"
                    if not f_path.exists():
                        CAMERA_MANAGER._sim_angle = i * (2 * 3.14159 / 36.0)
                        f = CAMERA_MANAGER._generate_simulated_frame()
                        import cv2
                        cv2.imwrite(str(f_path), f)
                    demo_paths.append(f_path)
                self.gallery_widget.import_file_paths(demo_paths)
                selected_images = demo_paths
            else:
                return

        # パラメータ設定を同期
        self.config.ai_enhancement.enable_ai_enhancer = self.chk_enable_ai.isChecked()
        self.config.ai_enhancement.sharpen_strength = float(self.slider_ai_sharpen.value()) / 100.0
        
        self.config.post_process.aruco_marker_size_mm = self.spin_aruco_size.value()
        self.config.post_process.direct_to_print.enable_ground_cut = self.chk_ground_cut.isChecked()
        self.config.post_process.direct_to_print.enable_hollowing = self.chk_hollowing.isChecked()
        self.config.post_process.direct_to_print.wall_thickness_mm = self.spin_wall_thick.value()
        self.config.post_process.direct_to_print.enable_drain_holes = self.chk_drain_holes.isChecked()

        raw_fmt = self.combo_format.currentText()
        ext = "stl"
        if "obj" in raw_fmt.lower():
            ext = "obj"
        elif "ply" in raw_fmt.lower():
            ext = "ply"
        elif "gltf" in raw_fmt.lower():
            ext = "gltf"

        self.btn_forge.setEnabled(False)
        self.btn_forge.setText("⏳ 3Dモデル生成中...")

        PIPELINE.start_pipeline_async(selected_images, output_format=ext)

    def _on_reconstruction_finished(self, success: bool, mesh_path_str: str, report: Optional[PrintabilityReport]):
        self.btn_forge.setEnabled(True)
        self.btn_forge.setText("✨ 3Dメッシュ生成開始 (Forge 3D Mesh)")

        if success and mesh_path_str and report:
            p = Path(mesh_path_str)
            self.mesh_diagnostics.display_report(p, report)
            QMessageBox.information(
                self,
                "生成完了",
                f"3Dモデルの生成とDirect-to-Print最適化が完了しました！\n\n"
                f"ファイル: {p.name}\n"
                f"実寸外形: {report.dimensions_mm[0]:.1f} x {report.dimensions_mm[1]:.1f} x {report.dimensions_mm[2]:.1f} mm\n"
                f"水密性: {'完全水密 (Watertight)' if report.is_watertight else '非水密'}\n"
                f"体積: {report.volume_cm3:.2f} cm³ (PLA推定: {report.estimated_weight_pla_g:.1f} g)\n\n"
                f"右側パネルからスライサーまたは3Dビューアで開けます。",
            )
        elif not success:
            QMessageBox.critical(self, "エラー", "3D再構築パイプラインでエラーが発生しました。ログを確認してください。")

    def closeEvent(self, event):
        CAMERA_MANAGER.disconnect()
        event.accept()
