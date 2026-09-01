"""
SolidForge 3D - Gallery Widget (画像インポート & 品質管理ギャラリー)
SDカード/フォルダインポート、サムネイル一覧、合否バッジ、ブレ画像の一括除外
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from solidforge.core.quality_gate import QUALITY_GATE, QualityEvaluationResult


class QualityBatchWorker(QThread):
    """バックグラウンド画像品質一括判定スレッド"""
    progress = Signal(int, int)  # (current, total)
    item_evaluated = Signal(Path, object, np.ndarray)  # (path, QualityEvaluationResult, thumbnail_rgb)
    finished_all = Signal(int, int)  # (accepted_count, rejected_count)

    def __init__(self, file_paths: List[Path], parent=None):
        super().__init__(parent)
        self.file_paths = file_paths

    def run(self):
        total = len(self.file_paths)
        accepted = 0
        rejected = 0

        for i, p in enumerate(self.file_paths):
            eval_res = QUALITY_GATE.evaluate_image(p)
            
            # サムネイル生成
            thumb_rgb = np.zeros((120, 160, 3), dtype=np.uint8)
            if p.exists() and p.is_file() and p.stat().st_size > 0:
                try:
                    buf = np.fromfile(str(p), dtype=np.uint8)
                    if len(buf) > 0:
                        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                        if img is not None:
                            thumb = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
                            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass

            if eval_res.is_accepted:
                accepted += 1
            else:
                rejected += 1

            self.item_evaluated.emit(p, eval_res, thumb_rgb)
            self.progress.emit(i + 1, total)

        self.finished_all.emit(accepted, rejected)


class ImageCardWidget(QFrame):
    """ギャラリー内の個別画像カード"""

    toggled = Signal(bool)

    def __init__(self, file_path: Path, eval_res: QualityEvaluationResult, thumb_rgb: np.ndarray, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.eval_res = eval_res
        self.is_checked = eval_res.is_accepted
        self._init_ui(thumb_rgb)

    def _init_ui(self, thumb_rgb: np.ndarray):
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedSize(170, 200)

        border_color = "#00e676" if self.eval_res.is_accepted else "#ff5252"
        self.setStyleSheet(f"""
            ImageCardWidget {{
                background-color: #1a1e2b;
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 4px;
            }}
            ImageCardWidget:hover {{
                border: 2px solid #00e5ff;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        # サムネイル
        h, w, ch = thumb_rgb.shape
        q_img = QImage(thumb_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(q_img)

        self.img_label = QLabel()
        self.img_label.setPixmap(pix)
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        # ファイル名
        name_label = QLabel(self.file_path.name)
        name_label.setStyleSheet("color: #e2e8f0; font-weight: bold; font-size: 11px;")
        name_label.setToolTip(str(self.file_path))
        layout.addWidget(name_label)

        # 鮮鋭度 & 特徴点スコア
        score_text = f"鮮鋭度: {self.eval_res.blur_score:.1f} | 特徴: {self.eval_res.feature_count}点"
        score_label = QLabel(score_text)
        score_label.setStyleSheet("color: #94a3b8; font-size: 10px;")
        layout.addWidget(score_label)

        # 下部コントロール (チェックボックス + 合否バッジ)
        bot_layout = QHBoxLayout()
        bot_layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox("使用")
        self.checkbox.setChecked(self.is_checked)
        self.checkbox.stateChanged.connect(self._on_check_changed)
        bot_layout.addWidget(self.checkbox)

        bot_layout.addStretch()

        self.badge_label = QLabel("合格" if self.eval_res.is_accepted else "除外推奨")
        self._update_badge_style(self.eval_res.is_accepted)
        if self.eval_res.rejection_reasons:
            self.badge_label.setToolTip("\n".join(self.eval_res.rejection_reasons))
        bot_layout.addWidget(self.badge_label)

        layout.addLayout(bot_layout)

    def _update_badge_style(self, accepted: bool):
        border_color = "#00e676" if accepted else "#ff5252"
        self.setStyleSheet(f"""
            ImageCardWidget {{
                background-color: #1a1e2b;
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 4px;
            }}
            ImageCardWidget:hover {{
                border: 2px solid #00e5ff;
            }}
        """)
        if accepted:
            self.badge_label.setText("合格")
            self.badge_label.setStyleSheet(
                "background-color: #00c853; color: #000; font-weight: bold; border-radius: 3px; padding: 1px 4px; font-size: 10px;"
            )
        else:
            self.badge_label.setText("除外推奨")
            self.badge_label.setStyleSheet(
                "background-color: #d50000; color: #fff; font-weight: bold; border-radius: 3px; padding: 1px 4px; font-size: 10px;"
            )

    def update_thresholds(self, blur_threshold: float, min_features: int):
        accepted = (self.eval_res.blur_score >= blur_threshold) and (self.eval_res.feature_count >= min_features)
        self.eval_res.is_accepted = accepted
        self._update_badge_style(accepted)
        self.set_checked(accepted)

    def _on_check_changed(self, state):
        self.is_checked = (state == Qt.Checked.value or state == 2)
        self.toggled.emit(self.is_checked)

    def set_checked(self, checked: bool):
        self.checkbox.blockSignals(True)
        self.checkbox.setChecked(checked)
        self.checkbox.blockSignals(False)
        self.is_checked = checked


class GalleryWidget(QWidget):
    """
    撮影画像 & インポート画像の管理ギャラリー
    """

    count_changed = Signal(int, int)  # (selected_count, total_count)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cards: List[ImageCardWidget] = []
        self._worker: Optional[QualityBatchWorker] = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # 上部アクションバー
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.btn_import_folder = QPushButton("📁 フォルダ / SDカードからインポート")
        self.btn_import_folder.clicked.connect(self._on_import_folder)
        top_bar.addWidget(self.btn_import_folder)

        self.btn_import_files = QPushButton("🖼️ 画像ファイルを選択追加")
        self.btn_import_files.clicked.connect(self._on_import_files)
        top_bar.addWidget(self.btn_import_files)

        top_bar.addStretch()

        self.btn_filter_rejects = QPushButton("🚫 不合格(ブレ画像)を一括除外")
        self.btn_filter_rejects.clicked.connect(self.deselect_rejected_images)
        top_bar.addWidget(self.btn_filter_rejects)

        self.btn_select_all = QPushButton("全選択")
        self.btn_select_all.clicked.connect(lambda: self.set_all_checked(True))
        top_bar.addWidget(self.btn_select_all)

        self.btn_clear = QPushButton("クリア")
        self.btn_clear.clicked.connect(self.clear_gallery)
        top_bar.addWidget(self.btn_clear)

        main_layout.addLayout(top_bar)

        # ステータス情報ラベル
        self.status_label = QLabel("画像未読込 (被写体の周囲を360度撮影した画像を30枚以上インポートしてください)")
        self.status_label.setStyleSheet("color: #00e5ff; font-weight: bold;")
        main_layout.addWidget(self.status_label)

        # スクロールエリア
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: #12141c; border: 1px solid #272c3d; border-radius: 6px;")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area, stretch=1)

    def add_image(self, file_path: Path, eval_res: QualityEvaluationResult, thumb_rgb: Optional[np.ndarray] = None):
        """1枚の画像をギャラリーに追加"""
        if thumb_rgb is None:
            img = cv2.imdecode(np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                thumb = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            else:
                thumb_rgb = np.zeros((120, 160, 3), dtype=np.uint8)

        card = ImageCardWidget(file_path, eval_res, thumb_rgb, self.grid_container)
        card.toggled.connect(self._update_status)
        self.cards.append(card)

        # グリッド配置 (1行あたり最大4〜5列)
        idx = len(self.cards) - 1
        cols = 4
        row = idx // cols
        col = idx % cols
        self.grid_layout.addWidget(card, row, col)

        self._update_status()

    def import_file_paths(self, paths: List[Path]):
        """複数画像ファイルをバックグラウンドで一括品質判定 & 投入"""
        if not paths:
            return

        self.status_label.setText(f"{len(paths)} 枚の画像をQuality Gateで品質判定中...")
        self.btn_import_folder.setEnabled(False)
        self.btn_import_files.setEnabled(False)

        self._worker = QualityBatchWorker(paths, self)
        self._worker.item_evaluated.connect(self.add_image)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished_all.connect(self._on_batch_finished)
        self._worker.start()

    def _on_batch_progress(self, current: int, total: int):
        self.status_label.setText(f"画像品質判定中: {current} / {total} 枚完了...")

    def _on_batch_finished(self, accepted: int, rejected: int):
        self.btn_import_folder.setEnabled(True)
        self.btn_import_files.setEnabled(True)
        self._update_status()

    def _on_import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "画像フォルダを選択 (SDカード等)")
        if folder:
            p = Path(folder)
            exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dng"}
            files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts]
            self.import_file_paths(files)

    def _on_import_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "画像ファイルを選択",
            "",
            "画像ファイル (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.dng)",
        )
        if files:
            self.import_file_paths([Path(f) for f in files])

    def get_selected_images(self) -> List[Path]:
        """再構築キューに投入対象としてチェックされている画像パス一覧を取得"""
        return [c.file_path for c in self.cards if c.is_checked]

    def deselect_rejected_images(self):
        """不合格 (ブレ・特徴不足) の画像のチェックを一括解除"""
        for c in self.cards:
            if not c.eval_res.is_accepted:
                c.set_checked(False)
        self._update_status()

    def set_all_checked(self, checked: bool):
        for c in self.cards:
            c.set_checked(checked)
        self._update_status()

    def update_thresholds(self, blur_threshold: float, min_features: int):
        for c in self.cards:
            c.update_thresholds(blur_threshold, min_features)
        self._update_status()

    def clear_gallery(self):
        for c in self.cards:
            self.grid_layout.removeWidget(c)
            c.deleteLater()
        self.cards.clear()
        self._update_status()

    def _update_status(self):
        total = len(self.cards)
        selected = len(self.get_selected_images())
        accepted = sum(1 for c in self.cards if c.eval_res.is_accepted)
        rejected = total - accepted

        if total == 0:
            self.status_label.setText("画像未読込 (被写体の周囲を360度撮影した画像を30枚以上インポートしてください)")
        else:
            color = "#00e676" if selected >= 30 else "#ffd600"
            warn = " (※推奨30枚以上)" if selected < 30 else " (再構築に十分な枚数です)"
            self.status_label.setText(
                f"総画像数: {total} 枚 | 合格: {accepted} 枚 | 除外推奨: {rejected} 枚 | "
                f"<span style='color: {color};'>再構築キュー選択: {selected} 枚{warn}</span>"
            )
        self.count_changed.emit(selected, total)
