"""
SolidForge 3D - Camera Selection & Smartphone Pairing Dialog
スマートフォン (WiFi/USB IP Webcam) / SONY / Canon / Nikon / USB UVC / RTSP カメラ接続ダイアログ
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from solidforge.config import CONFIG
from solidforge.core.camera_manager import CAMERA_MANAGER


class CameraSelectDialog(QDialog):
    """
    カメラ選択 & スマートフォン接続設定ダイアログ
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📷 カメラ選択 & スマートフォン接続設定")
        self.resize(560, 420)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # ヘッダー説明
        title = QLabel("使用するカメラソースを選択してください")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #00e5ff;")
        layout.addWidget(title)

        # タブコンテナ
        self.tabs = QTabWidget()

        # タブ1: スマートフォン (WiFi / IP Webcam)
        tab_phone = self._create_smartphone_tab()
        self.tabs.addTab(tab_phone, "📱 スマートフォン (WiFi / IP)")

        # タブ2: Insta360 (X5 / Ace Pro 2)
        tab_insta = self._create_insta360_tab()
        self.tabs.addTab(tab_insta, "🌀 Insta360 (X5 / Ace Pro 2)")

        # タブ3: SONY α / ZV-E10
        tab_sony = self._create_sony_tab()
        self.tabs.addTab(tab_sony, "📷 SONY α / ZV-E10")

        # タブ4: USBカメラ / 一眼レフ (Canon, Nikon, キャプチャカード)
        tab_usb = self._create_usb_tab()
        self.tabs.addTab(tab_usb, "🎥 USB / 一眼レフ (Canon/Nikon)")

        # タブ5: シミュレータ
        tab_sim = self._create_simulator_tab()
        self.tabs.addTab(tab_sim, "🧪 3Dシミュレータ")

        layout.addWidget(self.tabs, stretch=1)

        # アクションボタンバー
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()

        self.btn_cancel = QPushButton("キャンセル")
        self.btn_cancel.clicked.connect(self.reject)
        btn_bar.addWidget(self.btn_cancel)

        self.btn_connect = QPushButton("⚡ このカメラで接続開始")
        self.btn_connect.setObjectName("primaryActionBtn")
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        btn_bar.addWidget(self.btn_connect)

        layout.addLayout(btn_bar)

    def _create_smartphone_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel(
            "<b>iPhone / Android をフォトグラメトリ用カメラとして使用します。</b><br>"
            "スマホアプリ（例: 『IP Webcam』『DroidCam』『Camo』等）を起動し、"
            "画面に表示されたストリームURLを入力してください。"
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # プリセット選択
        preset_box = QHBoxLayout()
        preset_lbl = QLabel("プリセット:")
        preset_lbl.setStyleSheet("color: #00e5ff; font-weight: bold;")
        preset_box.addWidget(preset_lbl)

        self.combo_phone_presets = QComboBox()
        self.combo_phone_presets.addItems([
            "Android IP Webcam (http://192.168.1.50:8080/video)",
            "DroidCam (http://192.168.1.50:4747/video)",
            "RTSP 汎用ストリーム (rtsp://192.168.1.50:8554/live)",
            "カスタム URL",
        ])
        self.combo_phone_presets.currentIndexChanged.connect(self._on_phone_preset_changed)
        preset_box.addWidget(self.combo_phone_presets, stretch=1)
        layout.addLayout(preset_box)

        # URL 入力
        form = QFormLayout()
        self.txt_phone_url = QLineEdit(CONFIG.camera.smartphone_url)
        self.txt_phone_url.setPlaceholderText("http://192.168.x.x:8080/video")
        form.addRow("ストリーム URL:", self.txt_phone_url)
        layout.addLayout(form)

        layout.addStretch()
        return widget

    def _create_insta360_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel(
            "<b>Insta360 シリーズ (X5 / X4 / Ace Pro 2 / Link 2)</b><br>"
            "カメラ本体の『USBモード』を『<b>ウェブカメラ (UVC)</b>』に設定してUSB-C接続するか、"
            "WiFi RTSPライブストリームを指定して接続します。"
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # モデル選択
        m_box = QHBoxLayout()
        m_lbl = QLabel("機種・モデル:")
        m_lbl.setStyleSheet("color: #00e5ff; font-weight: bold;")
        m_box.addWidget(m_lbl)

        self.combo_insta_model = QComboBox()
        self.combo_insta_model.addItems([
            "Insta360 Ace Pro 2 (USB UVC 4K/8K)",
            "Insta360 Ace Pro (USB UVC 4K)",
            "Insta360 X5 (360° USB / 4方向自動展開)",
            "Insta360 X4 / X3 (360° USB / 4方向自動展開)",
            "Insta360 Link 2 / Link (4K AI PTZ)",
            "Insta360 GO 3S (超小型アクション)",
        ])
        m_box.addWidget(self.combo_insta_model, stretch=1)
        layout.addLayout(m_box)

        # 360パノラマ展開オプション (X5 / X4向け)
        from PySide6.QtWidgets import QCheckBox
        self.chk_insta_dewarp = QCheckBox("🌐 360°パノラマから透視投影マルチビュー自動展開 (4方向: 前/右/後/左)")
        self.chk_insta_dewarp.setChecked(True)
        self.chk_insta_dewarp.setStyleSheet("color: #00e676; font-weight: bold;")
        layout.addWidget(self.chk_insta_dewarp)

        # WiFi RTSPストリーム設定 (オプション)
        form = QFormLayout()
        self.txt_insta_rtsp = QLineEdit()
        self.txt_insta_rtsp.setPlaceholderText("WiFi接続時のみ入力 (例: rtsp://192.168.42.1/live)")
        form.addRow("WiFi RTSP URL (任意):", self.txt_insta_rtsp)
        layout.addLayout(form)

        layout.addStretch()
        return widget

    def _create_sony_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel(
            "<b>SONY ZV-E10 / αシリーズ 有線接続</b><br>"
            "カメラをUSBケーブルでPCに接続し、カメラ側メニューで「PCリモート」または「USBストリーミング」を選択してください。"
            "SONY Camera Remote SDKによる高精度制御またはUSB UVC直接キャプチャが行われます。"
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 12px; line-height: 1.4;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        return widget

    def _create_usb_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel(
            "<b>PC接続 USBカメラ / HDMIキャプチャカード / 一眼レフ</b><br>"
            "Canon EOS Webcam Utility、Nikon Webcam Utility、Elgato Cam Link 4K等で接続されたデバイスを選択します。"
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        dev_row = QHBoxLayout()
        self.combo_usb_devices = QComboBox()
        dev_row.addWidget(self.combo_usb_devices, stretch=1)

        self.btn_rescan = QPushButton("🔄 再スキャン")
        self.btn_rescan.clicked.connect(self._scan_usb_devices)
        dev_row.addWidget(self.btn_rescan)
        layout.addLayout(dev_row)

        self._scan_usb_devices()

        layout.addStretch()
        return widget

    def _create_simulator_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info = QLabel(
            "<b>3Dフォトグラメトリ シミュレータ</b><br>"
            "実機カメラが手元にない場合でも、テクスチャ付き多面体とArUcoマーカーを持つ擬似Live Viewで"
            "全機能（オニオンスキン、ブレ検知、3D再構築、Watertight修復）をテストできます。"
        )
        info.setStyleSheet("color: #cbd5e1; font-size: 12px; line-height: 1.4;")
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addStretch()
        return widget

    def _scan_usb_devices(self):
        """USBデバイスをスキャンしてドロップダウンに設定"""
        self.combo_usb_devices.clear()
        devices = CAMERA_MANAGER.list_available_cameras(max_devices=4)
        if devices:
            for dev in devices:
                self.combo_usb_devices.addItem(dev.name, dev.device_id)
        else:
            self.combo_usb_devices.addItem("検出されたカメラ (既定 Device #0)", 0)

    def _on_phone_preset_changed(self, index: int):
        if index == 0:
            self.txt_phone_url.setText("http://192.168.1.50:8080/video")
        elif index == 1:
            self.txt_phone_url.setText("http://192.168.1.50:4747/video")
        elif index == 2:
            self.txt_phone_url.setText("rtsp://192.168.1.50:8554/live")

    def _on_connect_clicked(self):
        current_tab_idx = self.tabs.currentIndex()

        if current_tab_idx == 0:  # スマホ (WiFi / IP)
            url = self.txt_phone_url.text().strip()
            if not url:
                QMessageBox.warning(self, "警告", "ストリームURLを入力してください。")
                return
            CONFIG.camera.smartphone_url = url
            CONFIG.camera.camera_type = "SMARTPHONE_IP"
            success = CAMERA_MANAGER.connect_smartphone_ip(url)
            if not success:
                QMessageBox.critical(
                    self,
                    "接続失敗",
                    f"スマートフォンストリームへの接続に失敗しました:\n{url}\n\n"
                    "スマホ側アプリが起動しており、同じWiFiネットワークに接続されているかご確認ください。"
                )
                return

        elif current_tab_idx == 1:  # Insta360
            model = self.combo_insta_model.currentText()
            rtsp = self.txt_insta_rtsp.text().strip()
            CONFIG.insta360.model_name = model
            CONFIG.insta360.is_360_dewarping_enabled = self.chk_insta_dewarp.isChecked()
            CONFIG.camera.camera_type = "USB_UVC"
            if rtsp:
                CAMERA_MANAGER.connect_insta360(model_name=model, rtsp_url=rtsp)
            else:
                CAMERA_MANAGER.connect_insta360(model_name=model)

        elif current_tab_idx == 2:  # SONY
            CONFIG.camera.camera_type = "SONY_SDK"
            CAMERA_MANAGER.connect_sony_sdk()

        elif current_tab_idx == 3:  # USB / Canon / Nikon
            dev_id = self.combo_usb_devices.currentData() or 0
            CONFIG.camera.device_index = dev_id
            CONFIG.camera.camera_type = "USB_UVC"
            CAMERA_MANAGER.connect_usb_uvc(device_id=dev_id)

        elif current_tab_idx == 4:  # シミュレータ
            CONFIG.camera.camera_type = "SIMULATOR"
            CAMERA_MANAGER.connect_simulator()

        self.accept()
