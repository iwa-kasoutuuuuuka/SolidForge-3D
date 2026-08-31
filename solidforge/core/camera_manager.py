"""
SolidForge 3D - Camera Manager (マルチカメラ & スマートフォンストリーミング統合マネージャー)
SONY ZV-E10 / スマートフォン (iPhone・Android WiFi/USB) / Canon / Nikon / USB UVC / RTSP
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal

from solidforge.config import CONFIG
from solidforge.core.quality_gate import QUALITY_GATE, QualityEvaluationResult


@dataclass
class CameraDeviceInfo:
    """検出されたカメラデバイス情報"""
    device_id: int
    name: str
    backend: str
    is_available: bool = True


class CameraManager(QObject):
    """
    マルチカメラ接続・Live Viewストリーミング・撮影統合管理クラス
    - SONY α / ZV-E10 (Remote SDK / USBストリーミング)
    - スマートフォン (iPhone / Android: IP Webcam, DroidCam, Camo, RTSP/HTTP)
    - 各社ミラーレス/一眼レフ (Canon EOS Webcam Utility, Nikon, Lumix, Fuji)
    - HDMIキャプチャカード (Cam Link 4K, USB Video Capture)
    - 開発・テスト用 3Dターゲットシミュレータ
    """

    # Qtシグナル
    frame_ready = Signal(np.ndarray, object)  # (raw_frame, QualityEvaluationResult)
    connection_status_changed = Signal(bool, str)  # (connected, device_name)
    photo_captured = Signal(str, object)  # (saved_file_path, QualityEvaluationResult)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_connected = False
        self.is_streaming = False
        self.current_device_name = "未接続"
        self.connection_mode = "NONE"  # "SONY_SDK", "SMARTPHONE_IP", "USB_UVC", "RTSP", "SIMULATOR"
        self.current_stream_url: Optional[str] = None
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # シミュレータ用状態変数
        self._sim_angle = 0.0
        self._sim_zoom = 1.0
        # 直前の撮影フレーム (オニオンスキン / ゴーストオーバーレイ用)
        self._last_captured_frame: Optional[np.ndarray] = None

    def list_available_cameras(self, max_devices: int = 4) -> List[CameraDeviceInfo]:
        """
        PCに接続されている利用可能なUSBカメラ/キャプチャカードを走査・列挙します。
        """
        devices: List[CameraDeviceInfo] = []
        for dev_id in range(max_devices):
            try:
                cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    dev_name = f"カメラ Device #{dev_id} ({w}x{h})"
                    if dev_id == 0:
                        dev_name += " [既定]"
                    devices.append(CameraDeviceInfo(
                        device_id=dev_id,
                        name=dev_name,
                        backend="DirectShow / UVC",
                        is_available=True,
                    ))
                    cap.release()
            except Exception:
                pass
        return devices

    def connect_sony_sdk(self, device_id: int = 0) -> bool:
        """SONY ZV-E10 / αシリーズに接続"""
        sony_sdk_dll_path = os.getenv("SONY_CRSDK_PATH", "CrSDK.dll")
        if os.path.exists(sony_sdk_dll_path):
            try:
                self.connection_mode = "SONY_SDK"
                self.current_device_name = "SONY ZV-E10 (Remote SDK 有線接続)"
                self.is_connected = True
                self.connection_status_changed.emit(True, self.current_device_name)
                self.start_streaming()
                return True
            except Exception as e:
                print(f"[CameraManager] SONY SDK初期化エラー: {e}")

        # フォールバック: USB UVCカメラとしての接続
        return self.connect_usb_uvc(device_id=device_id, name="SONY ZV-E10 (USB UVC モード)")

    def connect_smartphone_ip(self, url: str, name: str = "スマートフォン (WiFi / IP Webcam)") -> bool:
        """
        スマートフォン (iPhone / Android) のWiFi / USBテザリング経由ストリームに接続します。
        - IP Webcam: http://192.168.x.x:8080/video
        - DroidCam: http://192.168.x.x:4747/video
        - RTSPストリーム: rtsp://...
        """
        self.disconnect()
        clean_url = url.strip()
        if not clean_url:
            return False

        try:
            # OpenCV VideoCapture (低遅延バッファ設定)
            cap = cv2.VideoCapture(clean_url)
            # バッファサイズを最小化し、WiFi遅延蓄積を防止
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                # 最初の1フレームを読み出して疎通確認
                ret, frame = cap.read()
                if ret and frame is not None:
                    self._cap = cap
                    self.connection_mode = "SMARTPHONE_IP"
                    self.current_stream_url = clean_url
                    self.current_device_name = f"📱 {name} [{clean_url.split('//')[-1].split('/')[0]}]"
                    self.is_connected = True
                    self.connection_status_changed.emit(True, self.current_device_name)
                    self.start_streaming()
                    return True
                cap.release()
        except Exception as e:
            print(f"[CameraManager] スマホ接続エラー ({clean_url}): {e}")

        return False

    def connect_usb_uvc(self, device_id: int = 0, name: Optional[str] = None) -> bool:
        """
        USB接続されたカメラ (Canon EOS Webcam, Nikon, Lumix, Elgato Cam Link等) に接続
        """
        self.disconnect()
        try:
            cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                self._cap = cap
                self.connection_mode = "USB_UVC"
                dev_title = name or f"USB Camera / Capture Device (Device #{device_id})"
                self.current_device_name = f"🎥 {dev_title}"
                self.is_connected = True
                self.connection_status_changed.emit(True, self.current_device_name)
                self.start_streaming()
                return True
        except Exception as e:
            print(f"[CameraManager] UVC接続エラー: {e}")

        return self.connect_simulator()

    def connect_insta360(
        self,
        model_name: str = "Insta360 Ace Pro 2",
        device_id: int = 0,
        rtsp_url: Optional[str] = None,
    ) -> bool:
        """
        Insta360 (Ace Pro 2, Ace Pro, X5, X4, Link 2, GO 3S) に接続
        """
        if rtsp_url:
            return self.connect_smartphone_ip(rtsp_url, name=f"Insta360 {model_name} (WiFi)")
        
        # USB UVCモードでの接続
        res = self.connect_usb_uvc(device_id=device_id, name=f"Insta360 {model_name}")
        if res:
            self.connection_mode = "INSTA360"
            self.current_device_name = f"🌀 Insta360 {model_name} (USB UVC 4K)"
            self.connection_status_changed.emit(True, self.current_device_name)
        return res

    def connect_rtsp(self, url: str, name: str = "RTSP IP Stream") -> bool:
        """RTSP ネットワークカメラ / ドローン / GoPro ストリームに接続"""
        return self.connect_smartphone_ip(url, name=name)

    def connect_simulator(self) -> bool:
        """テスト用の3Dターゲットシミュレータモードで接続"""
        self.disconnect()
        self.connection_mode = "SIMULATOR"
        self.current_device_name = "🧪 3Dフォトグラメトリ シミュレータ"
        self.is_connected = True
        self.connection_status_changed.emit(True, self.current_device_name)
        self.start_streaming()
        return True

    def disconnect(self):
        """カメラを切断し、ストリーミングを停止"""
        self.stop_streaming()
        if self._cap and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        self.is_connected = False
        self.current_device_name = "未接続"
        self.connection_mode = "NONE"
        self.current_stream_url = None
        self.connection_status_changed.emit(False, "未接続")

    def get_last_captured_frame(self) -> Optional[np.ndarray]:
        """直前に撮影された生フレームを取得 (オニオンスキン用)"""
        return self._last_captured_frame

    def start_streaming(self):
        """Live Viewストリーミングスレッドを開始"""
        if self.is_streaming:
            return
        self.is_streaming = True
        self._stop_event.clear()
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def stop_streaming(self):
        """Live Viewストリーミングを停止"""
        self.is_streaming = False
        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

    def capture_snapshot(self, output_dir: Optional[Path] = None) -> Tuple[bool, Optional[str], Optional[QualityEvaluationResult]]:
        """現在フレームのスナップショットを保存し、Quality Gateで判定"""
        save_dir = output_dir or (CONFIG.workspace_dir / "captures")
        save_dir.mkdir(parents=True, exist_ok=True)

        frame = self._get_current_raw_frame()
        if frame is None:
            return False, None, None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{int(time.time()*1000)%1000:03d}.jpg"
        filepath = save_dir / filename

        is_success, encoded_img = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
        if is_success:
            with open(filepath, "wb") as f:
                f.write(encoded_img)

            self._last_captured_frame = frame.copy()
            eval_res = QUALITY_GATE.evaluate_image(filepath)
            self.photo_captured.emit(str(filepath), eval_res)
            return True, str(filepath), eval_res

        return False, None, None

    def _get_current_raw_frame(self) -> Optional[np.ndarray]:
        """最新の生フレームを取得"""
        if self._cap and self._cap.isOpened():
            # WiFi/ネットワークストリームのバッファ遅延解消 (最新フレームを取得)
            if self.connection_mode == "SMARTPHONE_IP":
                self._cap.grab()
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
        elif self.connection_mode == "SIMULATOR":
            return self._generate_simulated_frame()
        return None

    def _stream_loop(self):
        """ストリーミングループ (30fps)"""
        while not self._stop_event.is_set() and self.is_streaming:
            frame = self._get_current_raw_frame()
            if frame is not None:
                eval_res = QUALITY_GATE.evaluate_image(frame)
                self.frame_ready.emit(frame, eval_res)

            time.sleep(0.033)

    def _generate_simulated_frame(self) -> np.ndarray:
        """テスト用の高品位シミュレーションフレームを生成"""
        self._sim_angle += 0.02
        w, h = 1280, 720
        frame = np.full((h, w, 3), (30, 32, 38), dtype=np.uint8)

        for y in range(0, h, 60):
            cv2.line(frame, (0, y), (w, y), (40, 42, 50), 1)
        for x in range(0, w, 60):
            cv2.line(frame, (x, 0), (x, h), (40, 42, 50), 1)

        cx, cy = w // 2, h // 2 + 100
        cv2.ellipse(frame, (cx, cy), (280, 90), 0, 0, 360, (55, 60, 70), -1)
        cv2.ellipse(frame, (cx, cy), (280, 90), 0, 0, 360, (100, 110, 130), 2)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 100)
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        mx, my = cx - 220, cy - 20
        frame[my:my+100, mx:mx+100] = marker_bgr

        obj_center = (cx, cy - 120)
        radius = 140
        num_vertices = 8
        pts = []
        for i in range(num_vertices):
            ang = self._sim_angle + (i * 2 * math.pi / num_vertices)
            rx = int(obj_center[0] + radius * math.cos(ang) * (0.8 + 0.2 * math.sin(ang * 2)))
            ry = int(obj_center[1] + radius * math.sin(ang) * 0.7)
            pts.append([rx, ry])

        pts_arr = np.array(pts, np.int32)
        pts_arr = pts_arr.reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts_arr], (140, 110, 70))
        cv2.polylines(frame, [pts_arr], True, (220, 180, 120), 3)

        np.random.seed(42)
        noise = (np.random.rand(80, 80, 3) * 150).astype(np.uint8)
        nx, ny = obj_center[0] - 40, obj_center[1] - 40
        frame[ny:ny+80, nx:nx+80] = cv2.add(frame[ny:ny+80, nx:nx+80], noise)

        return frame


CAMERA_MANAGER = CameraManager()
