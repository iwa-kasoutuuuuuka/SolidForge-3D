"""
SolidForge 3D - システム設定 & 最適化パラメータ (拡張版)
NVIDIA RTX 5080 (Blackwell) 向けCUDA/TensorRT設定、Smart Shoot Assist、Direct-to-Print 幾何形状処理
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

# ベースディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_DIR = BASE_DIR / "workspace"
DEFAULT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CameraSourceConfig:
    """カメラ接続設定 (SONY / スマホ / USB / Canon / Nikon)"""
    camera_type: Literal["SONY_SDK", "USB_UVC", "SMARTPHONE_IP", "CANON_NIKON", "RTSP_STREAM", "SIMULATOR"] = "SONY_SDK"
    smartphone_url: str = "http://192.168.1.50:8080/video"
    device_index: int = 0
    camera_resolution: Tuple[int, int] = (1920, 1080)
    fps_target: int = 30


@dataclass
class MultiGPUConfig:
    """マルチGPU (複数GPU搭載PC) 設定"""
    mode: Literal["ALL_GPUS", "PRIMARY_ONLY", "CUSTOM"] = "ALL_GPUS"
    active_device_indices: List[int] = field(default_factory=lambda: [0])
    
    def get_colmap_gpu_index_str(self) -> str:
        """COLMAPの--SiftExtraction.gpu_index引数用文字列 (例: '0,1') を返します"""
        if self.mode == "PRIMARY_ONLY" or not self.active_device_indices:
            return "0"
        return ",".join(str(idx) for idx in self.active_device_indices)


@dataclass
class Insta360Config:
    """Insta360 (X5 / X4 / Ace Pro 2 / Link 2) 設定"""
    model_name: str = "Insta360 Ace Pro 2"
    is_360_dewarping_enabled: bool = True  # 360パノラマから透視投影マルチビュー展開
    virtual_cam_fov_deg: float = 90.0      # 展開視野角 (FOV)
    num_virtual_views: int = 4            # 水平方向の展開数 (4方向: 0°, 90°, 180°, 270°)
    output_view_resolution: Tuple[int, int] = (1920, 1080)


@dataclass
class SmartAssistConfig:
    """Smart Shoot Assist (撮影失敗防止) 設定"""
    # オニオンスキン (前フレーム透過重畳) 有効化
    enable_ghost_overlay: bool = True
    ghost_opacity: float = 0.4  # 透過度 (0.1 ~ 0.9)
    # 360度カバレッジ解析
    azimuth_sectors: int = 36  # 360度を何分割して評価するか (10度刻み)
    target_overlap_ratio_min: float = 0.60  # 推奨最小オーバーラップ 60%
    target_overlap_ratio_max: float = 0.80  # 推奨最大オーバーラップ 80%


@dataclass
class AIEnhancementConfig:
    """NVIDIA RTX 5080 TensorRT / ONNX Runtime AI超解像 & デブラー設定"""
    enable_ai_enhancer: bool = True
    backend: Literal["TENSORRT", "CUDA_ONNX", "AUTO"] = "AUTO"
    model_type: Literal["NAFNet_Deblur", "RealESRGAN_x2", "Hybrid_Fast"] = "Hybrid_Fast"
    denoise_strength: float = 0.5  # 高ISOノイズ除去強度
    sharpen_strength: float = 0.7  # 輪郭シャープネス強度
    batch_size: int = 4  # RTX 5080向け並列バッチサイズ


@dataclass
class DirectToPrintConfig:
    """Direct-to-Print 3Dプリント幾何形状処理設定"""
    # RANSAC 接地面自動検出 & フラットカット
    enable_ground_cut: bool = True
    ground_plane_ransac_distance_thresh: float = 1.0  # mm
    ground_cut_offset_mm: float = 0.2  # 接地面より何mm上でカットするか
    # SLA/光造形向け中空化 (Hollowing)
    enable_hollowing: bool = False
    wall_thickness_mm: float = 2.0  # 肉厚 (mm)
    # レジン排出穴 (Drain Holes)
    enable_drain_holes: bool = False
    drain_hole_radius_mm: float = 2.5  # 排出口半径 (mm)
    drain_hole_count: int = 2  # 排出口数


@dataclass
class QualityGateConfig:
    """画像品質判定 (Quality Gate) 設定"""
    blur_threshold: float = 50.0
    min_feature_count: int = 300
    feature_extractor: Literal["SIFT", "ORB"] = "SIFT"
    draw_feature_points: bool = True
    draw_grid_guide: bool = True
    draw_focus_peaking: bool = True


@dataclass
class HardwareOptimizationConfig:
    """NVIDIA RTX 5080 向けCUDA / VRAM 最適化設定"""
    gpu_name: str = "NVIDIA GeForce RTX 5080"
    enable_cuda: bool = True
    gpu_index: int = 0
    colmap_use_gpu: bool = True
    colmap_gpu_index: str = "0"
    colmap_sift_gpu_memory_limit_mb: int = 8192
    openmvs_cuda_threads: int = 16
    openmvs_max_memory_mb: int = 14000
    parallel_matching_threads: int = 8


@dataclass
class PostProcessingConfig:
    """3Dプリント用後処理 & スケール校正設定"""
    aruco_dict_name: str = "DICT_4X4_50"
    aruco_marker_size_mm: float = 50.0
    max_hole_edges: int = 500
    fix_non_manifold: bool = True
    smoothing_iterations: int = 3
    target_faces: int = 250000
    direct_to_print: DirectToPrintConfig = field(default_factory=DirectToPrintConfig)


def _find_colmap_binary() -> str:
    """ローカル bin/colmap またはシステム PATH から COLMAP 実行バイナリ (.exe) を自動検出"""
    candidates = [
        BASE_DIR / "bin" / "colmap" / "bin" / "colmap.exe",
        BASE_DIR / "bin" / "colmap" / "colmap.exe",
        BASE_DIR / "bin" / "colmap" / "bin" / "colmap",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    env_path = os.getenv("COLMAP_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    import shutil
    w = shutil.which("colmap")
    if w:
        return w
    # フォールバック
    if (BASE_DIR / "bin" / "colmap" / "COLMAP.bat").exists():
        return str(BASE_DIR / "bin" / "colmap" / "COLMAP.bat")
    return "colmap"


def _find_openmvs_dir() -> str:
    """ローカル bin/openmvs またはシステム PATH から OpenMVS を自動検出"""
    candidates = [
        BASE_DIR / "bin" / "openmvs",
        BASE_DIR / "bin" / "openmvs" / "bin" / "x64" / "Release",
    ]
    for c in candidates:
        if (c / "InterfaceCOLMAP.exe").exists() or (c / "InterfaceCOLMAP").exists():
            return str(c)
    env_path = os.getenv("OPENMVS_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    return "openmvs"


@dataclass
class AppConfig:
    """SolidForge 3D 全体設定"""
    app_name: str = "SolidForge 3D"
    version: str = "1.1.0"
    workspace_dir: Path = DEFAULT_WORKSPACE_DIR
    
    colmap_binary: str = field(default_factory=_find_colmap_binary)
    openmvs_dir: str = field(default_factory=_find_openmvs_dir)
    
    quality: QualityGateConfig = field(default_factory=QualityGateConfig)
    hardware: HardwareOptimizationConfig = field(default_factory=HardwareOptimizationConfig)
    post_process: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    smart_assist: SmartAssistConfig = field(default_factory=SmartAssistConfig)
    ai_enhancement: AIEnhancementConfig = field(default_factory=AIEnhancementConfig)
    camera: CameraSourceConfig = field(default_factory=CameraSourceConfig)
    insta360: Insta360Config = field(default_factory=Insta360Config)
    multi_gpu: MultiGPUConfig = field(default_factory=MultiGPUConfig)
    
    default_export_format: str = "STL (.stl)"


CONFIG = AppConfig()
