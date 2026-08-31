"""
SolidForge 3D - Hardware Optimizer & GPU Telemetry
RTX 5080 ハードウェア自動チューニング、動的VRAM管理、リアルタイムGPU状態監視
"""

from dataclasses import dataclass
import os
import psutil
from typing import Dict, Optional, Tuple

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

from solidforge.config import CONFIG, HardwareOptimizationConfig


@dataclass
class GPUTelemetry:
    """GPUテレメトリ情報"""
    gpu_name: str
    is_cuda_available: bool
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    vram_usage_percent: float
    cuda_compute_capability: str
    cpu_cores_logical: int
    cpu_usage_percent: float
    ram_usage_percent: float


class HardwareOptimizer:
    """
    RTX 5080 向けハードウェア自動チューニング & テレメトリマネージャー
    """

    def __init__(self):
        self._last_telemetry: Optional[GPUTelemetry] = None

    def get_telemetry(self) -> GPUTelemetry:
        """リアルタイムのGPU/CPU/RAM状態を取得"""
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        cpu_cores = os.cpu_count() or 8

        if HAS_TORCH_CUDA:
            device_id = 0
            gpu_name = torch.cuda.get_device_name(device_id)
            total_bytes = torch.cuda.get_device_properties(device_id).total_memory
            allocated_bytes = torch.cuda.memory_allocated(device_id)
            reserved_bytes = torch.cuda.memory_reserved(device_id)
            
            total_mb = int(total_bytes / (1024 * 1024))
            used_mb = int(reserved_bytes / (1024 * 1024))
            free_mb = max(0, total_mb - used_mb)
            vram_pct = round((used_mb / max(total_mb, 1)) * 100.0, 1)

            cap = torch.cuda.get_device_capability(device_id)
            compute_cap = f"Compute {cap[0]}.{cap[1]} (Blackwell/Ada)"
            is_cuda = True
        else:
            gpu_name = CONFIG.hardware.gpu_name
            total_mb = 16384  # デフォルト 16GB
            used_mb = 1024
            free_mb = 15360
            vram_pct = 6.2
            compute_cap = "CUDA Emulation"
            is_cuda = False

        self._last_telemetry = GPUTelemetry(
            gpu_name=gpu_name,
            is_cuda_available=is_cuda,
            vram_total_mb=total_mb,
            vram_used_mb=used_mb,
            vram_free_mb=free_mb,
            vram_usage_percent=vram_pct,
            cuda_compute_capability=compute_cap,
            cpu_cores_logical=cpu_cores,
            cpu_usage_percent=cpu_usage,
            ram_usage_percent=ram.percent,
        )
        return self._last_telemetry

    def auto_tune_configuration(self) -> HardwareOptimizationConfig:
        """搭載ハードウェアに合わせてパイプライン設定を自動最適化"""
        telemetry = self.get_telemetry()
        hw = CONFIG.hardware

        if telemetry.is_cuda_available:
            hw.gpu_name = telemetry.gpu_name
            hw.enable_cuda = True
            hw.colmap_use_gpu = True
            
            # VRAM容量に応じて動的スケーリング (16GBなら10GB、24GBなら16GBを割り当て)
            if telemetry.vram_total_mb >= 20000:
                hw.colmap_sift_gpu_memory_limit_mb = 16384
                hw.openmvs_max_memory_mb = 20000
                hw.openmvs_cuda_threads = min(32, telemetry.cpu_cores_logical * 2)
            else:
                hw.colmap_sift_gpu_memory_limit_mb = 10240
                hw.openmvs_max_memory_mb = 14000
                hw.openmvs_cuda_threads = min(16, telemetry.cpu_cores_logical)

            hw.parallel_matching_threads = max(4, telemetry.cpu_cores_logical // 2)

        return hw


HARDWARE_OPTIMIZER = HardwareOptimizer()
