"""
SolidForge 3D - Hardware Optimizer & Multi-GPU Telemetry
複数GPU (Multi-GPU CUDA クラスタ) 自動検出・動的VRAMプール管理・ハードウェア自動チューニング
"""

from dataclasses import dataclass
import os
import psutil
from typing import Dict, List, Optional, Tuple

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
    CUDA_DEVICE_COUNT = torch.cuda.device_count() if HAS_TORCH_CUDA else 0
except ImportError:
    HAS_TORCH_CUDA = False
    CUDA_DEVICE_COUNT = 0

from solidforge.config import CONFIG, HardwareOptimizationConfig


@dataclass
class GPUTelemetry:
    """GPU個別テレメトリ情報"""
    device_id: int
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
    マルチGPU & RTX 5080 ハードウェア自動チューニング & テレメトリマネージャー
    """

    def __init__(self):
        self._last_telemetries: List[GPUTelemetry] = []

    def list_all_gpus(self) -> List[GPUTelemetry]:
        """
        PCに搭載されているすべてのGPUを走査し、テレメトリ一覧を返します。
        """
        cpu_usage = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        cpu_cores = os.cpu_count() or 8
        results: List[GPUTelemetry] = []

        if HAS_TORCH_CUDA and CUDA_DEVICE_COUNT > 0:
            for dev_id in range(CUDA_DEVICE_COUNT):
                try:
                    gpu_name = torch.cuda.get_device_name(dev_id)
                    total_bytes = torch.cuda.get_device_properties(dev_id).total_memory
                    reserved_bytes = torch.cuda.memory_reserved(dev_id)

                    total_mb = int(total_bytes / (1024 * 1024))
                    used_mb = int(reserved_bytes / (1024 * 1024))
                    free_mb = max(0, total_mb - used_mb)
                    vram_pct = round((used_mb / max(total_mb, 1)) * 100.0, 1)

                    cap = torch.cuda.get_device_capability(dev_id)
                    compute_cap = f"Compute {cap[0]}.{cap[1]} (Blackwell/Ada)"

                    results.append(GPUTelemetry(
                        device_id=dev_id,
                        gpu_name=gpu_name,
                        is_cuda_available=True,
                        vram_total_mb=total_mb,
                        vram_used_mb=used_mb,
                        vram_free_mb=free_mb,
                        vram_usage_percent=vram_pct,
                        cuda_compute_capability=compute_cap,
                        cpu_cores_logical=cpu_cores,
                        cpu_usage_percent=cpu_usage,
                        ram_usage_percent=ram.percent,
                    ))
                except Exception:
                    pass
        else:
            # CPU/エミュレーション時またはテスト用
            results.append(GPUTelemetry(
                device_id=0,
                gpu_name=CONFIG.hardware.gpu_name,
                is_cuda_available=False,
                vram_total_mb=16384,
                vram_used_mb=1024,
                vram_free_mb=15360,
                vram_usage_percent=6.2,
                cuda_compute_capability="CUDA Emulation",
                cpu_cores_logical=cpu_cores,
                cpu_usage_percent=cpu_usage,
                ram_usage_percent=ram.percent,
            ))

        self._last_telemetries = results
        return results

    def get_telemetry(self, device_id: int = 0) -> GPUTelemetry:
        """指定したGPU (既定: 0) のテレメトリを取得"""
        all_gpus = self.list_all_gpus()
        for g in all_gpus:
            if g.device_id == device_id:
                return g
        return all_gpus[0]

    def auto_tune_configuration(self) -> HardwareOptimizationConfig:
        """搭載GPU数およびVRAMプールに合わせてパイプライン設定を自動最適化"""
        all_gpus = self.list_all_gpus()
        hw = CONFIG.hardware
        multi_cfg = CONFIG.multi_gpu

        active_indices = [g.device_id for g in all_gpus if g.is_cuda_available]
        if not active_indices:
            active_indices = [0]
        multi_cfg.active_device_indices = active_indices

        if all_gpus[0].is_cuda_available:
            hw.gpu_name = all_gpus[0].gpu_name
            if len(all_gpus) > 1:
                hw.gpu_name += f" (+{len(all_gpus)-1}基 Multi-GPU)"

            hw.enable_cuda = True
            hw.colmap_use_gpu = True

            # マルチGPU合算VRAMプール容量を計算
            total_vram_pool = sum(g.vram_total_mb for g in all_gpus)

            if total_vram_pool >= 30000:
                hw.colmap_sift_gpu_memory_limit_mb = 24576
                hw.openmvs_max_memory_mb = 32000
                hw.openmvs_cuda_threads = min(64, all_gpus[0].cpu_cores_logical * 4)
            elif total_vram_pool >= 20000:
                hw.colmap_sift_gpu_memory_limit_mb = 16384
                hw.openmvs_max_memory_mb = 20000
                hw.openmvs_cuda_threads = min(32, all_gpus[0].cpu_cores_logical * 2)
            else:
                hw.colmap_sift_gpu_memory_limit_mb = 10240
                hw.openmvs_max_memory_mb = 14000
                hw.openmvs_cuda_threads = min(16, all_gpus[0].cpu_cores_logical)

            hw.parallel_matching_threads = max(4, all_gpus[0].cpu_cores_logical // 2)

        return hw


HARDWARE_OPTIMIZER = HardwareOptimizer()
