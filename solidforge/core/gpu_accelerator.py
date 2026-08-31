"""
SolidForge 3D - Multi-GPU Accelerator (RTX 5080 / Multi-GPU 極限CUDA/FP16 高速化エンジン)
PyTorch Multi-CUDA テンソルによるGPU直接ラプラシアン演算 & 複数GPU並列バッチAI処理
"""

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    CUDA_COUNT = torch.cuda.device_count() if HAS_CUDA else 0
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False
    CUDA_COUNT = 0

from solidforge.config import CONFIG


class GPUAccelerator:
    """
    NVIDIA RTX 5080 & Multi-GPU (複数GPU並列クラスタ) 向けアクセラレータ
    - 複数GPU間での並列AIバッチ分散処理 (DataParallel)
    - GPU VRAM上で直接ラプラシアン鮮鋭度を算出 (Zero-Copy)
    - FP16 (Half Precision) Mixed-Precision バッチ超高速処理
    """

    def __init__(self):
        self.has_cuda = HAS_CUDA
        self.cuda_count = CUDA_COUNT
        self._laplacian_weights: Dict[int, torch.Tensor] = {}
        self._sharpen_weights: Dict[int, torch.Tensor] = {}
        self._init_gpu_kernels()

    def _init_gpu_kernels(self):
        """利用可能な全GPU上に畳み込みカーネルを初期化"""
        if self.has_cuda and HAS_TORCH:
            for dev_id in range(max(self.cuda_count, 1)):
                try:
                    dev = torch.device(f"cuda:{dev_id}")
                    k_lap = torch.tensor(
                        [[[[0.0, 1.0, 0.0],
                           [1.0, -4.0, 1.0],
                           [0.0, 1.0, 0.0]]]],
                        dtype=torch.float16,
                        device=dev,
                    )
                    self._laplacian_weights[dev_id] = k_lap

                    k_sharp = torch.tensor(
                        [[[[-0.1, -0.1, -0.1],
                           [-0.1,  1.8, -0.1],
                           [-0.1, -0.1, -0.1]]]],
                        dtype=torch.float16,
                        device=dev,
                    )
                    self._sharpen_weights[dev_id] = k_sharp
                except Exception:
                    pass

    def calculate_blur_score_gpu(self, image: np.ndarray, device_id: int = 0) -> float:
        """
        GPU VRAM上で直接ラプラシアン畳み込みと分散を計算し、高速に鮮鋭度スコアを算出します。
        """
        if image is None or image.size == 0:
            return 0.0

        if not (self.has_cuda and HAS_TORCH and device_id in self._laplacian_weights):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

        try:
            dev = torch.device(f"cuda:{device_id}")
            if len(image.shape) == 3:
                tensor = torch.from_numpy(image).to(device=dev, dtype=torch.float16)
                gray = 0.114 * tensor[:, :, 0] + 0.587 * tensor[:, :, 1] + 0.299 * tensor[:, :, 2]
            else:
                gray = torch.from_numpy(image).to(device=dev, dtype=torch.float16)

            gray_4d = gray.unsqueeze(0).unsqueeze(0)
            lap = F.conv2d(gray_4d, self._laplacian_weights[device_id], padding=1)
            variance = torch.var(lap).item()
            return round(float(variance), 2)
        except Exception:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)

    def enhance_batch_cuda_fp16(
        self,
        images: List[np.ndarray],
        sharpen_strength: float = 0.7,
        denoise_strength: float = 0.5,
    ) -> List[np.ndarray]:
        """
        複数GPUが利用可能な場合、画像をGPU間で分割して並列バッチ処理を実行します。
        """
        if not images:
            return []

        if not (self.has_cuda and HAS_TORCH):
            from solidforge.core.ai_enhancer import AI_ENHANCER
            return [AI_ENHANCER.enhance_image(img) for img in images]

        active_gpus = CONFIG.multi_gpu.active_device_indices
        if not active_gpus:
            active_gpus = [0]

        # 複数GPUへの分割処理 (Multi-GPU Data Parallel)
        if len(active_gpus) > 1 and len(images) >= len(active_gpus):
            chunks = np.array_split(images, len(active_gpus))
            futures = []
            with ThreadPoolExecutor(max_workers=len(active_gpus)) as executor:
                for dev_id, chunk in zip(active_gpus, chunks):
                    if len(chunk) > 0:
                        futures.append(executor.submit(
                            self._enhance_chunk_on_device,
                            list(chunk),
                            dev_id,
                            sharpen_strength,
                            denoise_strength,
                        ))
            results = []
            for f in futures:
                results.extend(f.result())
            return results
        else:
            return self._enhance_chunk_on_device(
                images,
                active_gpus[0],
                sharpen_strength,
                denoise_strength,
            )

    def _enhance_chunk_on_device(
        self,
        images: List[np.ndarray],
        device_id: int,
        sharpen_strength: float,
        denoise_strength: float,
    ) -> List[np.ndarray]:
        """単一GPUデバイス上でのFP16バッチ処理"""
        try:
            dev = torch.device(f"cuda:{device_id}")
            h, w = images[0].shape[:2]
            batch_np = np.stack([cv2.resize(img, (w, h)) if img.shape[:2] != (h, w) else img for img in images])
            tensor_batch = (
                torch.from_numpy(batch_np)
                .permute(0, 3, 1, 2)
                .to(device=dev, dtype=torch.float16)
            )

            # 空間平滑化 (ノイズ低減)
            if denoise_strength > 0:
                kernel_blur = torch.ones((1, 1, 3, 3), device=dev, dtype=torch.float16) / 9.0
                kernel_blur_3ch = kernel_blur.repeat(3, 1, 1, 1)
                denoised = F.conv2d(tensor_batch, kernel_blur_3ch, padding=1, groups=3)
            else:
                denoised = tensor_batch

            # 高周波エッジ復元
            high_freq = tensor_batch - denoised
            enhanced = tensor_batch + high_freq * (sharpen_strength * 1.5)
            enhanced = torch.clamp(enhanced, 0.0, 255.0)

            out_np = enhanced.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            return [out_np[i] for i in range(len(images))]
        except Exception:
            from solidforge.core.ai_enhancer import AI_ENHANCER
            return [AI_ENHANCER.enhance_image(img) for img in images]


GPU_ACCELERATOR = GPUAccelerator()
