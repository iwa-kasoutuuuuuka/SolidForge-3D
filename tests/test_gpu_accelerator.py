"""
Unit Tests for GPU Accelerator & Hardware Optimizer Modules
"""

import unittest
import numpy as np
import cv2
from solidforge.core.gpu_accelerator import GPUAccelerator
from solidforge.core.hardware_optimizer import HardwareOptimizer


class TestGPUAcceleratorAndOptimizer(unittest.TestCase):
    def setUp(self):
        self.gpu_acc = GPUAccelerator()
        self.optimizer = HardwareOptimizer()

    def test_gpu_blur_score_calculation(self):
        """GPU直接ラプラシアン鮮鋭度演算のテスト"""
        # シャープ画像
        sharp = np.zeros((300, 300, 3), dtype=np.uint8)
        for y in range(0, 300, 30):
            for x in range(0, 300, 30):
                if (x // 30 + y // 30) % 2 == 0:
                    sharp[y:y+30, x:x+30] = 255

        score_sharp = self.gpu_acc.calculate_blur_score_gpu(sharp)
        self.assertGreater(score_sharp, 100.0)

        # ボケ画像
        blurry = cv2.GaussianBlur(sharp, (51, 51), 30.0)
        score_blurry = self.gpu_acc.calculate_blur_score_gpu(blurry)
        self.assertLess(score_blurry, score_sharp)

    def test_gpu_enhance_batch_fp16(self):
        """GPU FP16 並列バッチエンハンスのテスト"""
        img1 = np.full((120, 160, 3), 100, dtype=np.uint8)
        img2 = np.full((120, 160, 3), 200, dtype=np.uint8)
        batch = [img1, img2]

        enhanced = self.gpu_acc.enhance_batch_cuda_fp16(batch, sharpen_strength=0.8, denoise_strength=0.5)
        self.assertEqual(len(enhanced), 2)
        self.assertEqual(enhanced[0].shape, (120, 160, 3))
        self.assertEqual(enhanced[1].shape, (120, 160, 3))

    def test_hardware_optimizer_telemetry_and_autotune(self):
        """ハードウェアテレメトリ取得 & オートチューニングのテスト"""
        telemetry = self.optimizer.get_telemetry()
        self.assertIsNotNone(telemetry.gpu_name)
        self.assertGreater(telemetry.vram_total_mb, 0)
        self.assertGreaterEqual(telemetry.vram_usage_percent, 0.0)
        self.assertGreater(telemetry.cpu_cores_logical, 0)

        tuned_config = self.optimizer.auto_tune_configuration()
        self.assertIsNotNone(tuned_config)
        self.assertGreater(tuned_config.colmap_sift_gpu_memory_limit_mb, 4000)


if __name__ == "__main__":
    unittest.main()
