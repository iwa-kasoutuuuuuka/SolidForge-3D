"""
Unit Tests for Multi-GPU Support (Multi-GPU CUDA Cluster & Dynamic VRAM Allocation)
"""

import unittest
import numpy as np
from solidforge.config import MultiGPUConfig
from solidforge.core.gpu_accelerator import GPUAccelerator
from solidforge.core.hardware_optimizer import HardwareOptimizer


class TestMultiGPU(unittest.TestCase):
    def setUp(self):
        self.optimizer = HardwareOptimizer()
        self.gpu_acc = GPUAccelerator()

    def test_multi_gpu_config_colmap_string(self):
        """COLMAP用GPUインデックス文字列生成のテスト"""
        cfg = MultiGPUConfig(mode="ALL_GPUS", active_device_indices=[0, 1])
        self.assertEqual(cfg.get_colmap_gpu_index_str(), "0,1")

        cfg_single = MultiGPUConfig(mode="PRIMARY_ONLY", active_device_indices=[0, 1])
        self.assertEqual(cfg_single.get_colmap_gpu_index_str(), "0")

        cfg_triple = MultiGPUConfig(mode="ALL_GPUS", active_device_indices=[0, 1, 2])
        self.assertEqual(cfg_triple.get_colmap_gpu_index_str(), "0,1,2")

    def test_hardware_optimizer_list_all_gpus(self):
        """全GPUの列挙とテレメトリ取得テスト"""
        gpus = self.optimizer.list_all_gpus()
        self.assertIsInstance(gpus, list)
        self.assertGreaterEqual(len(gpus), 1)
        self.assertIsNotNone(gpus[0].gpu_name)
        self.assertGreater(gpus[0].vram_total_mb, 0)

    def test_multi_gpu_batch_enhancement_execution(self):
        """マルチGPU並列バッチエンハンスの実行テスト"""
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 150, dtype=np.uint8)
        img3 = np.full((100, 100, 3), 200, dtype=np.uint8)
        img4 = np.full((100, 100, 3), 250, dtype=np.uint8)
        batch = [img1, img2, img3, img4]

        enhanced = self.gpu_acc.enhance_batch_cuda_fp16(batch)
        self.assertEqual(len(enhanced), 4)
        for out in enhanced:
            self.assertEqual(out.shape, (100, 100, 3))


if __name__ == "__main__":
    unittest.main()
