# -*- coding: utf-8 -*-
"""
SolidForge 3D - Precision & Ultra-Reconstruction Unit Tests
"""

import unittest
from solidforge.config import CONFIG, PrecisionConfig, AppConfig


class TestPrecisionConfig(unittest.TestCase):
    """3Dスキャン高精度化・極限復元設定の単体テスト"""

    def test_precision_defaults(self):
        prec = PrecisionConfig()
        self.assertEqual(prec.mode, "ULTRA_HIGH")
        self.assertEqual(prec.sift_peak_threshold, 0.002)
        self.assertEqual(prec.sift_max_features, 16384)
        self.assertTrue(prec.enable_guided_matching)
        self.assertTrue(prec.enable_refine_mesh)
        self.assertEqual(prec.dense_resolution_level, 1)

    def test_precision_modes(self):
        # Balanced
        prec_balanced = PrecisionConfig(mode="BALANCED", sift_peak_threshold=0.006, sift_max_features=8192)
        self.assertEqual(prec_balanced.mode, "BALANCED")
        self.assertEqual(prec_balanced.sift_peak_threshold, 0.006)

        # Extreme CAD
        prec_cad = PrecisionConfig(mode="EXTREME_CAD", sift_peak_threshold=0.001, sift_max_features=32768, dense_resolution_level=0)
        self.assertEqual(prec_cad.mode, "EXTREME_CAD")
        self.assertEqual(prec_cad.sift_peak_threshold, 0.001)
        self.assertEqual(prec_cad.dense_resolution_level, 0)

    def test_app_config_integration(self):
        cfg = AppConfig()
        self.assertIsNotNone(cfg.precision)
        self.assertEqual(cfg.precision.mode, "ULTRA_HIGH")
        self.assertTrue(cfg.precision.enable_refine_mesh)


if __name__ == "__main__":
    unittest.main()
