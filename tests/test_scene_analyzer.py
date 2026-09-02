# -*- coding: utf-8 -*-
"""
SolidForge 3D - SceneAnalyzer Unit Tests
"""

import unittest
from pathlib import Path
import numpy as np
import cv2

from solidforge.core.scene_analyzer import SCENE_ANALYZER, SceneAnalysisResult


class TestSceneAnalyzer(unittest.TestCase):
    """撮影画像認識 & 最適設定自動導出エンジンの単体テスト"""

    def setUp(self):
        self.test_dir = Path("workspace/test_scene_analyzer")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. テスト画像生成 (小型被写体・中央丸)
        self.img1 = self.test_dir / "sample_small.jpg"
        im1 = np.full((400, 400, 3), 200, dtype=np.uint8)
        cv2.circle(im1, (200, 200), 40, (30, 30, 180), -1)  # 小さな円
        cv2.imwrite(str(self.img1), im1)

        # 2. テスト画像生成 (大型被写体・全体模様)
        self.img2 = self.test_dir / "sample_large.jpg"
        im2 = np.random.randint(50, 220, (400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(self.img2), im2)

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_empty_dataset(self):
        res = SCENE_ANALYZER.analyze_dataset([])
        self.assertIsInstance(res, SceneAnalysisResult)
        self.assertEqual(res.sample_count, 0)
        self.assertEqual(res.rec_precision_mode, "ULTRA_HIGH")

    def test_analyze_small_object(self):
        res = SCENE_ANALYZER.analyze_dataset([self.img1])
        self.assertIsInstance(res, SceneAnalysisResult)
        self.assertEqual(res.sample_count, 1)
        self.assertTrue(res.rec_enable_bg_removal)
        self.assertIn(res.rec_precision_mode, ["ULTRA_HIGH", "EXTREME_CAD"])
        self.assertTrue(res.rec_enable_guided_matching)

    def test_analyze_recommendation_fields(self):
        res = SCENE_ANALYZER.analyze_dataset([self.img1, self.img2])
        self.assertGreater(res.sample_count, 0)
        self.assertGreater(res.rec_sift_max_features, 4000)
        self.assertLessEqual(res.rec_sift_peak_threshold, 0.006)
        self.assertIsNotNone(res.summary_text_ja)


if __name__ == "__main__":
    unittest.main()
