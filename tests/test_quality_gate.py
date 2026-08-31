"""
Unit Tests for Quality Gate Module
"""

import unittest
import numpy as np
import cv2
from solidforge.core.quality_gate import QualityGate, QualityEvaluationResult


class TestQualityGate(unittest.TestCase):
    def setUp(self):
        self.quality_gate = QualityGate()

    def test_calculate_blur_score_sharp(self):
        # シャープなチェッカーボード画像を生成
        sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        for y in range(0, 300, 30):
            for x in range(0, 300, 30):
                if (x // 30 + y // 30) % 2 == 0:
                    sharp_img[y:y+30, x:x+30] = 255

        score = self.quality_gate.calculate_blur_score(sharp_img)
        self.assertGreater(score, 500.0, f"Sharp image score should be high, got {score}")

    def test_calculate_blur_score_blurry(self):
        # ボケた画像を生成 (強いガウシアンブラー)
        sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        sharp_img[100:200, 100:200] = 255
        blurry_img = cv2.GaussianBlur(sharp_img, (51, 51), 30.0)

        score = self.quality_gate.calculate_blur_score(blurry_img)
        self.assertLess(score, 80.0, f"Blurry image score should be low, got {score}")

    def test_evaluate_image_criteria(self):
        # 豊かなテクスチャ画像
        np.random.seed(42)
        textured_img = (np.random.rand(400, 400, 3) * 255).astype(np.uint8)

        res = self.quality_gate.evaluate_image(textured_img, blur_threshold=100.0, min_feature_count=100)
        self.assertTrue(res.is_accepted)
        self.assertGreater(res.blur_score, 100.0)
        self.assertGreaterEqual(res.feature_count, 100)
        self.assertEqual(len(res.rejection_reasons), 0)

    def test_evaluate_image_rejects_blank(self):
        # 特徴のない無地画像
        blank_img = np.full((400, 400, 3), 128, dtype=np.uint8)

        res = self.quality_gate.evaluate_image(blank_img, blur_threshold=50.0, min_feature_count=100)
        self.assertFalse(res.is_accepted)
        self.assertGreater(len(res.rejection_reasons), 0)


if __name__ == "__main__":
    unittest.main()
