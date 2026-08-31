"""
Unit Tests for AI Enhancer Module
"""

import unittest
import numpy as np
import cv2
from solidforge.core.ai_enhancer import AIEnhancer
from solidforge.core.quality_gate import QualityGate


class TestAIEnhancer(unittest.TestCase):
    def setUp(self):
        self.enhancer = AIEnhancer()
        self.quality_gate = QualityGate()

    def test_ai_enhancement_sharpness_boost(self):
        # ややボケた画像を生成
        base_img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(base_img, (100, 100), 50, (255, 255, 255), -1)
        blurry = cv2.GaussianBlur(base_img, (15, 15), 3.0)

        initial_score = self.quality_gate.calculate_blur_score(blurry)

        # AIエンハンス適用
        enhanced = self.enhancer.enhance_image(blurry)
        enhanced_score = self.quality_gate.calculate_blur_score(enhanced)

        # エンハンス処理により鮮鋭度スコアが向上していることを確認
        self.assertGreater(enhanced_score, initial_score)
        self.assertEqual(enhanced.shape, blurry.shape)


if __name__ == "__main__":
    unittest.main()
