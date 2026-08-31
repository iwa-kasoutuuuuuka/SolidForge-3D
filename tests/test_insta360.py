"""
Unit Tests for Insta360 Adapter (X5 / Ace Pro 2 / Link 2 / 360 Dewarping)
"""

import unittest
import numpy as np
from solidforge.core.camera_manager import CameraManager
from solidforge.core.insta360_adapter import Insta360Adapter


class TestInsta360Adapter(unittest.TestCase):
    def setUp(self):
        self.adapter = Insta360Adapter()
        self.cam = CameraManager()

    def test_is_360_panorama_detection(self):
        """アスペクト比2:1のパノラマ検出テスト"""
        pano_img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        self.assertTrue(self.adapter.is_360_panorama(pano_img))

        standard_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertFalse(self.adapter.is_360_panorama(standard_img))

    def test_equirectangular_to_perspective(self):
        """360パノラマからの透視投影展開テスト"""
        # 擬似パノラマ画像生成
        pano = np.full((500, 1000, 3), 128, dtype=np.uint8)
        # 正面に白マーカー
        pano[200:300, 450:550] = 255

        # 正面 (Yaw=0) 展開
        front_view = self.adapter.equirectangular_to_perspective(
            pano, yaw_deg=0.0, pitch_deg=0.0, fov_deg=90.0, out_width=640, out_height=360
        )
        self.assertEqual(front_view.shape, (360, 640, 3))
        # 正面マーカーが含まれるため白色ピクセルが存在
        self.assertTrue(np.any(front_view == 255))

        # 右側 (Yaw=90) 展開
        right_view = self.adapter.equirectangular_to_perspective(
            pano, yaw_deg=90.0, pitch_deg=0.0, fov_deg=90.0, out_width=640, out_height=360
        )
        self.assertEqual(right_view.shape, (360, 640, 3))

    def test_generate_virtual_multiviews(self):
        """1枚の360パノラマから4方向・6方向のマルチビュー自動展開テスト"""
        pano = np.zeros((400, 800, 3), dtype=np.uint8)
        views_4 = self.adapter.generate_virtual_multiviews(pano, num_views=4)
        self.assertEqual(len(views_4), 4)
        labels = [v[0] for v in views_4]
        self.assertIn("front", labels)
        self.assertIn("right", labels)
        self.assertIn("back", labels)
        self.assertIn("left", labels)

        views_6 = self.adapter.generate_virtual_multiviews(pano, num_views=6)
        self.assertEqual(len(views_6), 6)

    def test_connect_insta360(self):
        """Insta360 接続ライフサイクルのテスト"""
        res = self.cam.connect_insta360(model_name="Ace Pro 2")
        self.assertTrue(res)
        self.assertTrue(self.cam.is_connected)
        self.cam.disconnect()
        self.assertFalse(self.cam.is_connected)


if __name__ == "__main__":
    unittest.main()
