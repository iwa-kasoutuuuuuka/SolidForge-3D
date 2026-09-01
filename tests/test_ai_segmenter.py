# -*- coding: utf-8 -*-
"""
Unit tests for AISegmenter (solidforge.core.ai_segmenter)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

from solidforge.core.ai_segmenter import AISegmenter, AI_SEGMENTER


class TestAISegmenter(unittest.TestCase):
    """Test suite for AI Object Segmentation and Mask Generation"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_img_path = self.temp_dir / "test_sample.jpg"

        # Create a synthetic image: black background with a bright red circle in center
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, (0, 0, 255), -1)
        cv2.imwrite(str(self.sample_img_path), img)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_segmenter_initialization(self):
        """Test AISegmenter instance creation"""
        segmenter = AISegmenter(model_name="u2net")
        self.assertEqual(segmenter.model_name, "u2net")

    def test_single_mask_generation(self):
        """Test generating mask for a single image"""
        out_mask_path = self.temp_dir / "mask.png"
        success = AI_SEGMENTER.generate_mask(
            self.sample_img_path,
            out_mask_path,
            margin_pixels=3,
        )
        self.assertTrue(success)
        self.assertTrue(out_mask_path.exists())

        mask = cv2.imread(str(out_mask_path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (200, 200))
        # Mask should have binary values (0 or 255)
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_batch_mask_generation(self):
        """Test batch mask generation producing COLMAP and OpenMVS masks"""
        masks_dir = self.temp_dir / "masks"
        img2_path = self.temp_dir / "test_sample2.jpg"
        shutil.copyfile(self.sample_img_path, img2_path)

        masks = AI_SEGMENTER.process_batch(
            [self.sample_img_path, img2_path],
            masks_dir,
            margin_pixels=2,
        )
        self.assertEqual(len(masks), 2)
        # Verify COLMAP mask exists
        self.assertTrue((masks_dir / "test_sample.jpg.png").exists())
        # Verify OpenMVS mask exists
        self.assertTrue((masks_dir / "test_sample.mask.png").exists())

    def test_fallback_grabcut_mask(self):
        """Test fallback GrabCut segmentation method"""
        segmenter = AISegmenter()
        mask = segmenter._fallback_grabcut_mask(self.sample_img_path)
        self.assertEqual(mask.shape, (200, 200))
        self.assertTrue(isinstance(mask, np.ndarray))


if __name__ == "__main__":
    unittest.main()
