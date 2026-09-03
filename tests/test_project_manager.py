# -*- coding: utf-8 -*-
"""
SolidForge 3D - ProjectManager Unit Tests
"""

import unittest
from pathlib import Path
import shutil
import numpy as np
import cv2
import trimesh

from solidforge.core.project_manager import PROJECT_MANAGER
from solidforge.core.post_processor import PrintabilityReport


class TestProjectManager(unittest.TestCase):
    """プロジェクト保存 & 復元エンジンの単体テスト"""

    def setUp(self):
        self.test_dir = Path("workspace/test_project_manager")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # テスト画像
        self.img1 = self.test_dir / "test_img1.jpg"
        cv2.imwrite(str(self.img1), np.full((100, 100, 3), 128, dtype=np.uint8))

        # テストモデル
        self.stl = self.test_dir / "test_model.stl"
        box = trimesh.creation.box(extents=[10, 10, 10])
        box.export(str(self.stl))

        # 診断レポート
        self.report = PrintabilityReport(
            is_watertight=True,
            is_winding_consistent=True,
            vertex_count=8,
            face_count=12,
            dimensions_mm=(10.0, 10.0, 10.0),
            volume_cm3=1.0,
            surface_area_cm2=6.0,
            estimated_weight_pla_g=1.24,
            scale_factor_applied=1.0,
            overhang_area_ratio_pct=16.6,
        )

        self.sforge_path = self.test_dir / "my_project.sforge"
        self.extract_dir = self.test_dir / "extracted"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_and_import_project(self):
        # 1. Export
        success = PROJECT_MANAGER.export_project(
            dest_file_path=self.sforge_path,
            project_name="Test Project",
            image_paths=[self.img1],
            model_path=self.stl,
            report=self.report,
        )
        self.assertTrue(success)
        self.assertTrue(self.sforge_path.exists())
        self.assertGreater(self.sforge_path.stat().st_size, 0)

        # 2. Import
        ok, meta, images, model = PROJECT_MANAGER.import_project(
            sforge_path=self.sforge_path,
            extract_root_dir=self.extract_dir,
        )
        self.assertTrue(ok)
        self.assertEqual(meta.get("project_name"), "Test Project")
        self.assertEqual(len(images), 1)
        self.assertIsNotNone(model)
        self.assertTrue(model.exists())
        self.assertEqual(meta["report"]["is_watertight"], True)


if __name__ == "__main__":
    unittest.main()
