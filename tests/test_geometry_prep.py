"""
Unit Tests for Direct-to-Print Geometry Prep Module
"""

import unittest
import numpy as np
import trimesh
from solidforge.core.geometry_prep import GeometryPrep


class TestGeometryPrep(unittest.TestCase):
    def setUp(self):
        self.prep = GeometryPrep()

    def test_detect_ground_plane_and_slice(self):
        # 台座付きのテスト用メッシュを作成 (円柱 + 球)
        base = trimesh.creation.cylinder(radius=20.0, height=10.0)
        base.apply_translation([0, 0, 5.0])
        sphere = trimesh.creation.icosphere(radius=15.0)
        sphere.apply_translation([0, 0, 25.0])
        mesh = trimesh.util.concatenate([base, sphere])

        # スライス前: 最底面は 0.0 付近
        self.assertAlmostEqual(float(np.min(mesh.vertices[:, 2])), 0.0, delta=0.5)

        # 底面フラットカットを実行 (オフセット 2.0mm)
        sliced = self.prep.slice_ground_plane(mesh, offset_mm=2.0)
        self.assertIsNotNone(sliced)
        self.assertTrue(len(sliced.faces) > 0)
        # スライス後: 最底面が Z=0 に正確にアライメントされていることを確認
        self.assertAlmostEqual(float(np.min(sliced.vertices[:, 2])), 0.0, delta=1e-3)

    def test_analyze_overhangs(self):
        # 直方体メッシュ (上面, 側面, 底面)
        box = trimesh.creation.box(extents=[10, 10, 10])
        box.apply_translation([0, 0, 5])
        ratio, mask = self.prep.analyze_overhangs(box, threshold_angle_deg=45.0)
        self.assertGreater(ratio, 0.0)
        self.assertTrue(np.any(mask))

    def test_add_base_pedestal(self):
        sphere = trimesh.creation.icosphere(radius=10.0)
        sphere.apply_translation([0, 0, 10.0])
        with_ped = self.prep.add_base_pedestal(sphere, thickness_mm=2.5, margin_mm=5.0)
        self.assertIsNotNone(with_ped)
        self.assertGreater(len(with_ped.faces), len(sphere.faces))
        self.assertAlmostEqual(float(np.min(with_ped.vertices[:, 2])), 0.0, delta=1e-2)

    def test_decimate_mesh(self):
        sphere = trimesh.creation.icosphere(radius=10.0, subdivisions=3)
        orig_faces = len(sphere.faces)
        decimated = self.prep.decimate_mesh(sphere, target_reduction=0.5)
        self.assertIsNotNone(decimated)
        self.assertLess(len(decimated.faces), orig_faces)


if __name__ == "__main__":
    unittest.main()
