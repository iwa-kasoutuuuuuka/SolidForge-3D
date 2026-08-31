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

    def test_hollow_mesh(self):
        sphere = trimesh.creation.icosphere(radius=20.0, subdivisions=2)
        initial_faces = len(sphere.faces)

        hollowed = self.prep.hollow_mesh(sphere, wall_thickness_mm=2.0)
        # 中空化により内壁面が追加され、面数が増加していることを確認
        self.assertEqual(len(hollowed.faces), initial_faces * 2)


if __name__ == "__main__":
    unittest.main()
