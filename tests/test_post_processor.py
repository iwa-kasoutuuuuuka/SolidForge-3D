"""
Unit Tests for 3D Print Post Processor & Watertight Repair
"""

import unittest
import trimesh
from solidforge.core.post_processor import MeshPostProcessor


class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.post_processor = MeshPostProcessor()

    def test_watertight_synthetic_mesh(self):
        mesh = self.post_processor.generate_synthetic_photogrammetry_mesh()
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(len(mesh.faces), 0)

    def test_process_model_diagnostics(self):
        mesh = self.post_processor.generate_synthetic_photogrammetry_mesh()
        processed_mesh, report = self.post_processor.process_model(mesh, scale_factor=2.0)

        self.assertTrue(report.is_watertight)
        self.assertGreater(report.volume_cm3, 0.0)
        self.assertGreater(report.dimensions_mm[0], 0.0)
        self.assertGreater(report.estimated_weight_pla_g, 0.0)
        self.assertEqual(report.scale_factor_applied, 2.0)


if __name__ == "__main__":
    unittest.main()
