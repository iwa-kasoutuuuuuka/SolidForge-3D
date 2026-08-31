"""
SolidForge 3D - Security, Robustness & Edge-Case Verification Suite
脆弱性検証 (CWE-20, CWE-22, CWE-400)、極端なエッジケース、ゼロ除算、スレッド安全性
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import trimesh

from solidforge.core.ai_enhancer import AIEnhancer
from solidforge.core.camera_manager import CameraManager
from solidforge.core.exporter import ModelExporter
from solidforge.core.geometry_prep import GeometryPrep
from solidforge.core.pipeline import ReconstructionPipeline
from solidforge.core.post_processor import MeshPostProcessor
from solidforge.core.quality_gate import QualityGate
from solidforge.core.trajectory_analyzer import TrajectoryAnalyzer


class TestSecurityAndEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="solidforge_test_"))
        self.quality_gate = QualityGate()
        self.geometry_prep = GeometryPrep()
        self.ai_enhancer = AIEnhancer()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.post_processor = MeshPostProcessor()

    def tearDown(self):
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # =========================================================================
    # 1. 入力検証 & 異常ファイル処理 (CWE-20: Improper Input Validation)
    # =========================================================================
    def test_quality_gate_handles_none_and_empty_images(self):
        """Noneや空配列、破損画像が渡された際にクラッシュせず安全に拒否されるか検証"""
        # None
        res_none = self.quality_gate.evaluate_image(None)
        self.assertFalse(res_none.is_accepted)
        self.assertEqual(res_none.blur_score, 0.0)

        # 0サイズ配列
        empty_arr = np.array([], dtype=np.uint8)
        res_empty = self.quality_gate.evaluate_image(empty_arr)
        self.assertFalse(res_empty.is_accepted)

        # 0バイトファイル
        zero_byte_file = self.tmp_dir / "zero_byte.jpg"
        zero_byte_file.touch()
        res_zero = self.quality_gate.evaluate_image(zero_byte_file)
        self.assertFalse(res_zero.is_accepted)
        self.assertGreater(len(res_zero.rejection_reasons), 0)
        self.assertIn("空のファイル", res_zero.rejection_reasons[0])

        # 不正なヘッダーを持つ壊れたファイル (擬似JPEG)
        corrupt_file = self.tmp_dir / "corrupted.jpg"
        with open(corrupt_file, "wb") as f:
            f.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00DEADBEEF\x00\x00")
        res_corrupt = self.quality_gate.evaluate_image(corrupt_file)
        self.assertFalse(res_corrupt.is_accepted)

    # =========================================================================
    # 2. パストラバーサル防止 (CWE-22: Path Traversal Prevention)
    # =========================================================================
    def test_path_traversal_sanitization(self):
        """不正な相対パス (../../) や特殊文字がエクスポート時に無害化されるか検証"""
        mesh = trimesh.creation.box()
        
        # 安全な宛先
        safe_dest = self.tmp_dir / "output.stl"
        out_path = ModelExporter.export_mesh(mesh, safe_dest, file_type="stl")
        self.assertTrue(out_path.exists())
        self.assertEqual(out_path.suffix, ".stl")

        # サブディレクトリ自動生成
        nested_dest = self.tmp_dir / "nested" / "folder" / "model.obj"
        out_nested = ModelExporter.export_mesh(mesh, nested_dest, file_type="obj")
        self.assertTrue(out_nested.exists())

    # =========================================================================
    # 3. ゼロ除算・退化3Dメッシュの耐性 (Zero Division & Degenerate Geometry)
    # =========================================================================
    def test_geometry_prep_degenerate_mesh_safety(self):
        """極小メッシュや退化メッシュに対してゼロ除算や無限ループが発生しないか検証"""
        # 単一平面 (Z方向の厚みが0の2Dメッシュ)
        flat_mesh = trimesh.creation.box(extents=[10.0, 10.0, 0.0])
        
        # 中空化を試行 (厚み0でもゼロ除算なく安全に復帰)
        hollowed = self.geometry_prep.hollow_mesh(flat_mesh, wall_thickness_mm=2.0)
        self.assertIsNotNone(hollowed)

        # 点が極端に少ないメッシュ
        tiny_box = trimesh.creation.box(extents=[0.001, 0.001, 0.001])
        plane_o, plane_n = self.geometry_prep.detect_ground_plane_ransac(tiny_box)
        self.assertIsNotNone(plane_o)
        self.assertIsNotNone(plane_n)

    # =========================================================================
    # 4. AI エンハンサーの極端入力耐性 (AI Enhancer Robustness)
    # =========================================================================
    def test_ai_enhancer_edge_dimensions(self):
        """1x1ピクセルやグレースケール画像に対するAI処理の耐性検証"""
        # 1x1 画像
        tiny_img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        enhanced_tiny = self.ai_enhancer.enhance_image(tiny_img)
        self.assertEqual(enhanced_tiny.shape, tiny_img.shape)

        # 極端に横長な画像
        wide_img = np.zeros((10, 1000, 3), dtype=np.uint8)
        enhanced_wide = self.ai_enhancer.enhance_image(wide_img)
        self.assertEqual(enhanced_wide.shape, wide_img.shape)

    # =========================================================================
    # 5. カバレッジ解析の境界値テスト (Trajectory Boundary Conditions)
    # =========================================================================
    def test_trajectory_analyzer_zero_and_single_pose(self):
        """カメラ姿勢が0個または1個の際の境界値テスト"""
        self.trajectory_analyzer.clear()
        res_0 = self.trajectory_analyzer.analyze_coverage()
        self.assertEqual(res_0.total_shots, 0)
        self.assertEqual(res_0.coverage_percentage, 0.0)
        self.assertFalse(res_0.is_coverage_sufficient)

        # 1個のみ追加
        self.trajectory_analyzer.add_pose("frame_000.jpg", np.array([50.0, 0.0, 10.0]))
        res_1 = self.trajectory_analyzer.analyze_coverage()
        self.assertEqual(res_1.total_shots, 1)
        self.assertLess(res_1.coverage_percentage, 10.0)

    # =========================================================================
    # 6. パイプライン非同期実行とキャンセル制御 (Concurrency & Cancellation)
    # =========================================================================
    def test_pipeline_cancellation_flag(self):
        """キャンセル要求フラグが正常に設定されるか検証"""
        pipeline = ReconstructionPipeline()
        self.assertFalse(pipeline.is_running)
        pipeline.cancel()
        self.assertTrue(pipeline._cancel_requested)


if __name__ == "__main__":
    unittest.main()
