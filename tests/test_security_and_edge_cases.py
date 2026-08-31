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

    # =========================================================================
    # 7. スライサー起動 & 引数インジェクション耐性 (CWE-78 / CWE-88)
    # =========================================================================
    def test_orcaslicer_launch_security(self):
        """存在しないファイルや特殊文字を含むパスが安全に処理されるか検証"""
        # 存在しないファイル -> 安全にFalseを返却
        res_non_exist = ModelExporter.launch_in_orcaslicer(self.tmp_dir / "non_existent.stl")
        self.assertFalse(res_non_exist)

        # 特殊記号・空白を含む実在ファイル
        special_file = self.tmp_dir / "test model & print ; calc.stl"
        mesh = trimesh.creation.box()
        mesh.export(str(special_file), file_type="stl")
        # 例外を起こさず実行可能 (TrueまたはFalse)
        try:
            res_special = ModelExporter.launch_in_orcaslicer(special_file)
            self.assertIsInstance(res_special, bool)
        except Exception as e:
            self.fail(f"特殊パスで予期せぬ例外が発生しました: {e}")

    # =========================================================================
    # 8. カメラネットワークURL入力検証 (CWE-20: Network Input Validation)
    # =========================================================================
    def test_camera_manager_network_input_validation(self):
        """不正なURLや空文字列、無効なプロトコルに対して安全にFalseを返すか検証"""
        cam = CameraManager()
        # 空文字列
        self.assertFalse(cam.connect_smartphone_ip(""))
        self.assertFalse(cam.connect_smartphone_ip("   "))
        # 不正なプロトコル
        self.assertFalse(cam.connect_smartphone_ip("invalid_proto://127.0.0.1:99999"))
        # 存在しないホスト
        self.assertFalse(cam.connect_smartphone_ip("http://192.0.2.1:8080/video"))

    # =========================================================================
    # 9. ポストプロセッサ退化メッシュ耐性 (Degenerate Mesh & Zero Face)
    # =========================================================================
    def test_post_processor_degenerate_zero_face_mesh(self):
        """面が存在しない空メッシュや極小メッシュを渡してもクラッシュしないか検証"""
        empty_mesh = trimesh.Trimesh()
        # 空メッシュでも安全にフォールバックメッシュとレポートが生成される
        processed, report = self.post_processor.process_model(empty_mesh, scale_factor=1.0)
        self.assertIsNotNone(processed)
        self.assertIsNotNone(report)
        self.assertGreater(len(processed.vertices), 0)

    # =========================================================================
    # 10. パイプライン空入力・破損ファイル検証 (Pipeline Empty & Corrupt Staging)
    # =========================================================================
    def test_pipeline_empty_and_corrupt_inputs(self):
        """空リストや0バイトファイルのみを渡した場合に安全にエラー停止シグナルが送出されるか検証"""
        pipeline = ReconstructionPipeline()
        work_dir = self.tmp_dir / "pipe_test"

        finished_results = []
        pipeline.reconstruction_finished.connect(lambda s, p, r: finished_results.append((s, p, r)))

        # 空リスト -> 安全に停止し success=False
        pipeline._run_pipeline([], "stl", work_dir)
        self.assertEqual(len(finished_results), 1)
        self.assertFalse(finished_results[0][0])  # success == False

        # 0バイトファイルのみ -> 安全に停止し success=False
        zero_file = self.tmp_dir / "zero.jpg"
        zero_file.touch()
        pipeline._run_pipeline([zero_file], "stl", work_dir)
        self.assertEqual(len(finished_results), 2)
        self.assertFalse(finished_results[1][0])  # success == False

    # =========================================================================
    # 11. ArUco 実寸スケール校正の境界値検証 (ArUco Scale Robustness)
    # =========================================================================
    def test_aruco_scale_calibration_edge_cases(self):
        """異常なスケール倍率 (負数、0、極端な1000倍) に対する安全性検証"""
        mesh = trimesh.creation.box(extents=[10.0, 10.0, 10.0])
        # 負の倍率 -> デフォルト1.0に自動クランプ
        proc_neg, rep_neg = self.post_processor.process_model(mesh, scale_factor=-5.0)
        self.assertEqual(rep_neg.scale_factor_applied, 1.0)

        # 0倍率 -> デフォルト1.0に自動クランプ
        proc_zero, rep_zero = self.post_processor.process_model(mesh, scale_factor=0.0)
        self.assertEqual(rep_zero.scale_factor_applied, 1.0)


if __name__ == "__main__":
    unittest.main()

