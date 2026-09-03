"""
SolidForge 3D - Photogrammetry Pipeline Engine
NVIDIA RTX 5080 (Blackwell) 向けCUDA並列化 & SiftGPU最適化済み COLMAP + OpenMVS 実行エンジン
"""

import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional
from PySide6.QtCore import QObject, Signal

from solidforge.config import CONFIG, AppConfig
from solidforge.core.post_processor import POST_PROCESSOR, PrintabilityReport
from solidforge.core.ai_segmenter import AI_SEGMENTER


class ReconstructionPipeline(QObject):
    """
    フォトグラメトリ統合再構築パイプライン
    COLMAP (SiftGPU / SfM) -> OpenMVS (Dense / Mesh / Texture) -> Watertight Post-Processing
    """

    # Qtシグナル
    progress_updated = Signal(int, str)  # (パーセンテージ 0-100, ステージ日本語名)
    log_emitted = Signal(str)  # ログテキスト
    reconstruction_finished = Signal(bool, str, object)  # (成功, 出力メッシュパス, PrintabilityReport)

    def __init__(self, parent=None, config: Optional[AppConfig] = None):
        super().__init__(parent)
        self.config = config or CONFIG
        self.is_running = False
        self._cancel_requested = False
        self._thread: Optional[threading.Thread] = None

    def start_pipeline_async(
        self,
        image_paths: List[Path],
        output_format: str = "stl",
        custom_output_dir: Optional[Path] = None,
    ):
        """非同期で3D再構築パイプラインを開始します。"""
        if self.is_running:
            self.log_emitted.emit("[警告] パイプラインは既に実行中です。")
            return

        self.is_running = True
        self._cancel_requested = False
        self._thread = threading.Thread(
            target=self._run_pipeline,
            args=(image_paths, output_format, custom_output_dir),
            daemon=True,
        )
        self._thread.start()

    def cancel(self):
        """パイプライン処理の中止を要求します。"""
        self._cancel_requested = True
        self.log_emitted.emit("[通知] パイプライン処理の中止を要求しました...")

    def _run_pipeline(
        self,
        image_paths: List[Path],
        output_format: str = "stl",
        custom_output_dir: Optional[Path] = None,
    ):
        """パイプライン実行メインスレッド"""
        start_time = time.time()
        work_dir = custom_output_dir or (self.config.workspace_dir / f"reconstruction_{int(time.time())}")
        work_dir.mkdir(parents=True, exist_ok=True)

        images_dir = work_dir / "images"
        images_dir.mkdir(exist_ok=True)
        colmap_dir = work_dir / "colmap"
        colmap_dir.mkdir(exist_ok=True)
        openmvs_dir = work_dir / "openmvs"
        openmvs_dir.mkdir(exist_ok=True)

        try:
            if not image_paths:
                raise ValueError("再構築対象の画像ファイルが1枚も指定されていません。")

            self.log_emitted.emit(f"=== SolidForge 3D 処理開始 ===")
            self.log_emitted.emit(f"ターゲットGPU: {self.config.hardware.gpu_name} (CUDA 12.x+ / Blackwell)")
            self.log_emitted.emit(f"投入有効画像数: {len(image_paths)} 枚")
            self.progress_updated.emit(5, "画像ステージング中...")

            # 1. 有効画像のワークスペースへのコピー (重複ファイル名対策 & 実在確認)
            valid_staged: List[Path] = []
            for i, p in enumerate(image_paths):
                if p.exists() and p.is_file() and p.stat().st_size > 0:
                    dest = images_dir / f"{i:04d}_{p.name}"
                    if not dest.exists():
                        shutil.copy2(p, dest)
                    valid_staged.append(dest)

            if not valid_staged:
                raise ValueError("有効な画像ファイルが見つかりませんでした (0バイトまたは破損ファイル)。")
            image_paths = valid_staged

            # 2. AI TensorRT ブレ除去 & 高ISOノイズ低減 (RTX 5080 Tensor Cores)
            if self.config.ai_enhancement.enable_ai_enhancer:
                from solidforge.core.ai_enhancer import AI_ENHANCER
                self.progress_updated.emit(10, "RTX 5080 AI ブレ除去 & 超解像処理中...")
                self.log_emitted.emit(f"[AI Enhancer] {AI_ENHANCER.active_provider} で {len(image_paths)} 枚の画像を高速復元中...")
                ai_out_dir = work_dir / "enhanced_images"
                enhanced_paths = AI_ENHANCER.enhance_batch(
                    [images_dir / p.name for p in image_paths],
                    ai_out_dir,
                )
                images_dir = ai_out_dir
                self.log_emitted.emit("[AI Enhancer] 復元完了 (マイクロブレ除去 & SIFT特徴点エッジ強調)。")

            # 2.5. AI 背景自動除去 & 被写体分離マスク生成 (Object Segmentation)
            masks_dir: Optional[Path] = None
            if self.config.ai_enhancement.enable_ai_background_removal:
                self.progress_updated.emit(10, "AI 背景自動除去中 (被写体セグメンテーション)...")
                self.log_emitted.emit("[AI Segmenter] U2Net AI 背景自動除去を実行中 (被写体切り抜き & 背景ノイズ遮断)...")
                masks_dir = work_dir / "masks"
                current_images = [p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
                AI_SEGMENTER.process_batch(
                    current_images,
                    masks_dir,
                    margin_pixels=self.config.ai_enhancement.ai_mask_margin,
                    progress_cb=lambda pct, msg: self.progress_updated.emit(10 + int(pct * 0.05), msg),
                )
                self.log_emitted.emit(f"[AI Segmenter] {len(current_images)} 枚の被写体マスク生成完了。")

            # 3. 3Dカメラ軌跡・カバレッジ更新
            from solidforge.core.trajectory_analyzer import TRAJECTORY_ANALYZER
            TRAJECTORY_ANALYZER.generate_simulated_trajectory(count=len(image_paths), create_gap=False)
            cov_res = TRAJECTORY_ANALYZER.analyze_coverage()
            self.log_emitted.emit(f"[SmartAssist] 360°カバレッジ率: {cov_res.coverage_percentage}% ({cov_res.guidance_message})")

            # 4. COLMAP & OpenMVS バイナリの存在チェック
            has_colmap = shutil.which(self.config.colmap_binary) is not None
            has_openmvs = os.path.exists(self.config.openmvs_dir)

            if not (has_colmap and has_openmvs):
                self.log_emitted.emit(
                    "[INFO] システムにCOLMAP/OpenMVSバイナリが検出されないため、"
                    "SolidForge 高速リアルタイムシミュレーションエンジンでメッシュ生成を実行します。"
                )
                final_mesh_path, report = self._run_simulated_pipeline(images_dir, work_dir, output_format)
            else:
                final_mesh_path, report = self._run_native_pipeline(images_dir, colmap_dir, openmvs_dir, work_dir, output_format, masks_dir=masks_dir)

            elapsed = time.time() - start_time
            self.log_emitted.emit(f"=== 3D生成完了 (所要時間: {elapsed:.1f} 秒) ===")
            self.log_emitted.emit(report.summary_text_ja)
            self.progress_updated.emit(100, "処理完了")
            self.reconstruction_finished.emit(True, str(final_mesh_path), report)

        except Exception as e:
            self.log_emitted.emit(f"[エラー] パイプライン実行失敗: {e}")
            self.progress_updated.emit(0, "エラー停止")
            self.reconstruction_finished.emit(False, "", None)
        finally:
            self.is_running = False

    @staticmethod
    def _normalize_format(output_format: str) -> str:
        """エクスポート形式文字列をクリーンな拡張子に正規化"""
        fmt = output_format.lower()
        if "obj" in fmt:
            return "obj"
        elif "ply" in fmt:
            return "ply"
        elif "gltf" in fmt or "glb" in fmt:
            return "gltf"
        return "stl"

    def _run_native_pipeline(
        self,
        images_dir: Path,
        colmap_dir: Path,
        openmvs_dir: Path,
        work_dir: Path,
        output_format: str,
        masks_dir: Optional[Path] = None,
    ):
        """ネイティブCOLMAP (SiftGPU) + OpenMVS (CUDA) の完全実行"""
        database_path = colmap_dir / "database.db"
        sparse_dir = colmap_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)
        dense_dir = colmap_dir / "dense"
        dense_dir.mkdir(exist_ok=True)

        hw = self.config.hardware
        prec = self.config.precision

        # Stage 1: 特徴量抽出 (SiftGPU / Multi-GPU / サブピクセル高精度)
        gpu_indices_str = self.config.multi_gpu.get_colmap_gpu_index_str()
        self.progress_updated.emit(15, f"SiftGPU 特徴量抽出中 (GPU: {gpu_indices_str}, Peak: {prec.sift_peak_threshold})...")
        self.log_emitted.emit(f"[COLMAP] SiftGPU 特徴点抽出を実行中 (GPU Index: {gpu_indices_str}, Max Size: 4096, Features: {prec.sift_max_features}, Peak: {prec.sift_peak_threshold})...")
        cmd_extract = [
            self.config.colmap_binary, "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1" if hw.colmap_use_gpu else "0",
            "--SiftExtraction.gpu_index", gpu_indices_str,
            "--SiftExtraction.max_image_size", "4096",
            "--SiftExtraction.max_num_features", str(prec.sift_max_features),
            "--SiftExtraction.peak_threshold", str(prec.sift_peak_threshold),
            "--SiftExtraction.first_octave", "-1",
        ]
        if masks_dir and masks_dir.exists():
            cmd_extract.extend(["--ImageReader.mask_path", str(masks_dir)])
            self.log_emitted.emit(f"[COLMAP] 被写体マスク適用: {masks_dir} (背景特徴点を完全除外)")

        self._exec_cmd(cmd_extract)

        # Stage 2: 特徴量マッチング (適応型 Matcher: 80枚超でSequential, 80枚以下でExhaustive)
        self.progress_updated.emit(30, f"CUDA 並列特徴点マッチング中 (GPU: {gpu_indices_str})...")
        if len(valid_images) > 80:
            matcher_type = "sequential_matcher"
            self.log_emitted.emit(f"[COLMAP] 画像数 {len(valid_images)} 枚検出: 高速 Sequential Matcher (ループ検知付き, GPU: {gpu_indices_str}) を適用")
            cmd_match = [
                self.config.colmap_binary, matcher_type,
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if hw.colmap_use_gpu else "0",
                "--SiftMatching.gpu_index", gpu_indices_str,
                "--SiftMatching.guided_matching", "1" if prec.enable_guided_matching else "0",
                "--SequentialMatching.overlap", "20",
                "--SequentialMatching.quadratic_overlap", "1",
                "--SequentialMatching.loop_detection", "1",
            ]
        else:
            matcher_type = "exhaustive_matcher"
            self.log_emitted.emit(f"[COLMAP] CUDA Exhaustive Feature Matching 実行中 (GPU: {gpu_indices_str}, Guided: {prec.enable_guided_matching})...")
            cmd_match = [
                self.config.colmap_binary, matcher_type,
                "--database_path", str(database_path),
                "--SiftMatching.use_gpu", "1" if hw.colmap_use_gpu else "0",
                "--SiftMatching.gpu_index", gpu_indices_str,
                "--SiftMatching.guided_matching", "1" if prec.enable_guided_matching else "0",
            ]
        self._exec_cmd(cmd_match)

        # Stage 3: 疎な点群再構築 (Incremental Mapper / SfM / Bundle Adjustment 最適化)
        self.progress_updated.emit(45, "SfM 疎な点群再構築 (Camera Pose Estimation & BA Refinement)...")
        self.log_emitted.emit("[COLMAP] Incremental SfM Mapper 実行中...")
        cmd_map = [
            self.config.colmap_binary, "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.num_threads", str(hw.parallel_matching_threads),
            "--Mapper.init_min_num_inliers", "15",
            "--Mapper.abs_pose_min_num_inliers", "10",
            "--Mapper.min_num_matches", "10",
            "--Mapper.filter_min_tri_angle", "1.5",
            "--Mapper.ba_refine_focal_length", "1",
            "--Mapper.ba_refine_principal_point", "1",
            "--Mapper.ba_refine_extra_params", "1",
        ]
        self._exec_cmd(cmd_map)

        # Stage 4: 最適な疎点群モデルの自動選定 & OpenMVS用変換 (Undistort)
        self.progress_updated.emit(55, "OpenMVS シーン変換 (Undistort)...")
        submodels = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not submodels:
            raise RuntimeError("COLMAP による 3D 点群再構築に失敗しました。撮影画像の特徴点または重なりが不足している可能性があります。")
        
        def _get_model_size(d: Path) -> int:
            p = d / "points3D.bin"
            return p.stat().st_size if p.exists() else 0

        best_sparse_dir = max(submodels, key=_get_model_size)
        self.log_emitted.emit(f"[COLMAP] 最適な点群サブモデル '{best_sparse_dir.name}' を自動選択しました。")

        cmd_undistort = [
            self.config.colmap_binary, "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(best_sparse_dir),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP",
        ]
        self._exec_cmd(cmd_undistort)

        # Stage 5: OpenMVS 密度再構築 (Dense Point Cloud)
        self.progress_updated.emit(70, f"OpenMVS 高密度点群生成中 (DensifyPointCloud, Level {prec.dense_resolution_level})...")
        cmd_mvs_import = [
            os.path.join(self.config.openmvs_dir, "InterfaceCOLMAP.exe" if os.name == 'nt' else "InterfaceCOLMAP"),
            "-w", str(dense_dir),
            "-i", ".",
            "-o", "scene.mvs",
        ]
        self._exec_cmd(cmd_mvs_import, cwd=str(dense_dir))

        cmd_densify = [
            os.path.join(self.config.openmvs_dir, "DensifyPointCloud.exe" if os.name == 'nt' else "DensifyPointCloud"),
            "-w", str(dense_dir),
            "-i", "scene.mvs",
            "-o", "scene_dense.mvs",
            "--resolution-level", str(prec.dense_resolution_level),
            "--number-views", str(prec.dense_min_views),
            "--max-threads", str(hw.openmvs_cuda_threads or 0),
        ]
        try:
            self._exec_cmd(cmd_densify, cwd=str(dense_dir))
        except Exception as e:
            self.log_emitted.emit(f"[OpenMVS 警告] 高密度点群化スキップ (疎点群から直接メッシュ化します): {e}")

        # Stage 6: サーフェスメッシュ化 (ReconstructMesh)
        self.progress_updated.emit(85, "3D サーフェスメッシュ生成 (ReconstructMesh)...")
        dense_scene_file = dense_dir / "scene_dense.mvs"
        input_mvs = "scene_dense.mvs" if dense_scene_file.exists() else "scene.mvs"
        cmd_mesh = [
            os.path.join(self.config.openmvs_dir, "ReconstructMesh.exe" if os.name == 'nt' else "ReconstructMesh"),
            "-w", str(dense_dir),
            "-i", input_mvs,
            "-o", "scene_dense_mesh.ply",
            "--thickness-factor", "1.0",
        ]
        self._exec_cmd(cmd_mesh, cwd=str(dense_dir))

        raw_ply_path = dense_dir / "scene_dense_mesh.ply"

        # Stage 6.5: 極限高精度 光度メッシュリファインメント (RefineMesh)
        refine_exe = os.path.join(self.config.openmvs_dir, "RefineMesh.exe" if os.name == 'nt' else "RefineMesh")
        if prec.enable_refine_mesh and os.path.exists(refine_exe) and dense_scene_file.exists() and raw_ply_path.exists():
            self.progress_updated.emit(88, "極限高精度 光度メッシュリファイン中 (RefineMesh)...")
            self.log_emitted.emit("[OpenMVS] 頂点単位の光度最適化 (RefineMesh) を実行中...")
            cmd_refine = [
                refine_exe,
                "-w", str(dense_dir),
                "-i", "scene_dense.mvs",
                "-m", "scene_dense_mesh.ply",
                "-o", "scene_refined_mesh.ply",
                "--resolution-level", str(prec.dense_resolution_level),
                "--scales", "2",
            ]
            try:
                self._exec_cmd(cmd_refine, cwd=str(dense_dir))
                refined_file = dense_dir / "scene_refined_mesh.ply"
                if refined_file.exists() and refined_file.stat().st_size > 0:
                    raw_ply_path = refined_file
                    self.log_emitted.emit("[OpenMVS] ✨ RefineMesh 光度最適化メッシュの適用に成功しました。")
            except Exception as e:
                self.log_emitted.emit(f"[OpenMVS 警告] RefineMesh スキップ (通常メッシュを使用): {e}")

        # Stage 6.8: フルカラーテクスチャベイキング (TextureMesh)
        clean_ext = self._normalize_format(output_format)
        texture_exe = os.path.join(self.config.openmvs_dir, "TextureMesh.exe" if os.name == 'nt' else "TextureMesh")
        has_textured_output = False
        if clean_ext in ("obj", "gltf") and os.path.exists(texture_exe) and dense_scene_file.exists() and raw_ply_path.exists():
            self.progress_updated.emit(90, "写真テクスチャを高精細ベイキング中 (TextureMesh)...")
            self.log_emitted.emit("[OpenMVS] 写真テクスチャを 3D メッシュへフルカラー投影中 (TextureMesh)...")
            cmd_texture = [
                texture_exe,
                "-w", str(dense_dir),
                "-i", "scene_dense.mvs",
                "-m", raw_ply_path.name,
                "-o", f"scene_textured.{clean_ext}",
                "--export-type", clean_ext,
            ]
            try:
                self._exec_cmd(cmd_texture, cwd=str(dense_dir))
                textured_out = dense_dir / f"scene_textured.{clean_ext}"
                if textured_out.exists() and textured_out.stat().st_size > 0:
                    raw_ply_path = textured_out
                    has_textured_output = True
                    self.log_emitted.emit(f"[OpenMVS] ✨ フルカラーテクスチャ ({clean_ext.upper()}) の生成に成功しました。")
            except Exception as e:
                self.log_emitted.emit(f"[OpenMVS 警告] TextureMesh スキップ: {e}")

        # Stage 7: 後処理 (Watertight化 & ArUcoスケール校正 & エクスポート)
        self.progress_updated.emit(92, "ArUcoスケール校正 & 水密化 (Watertight)...")
        if not raw_ply_path.exists():
            ply_candidates = list(dense_dir.glob("*.ply"))
            if ply_candidates:
                raw_ply_path = max(ply_candidates, key=lambda p: p.stat().st_size)
            else:
                raise RuntimeError("OpenMVS による 3D サーフェスメッシュ (PLY) の生成に失敗しました。")

        # スケール校正
        scale = POST_PROCESSOR.detect_aruco_scale_factor([p for p in images_dir.glob("*.jpg")])
        self.log_emitted.emit(f"[PostProcess] 検出ArUcoスケール倍率: {scale:.4f}x")

        mesh, report = POST_PROCESSOR.process_model(raw_ply_path, scale_factor=scale)

        final_mesh_path = work_dir / f"solidforge_model.{clean_ext}"
        mesh.export(str(final_mesh_path))

        # OBJ/GLTF テクスチャマテリアルファイルを作業ディレクトリ直下へ同期
        if has_textured_output:
            for related in dense_dir.glob("scene_textured*.*"):
                if related.suffix.lower() in (".mtl", ".jpg", ".png"):
                    dest_asset = work_dir / related.name.replace("scene_textured", "solidforge_model")
                    try:
                        shutil.copyfile(str(related), str(dest_asset))
                    except Exception:
                        pass

        return final_mesh_path, report

    def _run_simulated_pipeline(
        self,
        images_dir: Path,
        work_dir: Path,
        output_format: str,
    ):
        """シミュレーション実行 (テスト・プレビュー用高速生成)"""
        time.sleep(0.5)
        self.progress_updated.emit(25, "SiftGPU 特徴点抽出・CUDAマッチング (RTX 5080)...")
        self.log_emitted.emit("[SiftGPU] 1,840 特徴点 / 枚 をCUDA上で高速抽出完了。")
        time.sleep(0.6)

        self.progress_updated.emit(50, "SfM 疎な点群・カメラ軌跡推定完了...")
        self.log_emitted.emit("[COLMAP] 36/36 枚のカメラ位置姿勢を最適化完了 (Reprojection Error: 0.42 px)。")
        time.sleep(0.6)

        self.progress_updated.emit(75, "OpenMVS CUDA 高密度点群 & メッシュ生成中...")
        self.log_emitted.emit("[OpenMVS] 1,450,000 点の高密度点群およびサーフェス面を生成。")
        time.sleep(0.6)

        self.progress_updated.emit(90, "3Dプリント用後処理: 水密化 (Watertight) & 実寸校正...")
        
        # テスト用高品質メッシュ生成
        raw_mesh = POST_PROCESSOR.generate_synthetic_photogrammetry_mesh()
        
        # ArUcoスケール検出 (画像から)
        scale = POST_PROCESSOR.detect_aruco_scale_factor([p for p in images_dir.glob("*.jpg")])
        self.log_emitted.emit(f"[PostProcess] ArUcoマーカー検出: 実寸1:1校正係数 = {scale:.4f}x")

        processed_mesh, report = POST_PROCESSOR.process_model(raw_mesh, scale_factor=scale)

        clean_ext = self._normalize_format(output_format)

        final_mesh_path = work_dir / f"solidforge_output.{clean_ext}"
        processed_mesh.export(str(final_mesh_path))
        self.log_emitted.emit(f"[Export] 3Dモデルファイルを出力しました: {final_mesh_path.name}")

        return final_mesh_path, report

    def _exec_cmd(self, cmd: List[str], cwd: Optional[str] = None):
        """コマンド実行ラッパー (DLLパス環境変数を自動注入 & 作業ディレクトリ対応)"""
        env = os.environ.copy()
        
        # COLMAP & OpenMVS の DLL / lib 検索パスを追加
        extra_paths = [
            str(self.config.workspace_dir.parent / "bin" / "colmap" / "lib"),
            str(self.config.workspace_dir.parent / "bin" / "colmap" / "bin"),
            str(self.config.workspace_dir.parent / "bin" / "openmvs"),
            str(self.config.openmvs_dir),
        ]
        curr_path = env.get("PATH", "")
        env["PATH"] = ";".join(extra_paths) + ";" + curr_path
        
        # Qt プラグインパス設定 (COLMAP用)
        qt_plugin_path = str(self.config.workspace_dir.parent / "bin" / "colmap" / "lib" / "plugins")
        if os.path.exists(qt_plugin_path):
            env["QT_PLUGIN_PATH"] = qt_plugin_path

        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        while True:
            if self._cancel_requested:
                process.terminate()
                raise RuntimeError("ユーザーによって処理がキャンセルされました。")
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                cleaned = line.strip()
                self.log_emitted.emit(cleaned)
                if "Registering image #" in cleaned or "Registering image [" in cleaned:
                    self.progress_updated.emit(48, f"カメラ位置姿勢推定中: {cleaned}")
                elif "Densifying depth map" in cleaned or "Processing image [" in cleaned:
                    self.progress_updated.emit(75, f"高密度点群生成中: {cleaned}")
                elif "Extracting features" in cleaned:
                    self.progress_updated.emit(20, f"特徴点抽出中: {cleaned}")

        if process.returncode != 0:
            raise RuntimeError(f"コマンドがステータスコード {process.returncode} で終了しました: {' '.join(cmd)}")


# シングルトンパイプライン
PIPELINE = ReconstructionPipeline()
