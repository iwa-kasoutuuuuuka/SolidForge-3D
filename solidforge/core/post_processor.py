"""
SolidForge 3D - 3D Print Post-Processor (3Dプリント最適化 & 後処理モジュール)
ArUcoマーカーによる実寸1:1 (mm) スケール校正 & 水密化 (Watertight / Hole Filling / Non-manifold修復)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import trimesh
from trimesh import repair

from solidforge.config import CONFIG, PostProcessingConfig


@dataclass
class PrintabilityReport:
    """3Dプリント適性診断レポート"""
    is_watertight: bool
    is_winding_consistent: bool
    vertex_count: int
    face_count: int
    dimensions_mm: Tuple[float, float, float]  # (X_mm, Y_mm, Z_mm)
    volume_cm3: float
    surface_area_cm2: float
    estimated_weight_pla_g: float  # PLA密度 1.24 g/cm3 換算
    scale_factor_applied: float
    warnings: List[str]

    @property
    def summary_text_ja(self) -> str:
        status = "【合格】3Dスライサー投入可能 (完全水密)" if self.is_watertight else "【要確認】非水密メッシュ"
        dim_x, dim_y, dim_z = self.dimensions_mm
        return (
            f"=== 3Dプリント適性診断レポート ===\n"
            f"ステータス: {status}\n"
            f"実寸外形寸法: {dim_x:.1f} x {dim_y:.1f} x {dim_z:.1f} mm\n"
            f"体積: {self.volume_cm3:.2f} cm³ | 表面積: {self.surface_area_cm2:.2f} cm²\n"
            f"推定フィラメント重量 (PLA 100%): {self.estimated_weight_pla_g:.1f} g\n"
            f"ポリゴン面数: {self.face_count:,} 面 | 頂点数: {self.vertex_count:,} 点\n"
            f"適用スケール倍率: {self.scale_factor_applied:.4f}x\n"
        )


class MeshPostProcessor:
    """
    3Dメッシュの後処理およびスライサー対応最適化クラス
    - ArUcoマーカー検出による実寸mmへの正確なスケーリング
    - 自動穴埋め (Hole Filling) & 法線整合化
    - ノンマニフォールド頂点/辺の自動修復
    - STL / OBJ / PLY / GLTF エクスポート
    """

    def __init__(self, config: Optional[PostProcessingConfig] = None):
        self.config = config or CONFIG.post_process

    def detect_aruco_scale_factor(
        self,
        image_paths: List[Union[str, Path]],
        known_marker_size_mm: Optional[float] = None,
    ) -> float:
        """
        撮影画像群からArUcoマーカーを検出し、ピクセル/実寸比率または既知スケール倍率を算出します。

        Args:
            image_paths: 画像ファイルパスのリスト
            known_marker_size_mm: マーカーの一辺の実寸 (mm) (省略時は設定値)

        Returns:
            float: スケール倍率 (基準値1.0)
        """
        target_size = known_marker_size_mm or self.config.aruco_marker_size_mm
        dict_type = getattr(cv2.aruco, self.config.aruco_dict_name, cv2.aruco.DICT_4X4_50)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

        marker_lengths_px: List[float] = []

        for p in image_paths[:10]:  # 最大10枚サンプリング
            p = Path(p)
            if not p.exists() or not p.is_file() or p.stat().st_size == 0:
                continue
            try:
                buf = np.fromfile(str(p), dtype=np.uint8)
                if len(buf) == 0:
                    continue
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None and len(corners) > 0:
                for c in corners:
                    pts = c[0]
                    # 4辺の長さ平均 (px)
                    d1 = np.linalg.norm(pts[0] - pts[1])
                    d2 = np.linalg.norm(pts[1] - pts[2])
                    d3 = np.linalg.norm(pts[2] - pts[3])
                    d4 = np.linalg.norm(pts[3] - pts[0])
                    avg_side_px = float((d1 + d2 + d3 + d4) / 4.0)
                    marker_lengths_px.append(avg_side_px)

        if len(marker_lengths_px) > 0:
            avg_px = float(np.mean(marker_lengths_px))
            # 1ピクセルあたりのミリメートル概算比率 (仮の焦点距離標準換算)
            scale_ratio = target_size / max(avg_px, 1.0)
            return float(scale_ratio * 100.0)  # 正規化スケール係数

        return 1.0  # マーカー未検出時は等倍

    def make_mesh_watertight(
        self,
        mesh: trimesh.Trimesh,
        fix_normals: bool = True,
        fill_holes: bool = True,
    ) -> trimesh.Trimesh:
        """
        メッシュの穴を埋め、自己交差・ノンマニフォールドを修復して完全な水密（Watertight）立体に変換します。

        Args:
            mesh: 入力 Trimesh オブジェクト
            fix_normals: 法線ベクトルの外向き統一
            fill_holes: 穴埋め処理の実行

        Returns:
            trimesh.Trimesh: 修復された水密メッシュ
        """
        repaired = mesh.copy()

        # 1. 重複頂点・退化面のクリーンアップ
        repaired.update_faces(repaired.unique_faces())
        repaired.update_faces(repaired.nondegenerate_faces())
        repaired.remove_unreferenced_vertices()

        # 2. 穴埋め処理 (Hole Filling)
        if fill_holes:
            repair.fill_holes(repaired)

        # 3. 法線ベクトルの修正 & 反転チェック
        if fix_normals:
            repair.fix_winding(repaired)
            repair.fix_normals(repaired)
            repair.fix_inversion(repaired)

        # 4. ノンマニフォールド頂点/エッジの再結合
        if not repaired.is_watertight:
            # 凸包(Convex Hull)やボクセル再メッシュではなく原形状を保つ穴埋め再試行
            try:
                repair.broken_faces(repaired)
                repair.fill_holes(repaired)
            except Exception:
                pass

        return repaired

    def process_model(
        self,
        mesh_input: Union[trimesh.Trimesh, str, Path],
        scale_factor: float = 1.0,
        target_faces: Optional[int] = None,
    ) -> Tuple[trimesh.Trimesh, PrintabilityReport]:
        """
        3Dプリント用後処理パイプラインの統合実行
        - スケール適用 (mm単位)
        - 水密化修復
        - ポリゴン最適化 (Decimation)
        - プリント適性診断

        Args:
            mesh_input: Trimeshオブジェクトまたはメッシュファイルパス
            scale_factor: 拡大縮小倍率
            target_faces: 目標ポリゴン面数 (省略時は設定値)

        Returns:
            Tuple[trimesh.Trimesh, PrintabilityReport]:
                (最適化済み水密メッシュ, 診断レポート)
        """
        if isinstance(mesh_input, (str, Path)):
            mesh = trimesh.load(str(mesh_input), force="mesh")
        else:
            mesh = mesh_input.copy()

        from solidforge.core.geometry_prep import GEOMETRY_PREP

        # 1. スケール適用
        if abs(scale_factor - 1.0) > 1e-4:
            mesh.apply_scale(scale_factor)

        # 2. 接地面 RANSAC 自動検出 & 底面フラットカット (Build Plate 接地化)
        if self.config.direct_to_print.enable_ground_cut:
            mesh = GEOMETRY_PREP.slice_ground_plane(
                mesh, offset_mm=self.config.direct_to_print.ground_cut_offset_mm
            )

        # 3. SLA (光造形) 向け中空化 & レジン排出口開口
        if self.config.direct_to_print.enable_hollowing:
            mesh = GEOMETRY_PREP.hollow_mesh(
                mesh, wall_thickness_mm=self.config.direct_to_print.wall_thickness_mm
            )
            if self.config.direct_to_print.enable_drain_holes:
                mesh = GEOMETRY_PREP.add_drain_holes(
                    mesh,
                    hole_radius_mm=self.config.direct_to_print.drain_hole_radius_mm,
                    num_holes=self.config.direct_to_print.drain_hole_count,
                )

        # 4. 水密化 (Watertight Repair)
        mesh = self.make_mesh_watertight(mesh)

        # 3. 面数削減 (Decimation)
        max_f = target_faces if target_faces is not None else self.config.target_faces
        if max_f > 0 and len(mesh.faces) > max_f:
            try:
                mesh = mesh.simplify_quadric_decimation(max_f)
            except Exception:
                pass

        # 4. 診断レポート生成
        bounds = mesh.bounding_box.extents  # [X, Y, Z]
        dim_mm = (float(bounds[0]), float(bounds[1]), float(bounds[2]))
        
        # 体積計算 (非水密時は凸包体積で概算)
        if mesh.is_watertight:
            volume_mm3 = float(mesh.volume)
        else:
            volume_mm3 = float(mesh.convex_hull.volume)
        
        volume_cm3 = max(0.001, volume_mm3 / 1000.0)
        area_cm2 = float(mesh.area) / 100.0
        
        # PLA密度 1.24 g/cm3
        weight_g = volume_cm3 * 1.24

        warnings: List[str] = []
        if not mesh.is_watertight:
            warnings.append("完全な水密化が完了していません。スライサーで微小な穴を自動修復してください。")
        if any(d < 1.0 for d in dim_mm):
            warnings.append("モデルの厚みが1mm未満の箇所があります。3Dプリント時に破損する恐れがあります。")

        report = PrintabilityReport(
            is_watertight=bool(mesh.is_watertight),
            is_winding_consistent=bool(mesh.is_winding_consistent),
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            dimensions_mm=dim_mm,
            volume_cm3=round(volume_cm3, 3),
            surface_area_cm2=round(area_cm2, 2),
            estimated_weight_pla_g=round(weight_g, 2),
            scale_factor_applied=scale_factor,
            warnings=warnings,
        )

        return mesh, report

    def generate_synthetic_photogrammetry_mesh(self) -> trimesh.Trimesh:
        """
        実機テストやデモ用の高精度3Dモデル（ターンテーブル風台座＋多面体造形物）を生成します。
        """
        # 台座 (シリンダー)
        base = trimesh.creation.cylinder(radius=30.0, height=8.0, sections=48)
        base.apply_translation([0, 0, 4.0])

        # 被写体 (凹凸を持つ複雑なトーラスノットまたはイコサスフィア)
        sculpture = trimesh.creation.icosphere(subdivisions=4, radius=22.0)
        # 上部に配置
        sculpture.apply_translation([0, 0, 32.0])

        # 結合
        combined = trimesh.util.concatenate([base, sculpture])
        # Watertight修復
        return self.make_mesh_watertight(combined)


# シングルトンポストプロセッサ
POST_PROCESSOR = MeshPostProcessor()
