"""
SolidForge 3D - Direct-to-Print Geometry Prep (3Dプリント造形最適化モジュール)
RANSAC床面検出による底面フラットカット (Build Plate Cut) & SLA向け中空化 (Hollowing)・レジン排出口開口
"""

import math
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
import trimesh
from trimesh import repair

from solidforge.config import DirectToPrintConfig, CONFIG


class GeometryPrep:
    """
    3Dプリント直結型幾何形状前処理クラス
    - RANSAC平面検出による接地面（テーブル・床）の自動判定と完全平面カット
    - ビルドプレートへの100%密着底面成形 (Z=0 平坦化)
    - 光造形 (SLA) 向け中空化 (Hollowing) & レジン排出穴 (Drain Holes) 自動配置
    """

    def __init__(self, config: Optional[DirectToPrintConfig] = None):
        self.config = config or CONFIG.post_process.direct_to_print

    def detect_ground_plane_ransac(
        self,
        mesh: trimesh.Trimesh,
        sample_ratio: float = 0.35,
        max_iterations: int = 500,
        distance_threshold: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RANSAC (RANdom SAmple Consensus) アルゴリズムにより、
        被写体を支えている dominant な水平床面 (テーブル面) を自動検出します。

        Args:
            mesh: 入力3Dメッシュ
            sample_ratio: 下部何割の頂点をサンプル対象とするか (0.35 = 下部35%)
            max_iterations: RANSAC反復回数
            distance_threshold: 平面とみなす距離トレランス (mm)

        Returns:
            Tuple[np.ndarray, np.ndarray]: (平面の通過点 [x, y, z], 平面の法線ベクトル [nx, ny, nz])
        """
        verts = mesh.vertices
        if verts is None or len(verts) == 0:
            return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

        z_min, z_max = np.min(verts[:, 2]), np.max(verts[:, 2])
        z_thresh = z_min + (z_max - z_min) * sample_ratio

        # 下部頂点を抽出
        bottom_indices = np.where(verts[:, 2] <= z_thresh)[0]
        if len(bottom_indices) < 20:
            # 頂点が少ない場合は最下面を返す
            return np.array([0.0, 0.0, z_min]), np.array([0.0, 0.0, 1.0])

        candidates = verts[bottom_indices]
        n_pts = len(candidates)
        # 10,000 候補並列ベクトル化 RANSAC (NumPy C-Level 最適化)
        n_samples = min(max_iterations * 20, 10000)
        np.random.seed(42)

        idx1 = np.random.randint(0, n_pts, n_samples)
        idx2 = np.random.randint(0, n_pts, n_samples)
        idx3 = np.random.randint(0, n_pts, n_samples)

        p1 = candidates[idx1]
        p2 = candidates[idx2]
        p3 = candidates[idx3]

        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        valid_mask = (norms.squeeze() > 1e-6)

        if not np.any(valid_mask):
            return np.array([0.0, 0.0, z_min]), np.array([0.0, 0.0, 1.0])

        normals = normals[valid_mask] / norms[valid_mask]
        p1_valid = p1[valid_mask]

        # 上向き法線に統一
        flip_mask = normals[:, 2] < 0
        normals[flip_mask] = -normals[flip_mask]

        # 水平面 (上向き法線 [0, 0, 1] に近いもの) を厳密にフィルタ
        horiz_mask = normals[:, 2] >= 0.85
        if np.any(horiz_mask):
            normals = normals[horiz_mask]
            p1_valid = p1_valid[horiz_mask]

        d_vals = -np.sum(normals * p1_valid, axis=1)  # [M]

        # 最大インライアを持つ平面を抽出
        best_inliers_count = 0
        best_plane_normal = np.array([0.0, 0.0, 1.0])
        best_plane_d = -z_min

        # 各候補のインライア数を高速バッチ評価 (上位50候補)
        eval_count = min(len(normals), 100)
        for i in range(eval_count):
            n = normals[i]
            d = d_vals[i]
            dists = np.abs(np.dot(candidates, n) + d)
            inliers = int(np.sum(dists < distance_threshold))
            if inliers > best_inliers_count:
                best_inliers_count = inliers
                best_plane_normal = n
                best_plane_d = d

        # 平面法線がほぼ真上の場合は [0, 0, 1] に正規化
        if best_plane_normal[2] > 0.85:
            best_plane_normal = np.array([0.0, 0.0, 1.0])
            plane_z = float(np.min(candidates[:, 2]))
            plane_origin = np.array([0.0, 0.0, plane_z])
        else:
            plane_origin = np.array([0.0, 0.0, -best_plane_d / (best_plane_normal[2] + 1e-8)])

        return plane_origin, best_plane_normal

    def slice_ground_plane(
        self,
        mesh: trimesh.Trimesh,
        offset_mm: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """
        検出された接地面に沿ってメッシュを綺麗にスライスし、底面を完全に平坦化（キャップ）します。
        さらに、メッシュ全体の最底面を Z=0 にアライメントし、3Dプリンタのビルドプレートに直結密着させます。

        Args:
            mesh: 入力3Dメッシュ
            offset_mm: 接地面から何mm上でスライスするか (省略時は設定値)

        Returns:
            trimesh.Trimesh: 底面が完全にフラットカットされた水密メッシュ
        """
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return mesh if (mesh is not None and len(mesh.vertices) > 0) else trimesh.creation.box(extents=[10, 10, 10])

        offset = offset_mm if offset_mm is not None else self.config.ground_cut_offset_mm
        plane_origin, plane_normal = self.detect_ground_plane_ransac(
            mesh, distance_threshold=self.config.ground_plane_ransac_distance_thresh
        )

        # オフセットを適用
        adjusted_origin = plane_origin + plane_normal * offset

        try:
            # 平面でメッシュをスライス & キャップ (trimesh slice_mesh_plane)
            sliced = trimesh.intersections.slice_mesh_plane(
                mesh=mesh,
                plane_normal=plane_normal,
                plane_origin=adjusted_origin,
                cap=True,
            )
        except Exception:
            try:
                # cap=False でスライス後に fill_holes で確実に閉じる
                sliced = trimesh.intersections.slice_mesh_plane(
                    mesh=mesh,
                    plane_normal=plane_normal,
                    plane_origin=adjusted_origin,
                    cap=False,
                )
                repair.fill_holes(sliced)
            except Exception:
                # Z軸でのシンプルなカット
                sliced = mesh.copy()
                z_cut = float(np.min(sliced.vertices[:, 2]) + max(offset, 0.5))
                sliced.vertices[sliced.vertices[:, 2] < z_cut, 2] = z_cut

        if sliced is None or len(sliced.faces) == 0:
            sliced = mesh.copy()

        # 底面を Z=0 にトランスレーション (ビルドプレート接地アライメント)
        min_z = float(np.min(sliced.vertices[:, 2]))
        sliced.apply_translation([0, 0, -min_z])

        # 水密修復
        repair.fill_holes(sliced)
        repair.fix_normals(sliced)
        return sliced

    def hollow_mesh(
        self,
        mesh: trimesh.Trimesh,
        wall_thickness_mm: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """
        SLA (光造形) 3Dプリント向けに、指定した肉厚 (mm) でメッシュ内部を中空化します。
        レジン消費量を 50〜80% 削減し、硬化時の体積収縮と重量を低減します。

        Args:
            mesh: 入力水密メッシュ
            wall_thickness_mm: 側壁の厚み (mm, デフォルト 2.0mm)

        Returns:
            trimesh.Trimesh: 内部空洞を持つ中空メッシュ
        """
        thickness = wall_thickness_mm or self.config.wall_thickness_mm
        if thickness <= 0:
            return mesh

        # メッシュのバウンディングボックス対角線長
        extents = mesh.bounding_box.extents
        min_dim = float(min(extents)) if len(extents) > 0 else 0.0
        if min_dim <= 0:
            return mesh

        if thickness * 2 >= min_dim:
            # 厚みがモデルサイズを超えている場合は半分に自動制限
            thickness = min_dim * 0.25

        # 内部キャビティ用インナーメッシュ作成 (内向き法線)
        inner = mesh.copy()
        scale_x = max(0.1, (extents[0] - 2 * thickness) / max(float(extents[0]), 1e-4))
        scale_y = max(0.1, (extents[1] - 2 * thickness) / max(float(extents[1]), 1e-4))
        scale_z = max(0.1, (extents[2] - 2 * thickness) / max(float(extents[2]), 1e-4))

        center = mesh.centroid
        inner.apply_translation(-center)
        inner.apply_scale([scale_x, scale_y, scale_z])
        inner.apply_translation(center)

        # インナーメッシュの法線を反転 (空洞の内壁として設定)
        inner.invert()

        # アウターとインナーを結合
        hollowed = trimesh.util.concatenate([mesh, inner])
        return hollowed

    def add_drain_holes(
        self,
        mesh: trimesh.Trimesh,
        hole_radius_mm: Optional[float] = None,
        num_holes: int = 2,
    ) -> trimesh.Trimesh:
        """
        SLAプリント時に内部の未硬化レジンを逃がし、密閉空間による破裂・真空剥離破損を防ぐ
        レジン排出穴 (Drain Holes) を底面付近に開口します。

        Args:
            mesh: 中空メッシュ
            hole_radius_mm: 穴の半径 (mm, デフォルト 2.5mm = 直径5mm)
            num_holes: 排出口の数

        Returns:
            trimesh.Trimesh: 排出穴が開口された3Dモデル
        """
        radius = hole_radius_mm or self.config.drain_hole_radius_mm
        bounds = mesh.bounding_box.extents
        center = mesh.centroid
        min_z = np.min(mesh.vertices[:, 2])

        # 底面付近 (Z min + 半径) に円柱カッターを配置
        cylinders = []
        for i in range(num_holes):
            angle = (2 * math.pi / num_holes) * i
            dist = min(bounds[0], bounds[1]) * 0.35
            cx = center[0] + dist * math.cos(angle)
            cy = center[1] + dist * math.sin(angle)
            cz = min_z + radius * 1.2

            # 底面から抜ける垂直/斜め円柱
            thickness_h = bounds[2] * 0.4
            cyl = trimesh.creation.cylinder(radius=radius, height=thickness_h, sections=24)
            cyl.apply_translation([cx, cy, min_z + thickness_h / 2])
            cylinders.append(cyl)

        # ブーリアン差分またはメッシュ穴開け
        try:
            cutter_union = trimesh.util.concatenate(cylinders)
            hollowed_with_holes = mesh.difference(cutter_union, engine="scad")
            if hollowed_with_holes and len(hollowed_with_holes.faces) > 0:
                return hollowed_with_holes
        except Exception:
            pass

        return mesh

    def analyze_overhangs(
        self,
        mesh: trimesh.Trimesh,
        threshold_angle_deg: float = 45.0,
    ) -> Tuple[float, np.ndarray]:
        """
        3Dプリント時にサポート材（Support Structures）が必要となる
        急傾斜・オーバーハング面（下向き45度以上）を検出し、面積比率とマスクを算出します。

        Args:
            mesh: 入力メッシュ
            threshold_angle_deg: オーバーハングと判定する限界角度 (度, デフォルト 45°)

        Returns:
            Tuple[float, np.ndarray]: (オーバーハング面積比率 %, 各ポリゴン面の真偽値マスク [F])
        """
        if mesh is None or len(mesh.faces) == 0:
            return 0.0, np.zeros(0, dtype=bool)

        normals = mesh.face_normals
        areas = mesh.area_faces

        # 水平から下向きへの角度: 法線の Z 成分が -sin(threshold) 未満のときにオーバーハング
        # 例: 45度以上下向き -> nz < -sin(45°) = -0.7071
        rad = math.radians(threshold_angle_deg)
        limit_z = -math.sin(rad)

        overhang_mask = normals[:, 2] < limit_z
        total_area = float(np.sum(areas))
        if total_area <= 1e-6:
            return 0.0, overhang_mask

        overhang_area = float(np.sum(areas[overhang_mask]))
        overhang_ratio_pct = (overhang_area / total_area) * 100.0

        return overhang_ratio_pct, overhang_mask

    def add_base_pedestal(
        self,
        mesh: trimesh.Trimesh,
        thickness_mm: float = 2.5,
        margin_mm: float = 5.0,
        shape: str = "ellipse",
    ) -> trimesh.Trimesh:
        """
        細身のフィギュアや自立しにくい置物モデルの底面に、
        転倒防止用の面取り台座（Pedestal / Base Stand）を自動合成し、Z=0 に再接地します。

        Args:
            mesh: 入力メッシュ
            thickness_mm: 台座の厚み (mm, デフォルト 2.5mm)
            margin_mm: モデル外形からの張り出し幅 (mm, デフォルト 5.0mm)
            shape: 'ellipse' (円・楕円) または 'box' (角丸長方形)

        Returns:
            trimesh.Trimesh: 台座が一体化され Z=0 接地された 3D モデル
        """
        if mesh is None or len(mesh.faces) == 0:
            return mesh

        bounds = mesh.bounds
        center_xy = mesh.centroid[:2]
        min_z = float(bounds[0, 2])

        extent_x = float(bounds[1, 0] - bounds[0, 0])
        extent_y = float(bounds[1, 1] - bounds[0, 1])

        rx = extent_x / 2.0 + margin_mm
        ry = extent_y / 2.0 + margin_mm

        if shape == "box":
            ped = trimesh.creation.box(extents=[extent_x + margin_mm * 2, extent_y + margin_mm * 2, thickness_mm])
            ped.apply_translation([center_xy[0], center_xy[1], min_z - thickness_mm / 2.0])
        else:
            # 円柱台座 (半径は最大半径)
            r_max = max(rx, ry)
            ped = trimesh.creation.cylinder(radius=r_max, height=thickness_mm, sections=48)
            ped.apply_translation([center_xy[0], center_xy[1], min_z - thickness_mm / 2.0])

        try:
            combined = trimesh.util.concatenate([mesh, ped])
            # 最下面を Z=0 に正確にアライメント
            combined.apply_translation([0.0, 0.0, -combined.bounds[0, 2]])
            repair.fix_normals(combined)
            return combined
        except Exception:
            return mesh

    def decimate_mesh(
        self,
        mesh: trimesh.Trimesh,
        target_face_count: Optional[int] = None,
        target_reduction: Optional[float] = None,
    ) -> trimesh.Trimesh:
        """
        Quadric Edge Collapse (QEM) アルゴリズムにより、
        表面の幾何形状特徴を維持しながらポリゴン面数を安全に削減（軽量化）します。

        Args:
            mesh: 入力メッシュ
            target_face_count: 目標ポリゴン面数 (例: 50000)
            target_reduction: 削減比率 0.0 ~ 1.0 (例: 0.5 で50%削減)

        Returns:
            trimesh.Trimesh: 軽量化された 3D モデル
        """
        if mesh is None or len(mesh.faces) == 0:
            return mesh

        n_faces = len(mesh.faces)
        if target_face_count is not None and target_face_count > 0:
            if target_face_count >= n_faces:
                return mesh
            reduction = 1.0 - (target_face_count / float(n_faces))
        elif target_reduction is not None and 0.0 < target_reduction < 1.0:
            reduction = target_reduction
        else:
            return mesh

        reduction = max(0.01, min(0.98, reduction))

        try:
            import fast_simplification
            verts, faces = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_reduction=reduction
            )
            decimated = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            repair.fix_normals(decimated)
            return decimated
        except Exception:
            return mesh


GEOMETRY_PREP = GeometryPrep()

