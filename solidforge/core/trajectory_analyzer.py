"""
SolidForge 3D - 3D Camera Trajectory & 360° Coverage Analyzer
カメラ軌跡の3D空間推定、360度カバレッジ解析、未撮影ギャップ（死角）の自動警告
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CameraPose:
    """単一カメラの3D位置と姿勢"""
    image_name: str
    position: np.ndarray  # [X, Y, Z]
    forward_vector: np.ndarray  # [dx, dy, dz]
    azimuth_deg: float  # 水平方位角 (0 - 360度)
    elevation_deg: float  # 仰角 (-90 - +90度)


@dataclass
class CoverageAnalysisResult:
    """360度撮影カバレッジの解析結果"""
    total_shots: int
    coverage_percentage: float  # 0.0 - 100.0%
    sector_counts: List[int]  # 各セクターの撮影枚数
    missing_angle_ranges: List[Tuple[float, float]]  # 未撮影角度区間 [(start_deg, end_deg), ...]
    average_angular_step_deg: float
    is_coverage_sufficient: bool  # 360度十分か
    guidance_message: str


class TrajectoryAnalyzer:
    """
    SfMカメラ位置姿勢およびLive View手持ち撮影軌跡の解析クラス
    """

    def __init__(self, num_sectors: int = 36):
        self.num_sectors = num_sectors  # 360度 / 36 = 10度刻み
        self.sector_span = 360.0 / num_sectors
        self.poses: List[CameraPose] = []

    def clear(self):
        self.poses.clear()

    def add_pose(self, name: str, position: np.ndarray, forward: Optional[np.ndarray] = None) -> CameraPose:
        """カメラ位置を追加し、方位角・仰角を算出"""
        pos = np.array(position, dtype=np.float32)
        # オブジェクト中心を (0,0,0) と仮定した相対方位
        azimuth = math.degrees(math.atan2(pos[1], pos[0])) % 360.0
        dist_xy = math.sqrt(pos[0]**2 + pos[1]**2)
        elevation = math.degrees(math.atan2(pos[2], max(dist_xy, 1e-4)))

        if forward is None:
            # デフォルトで原点(被写体)を向いていると仮定
            fwd = -pos / (np.linalg.norm(pos) + 1e-6)
        else:
            fwd = np.array(forward, dtype=np.float32)

        pose = CameraPose(
            image_name=name,
            position=pos,
            forward_vector=fwd,
            azimuth_deg=round(azimuth, 1),
            elevation_deg=round(elevation, 1),
        )
        self.poses.append(pose)
        return pose

    def generate_simulated_trajectory(self, count: int = 32, create_gap: bool = True) -> List[CameraPose]:
        """テスト用の360度カメラ軌跡を生成 (ギャップ有無対応)"""
        self.clear()
        radius = 80.0
        z_height = 25.0

        for i in range(count):
            angle = (360.0 / count) * i
            # ギャップテスト: 45度〜85度の撮影を意図的にスキップ
            if create_gap and (45.0 <= angle <= 85.0):
                continue

            rad = math.radians(angle)
            x = radius * math.cos(rad)
            y = radius * math.sin(rad)
            z = z_height + np.random.uniform(-1, 1)

            self.add_pose(f"frame_{i:03d}.jpg", np.array([x, y, z]))

        return self.poses

    def analyze_coverage(self) -> CoverageAnalysisResult:
        """360度全方位のカバレッジと未撮影ギャップ領域を解析"""
        if not self.poses:
            return CoverageAnalysisResult(
                total_shots=0,
                coverage_percentage=0.0,
                sector_counts=[0] * self.num_sectors,
                missing_angle_ranges=[(0.0, 360.0)],
                average_angular_step_deg=0.0,
                is_coverage_sufficient=False,
                guidance_message="撮影データがありません。被写体の周囲を360度撮影してください。",
            )

        counts = [0] * self.num_sectors
        for p in self.poses:
            sec_idx = int(p.azimuth_deg // self.sector_span) % self.num_sectors
            counts[sec_idx] += 1

        covered_sectors = sum(1 for c in counts if c > 0)
        coverage_pct = round((covered_sectors / self.num_sectors) * 100.0, 1)

        # ギャップ区間の検出
        missing_ranges: List[Tuple[float, float]] = []
        in_gap = False
        gap_start = 0.0

        for i in range(self.num_sectors):
            if counts[i] == 0:
                if not in_gap:
                    in_gap = True
                    gap_start = i * self.sector_span
            else:
                if in_gap:
                    in_gap = False
                    gap_end = i * self.sector_span
                    missing_ranges.append((gap_start, gap_end))

        if in_gap:
            missing_ranges.append((gap_start, 360.0))

        # メッセージ生成
        if not missing_ranges and coverage_pct >= 90.0:
            is_sufficient = True
            msg = "✨ 【カバレッジ良好】 360度全周囲の撮影が均等にカバーされています。"
        else:
            is_sufficient = False
            gaps_str = ", ".join([f"{int(s)}°〜{int(e)}°" for s, e in missing_ranges[:3]])
            msg = f"⚠️ 【撮影ギャップ検知】 方位 {gaps_str} の撮影が抜けています。この角度から追加撮影してください。"

        # 平均角度ステップ
        sorted_angles = sorted([p.azimuth_deg for p in self.poses])
        diffs = [
            (sorted_angles[(i + 1) % len(sorted_angles)] - sorted_angles[i]) % 360.0
            for i in range(len(sorted_angles))
        ]
        avg_step = float(np.mean(diffs)) if diffs else 0.0

        return CoverageAnalysisResult(
            total_shots=len(self.poses),
            coverage_percentage=coverage_pct,
            sector_counts=counts,
            missing_angle_ranges=missing_ranges,
            average_angular_step_deg=round(avg_step, 1),
            is_coverage_sufficient=is_sufficient,
            guidance_message=msg,
        )


TRAJECTORY_ANALYZER = TrajectoryAnalyzer()
