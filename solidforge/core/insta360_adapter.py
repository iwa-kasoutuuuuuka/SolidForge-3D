"""
SolidForge 3D - Insta360 Adapter & 360° Perspective Multi-Cam Unwrapper
Insta360 X5 / X4 / X3 向け 360度パノラマ透視投影展開 & Ace Pro 2 / Link 2 最適化
"""

import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from solidforge.config import CONFIG, Insta360Config


class Insta360Adapter:
    """
    Insta360 シリーズ専用アダプタ & 360°パノラマ透視投影展開プロセッサ
    - Insta360 Ace Pro 2 / Ace Pro / Link 2: 4K低遅延UVCキャプチャ
    - Insta360 X5 / X4 / X3: 正距円筒図法 (Equirectangular) から4〜6方向の仮想ピンホール透視画像を生成
    """

    def __init__(self, config: Optional[Insta360Config] = None):
        self.config = config or CONFIG.insta360
        self._remap_cache: Dict[Tuple[int, int, float, float, float], Tuple[np.ndarray, np.ndarray]] = {}

    def is_360_panorama(self, image: np.ndarray) -> bool:
        """画像が360度正距円筒図法 (アスペクト比 2:1) かどうかを判定"""
        if image is None or image.size == 0:
            return False
        h, w = image.shape[:2]
        aspect = w / max(h, 1)
        return 1.85 <= aspect <= 2.15

    def equirectangular_to_perspective(
        self,
        image: np.ndarray,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        fov_deg: float = 90.0,
        out_width: int = 1280,
        out_height: int = 720,
    ) -> np.ndarray:
        """
        正距円筒図法 (360° Equirectangular) 画像から、指定した向き (Yaw, Pitch) と画角 (FOV) の
        歪みのない透視投影 (Rectilinear Perspective) 画像を展開します。

        Args:
            image: 入力360°正距円筒画像 (BGR)
            yaw_deg: 水平回転角 (0°=正面, 90°=右, 180°=背面, 270°=左)
            pitch_deg: 垂直仰俯角 (+度=見上げ, -度=見下ろし)
            fov_deg: 水平視野角 (FOV)
            out_width: 出力幅 (px)
            out_height: 出力高さ (px)

        Returns:
            np.ndarray: 透視変換済み画像
        """
        if image is None or image.size == 0:
            return np.zeros((out_height, out_width, 3), dtype=np.uint8)

        in_h, in_w = image.shape[:2]
        cache_key = (in_w, in_h, round(yaw_deg, 1), round(pitch_deg, 1), round(fov_deg, 1))

        if cache_key in self._remap_cache:
            map_x, map_y = self._remap_cache[cache_key]
        else:
            # 透視投影レイの3次元方向ベクトルを計算
            f = 0.5 * out_width / math.tan(math.radians(fov_deg) * 0.5)
            cx = (out_width - 1) * 0.5
            cy = (out_height - 1) * 0.5

            x_grid, y_grid = np.meshgrid(np.arange(out_width), np.arange(out_height))
            x_norm = (x_grid - cx) / f
            y_norm = -(y_grid - cy) / f
            z_norm = np.ones_like(x_norm)

            # 回転行列の計算
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)

            # Yaw (Y軸回転) & Pitch (X軸回転)
            R_y = np.array([
                [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
                [0, 1, 0],
                [-math.sin(yaw_rad), 0, math.cos(yaw_rad)],
            ])
            R_x = np.array([
                [1, 0, 0],
                [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
                [0, math.sin(pitch_rad), math.cos(pitch_rad)],
            ])
            R = np.dot(R_y, R_x)

            # 3Dベクトルの回転
            rays = np.stack([x_norm, y_norm, z_norm], axis=-1)  # [H, W, 3]
            rays_rot = np.dot(rays, R.T)

            rx = rays_rot[..., 0]
            ry = rays_rot[..., 1]
            rz = rays_rot[..., 2]

            # 球面座標系へマッピング
            rho = np.sqrt(rx**2 + ry**2 + rz**2)
            theta = np.arctan2(rx, rz)  # [-pi, pi]
            phi = np.arcsin(np.clip(ry / rho, -1.0, 1.0))  # [-pi/2, pi/2]

            # 正距円筒図法のピクセル座標へ変換
            map_x = ((theta / math.pi + 1.0) * 0.5 * in_w).astype(np.float32)
            map_y = ((0.5 - phi / math.pi) * in_h).astype(np.float32)

            self._remap_cache[cache_key] = (map_x, map_y)

        # 高速バイリニア補間リマップ
        dewarped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        return dewarped

    def generate_virtual_multiviews(
        self,
        image_360: np.ndarray,
        num_views: int = 4,
        fov_deg: float = 90.0,
    ) -> List[Tuple[str, np.ndarray]]:
        """
        1枚の360度正距円筒パノラマから、SfM再構築に適したマルチアングル透視画像リストを生成します。

        Args:
            image_360: 360度パノラマ画像
            num_views: 展開方向数 (4 = 前/右/後/左, 6 = 前/右/後/左/斜上/斜下)
            fov_deg: 視野角

        Returns:
            List[Tuple[str, np.ndarray]]: (方向ラベル, 透視画像) のリスト
        """
        results: List[Tuple[str, np.ndarray]] = []
        if not self.is_360_panorama(image_360):
            return [("front", image_360)]

        # 水平4方向 (0°, 90°, 180°, 270°)
        angles = [
            ("front", 0.0, 0.0),
            ("right", 90.0, 0.0),
            ("back", 180.0, 0.0),
            ("left", 270.0, 0.0),
        ]

        if num_views >= 6:
            angles.extend([
                ("up_front", 0.0, 35.0),
                ("down_front", 0.0, -35.0),
            ])

        for label, yaw, pitch in angles:
            view_img = self.equirectangular_to_perspective(
                image_360,
                yaw_deg=yaw,
                pitch_deg=pitch,
                fov_deg=fov_deg,
                out_width=1280,
                out_height=720,
            )
            results.append((label, view_img))

        return results


INSTA360_ADAPTER = Insta360Adapter()
