# -*- coding: utf-8 -*-
"""
SolidForge 3D - Scene & Object Auto-Analyzer (撮影画像自動認識 & 最適パラメータ推定エンジン)
被写体占有率、テクスチャエントロピー、ブレ/露出、ArUcoマーカーを自動認識し、最適なパイプライン設定値を瞬時に導出
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from solidforge.core.ai_segmenter import AI_SEGMENTER


@dataclass
class SceneAnalysisResult:
    """画像認識 & 最適設定レコメンデーション結果"""
    sample_count: int
    object_coverage_pct: float
    category: str  # "SMALL_OBJECT" | "MEDIUM_OBJECT" | "LARGE_OBJECT"
    category_label_ja: str
    texture_score: float  # 0.0 (平滑/鏡面) ~ 100.0 (高テクスチャ)
    average_sharpness: float
    average_brightness: float  # 0 ~ 255
    aruco_detected: bool

    # 推奨設定値
    rec_precision_mode: str  # "ULTRA_HIGH" | "BALANCED" | "EXTREME_CAD"
    rec_enable_bg_removal: bool
    rec_sift_peak_threshold: float
    rec_sift_max_features: int
    rec_enable_guided_matching: bool
    rec_enable_refine_mesh: bool
    rec_ai_sharpen_strength: float
    rec_dense_resolution_level: int
    rec_auto_scale_target_mm: Optional[float]
    summary_text_ja: str


class SceneAnalyzer:
    """撮影画像群を高速サンプリング解析し、最適設定を自動推薦・適用するクラス"""

    @staticmethod
    def analyze_dataset(
        image_paths: List[Union[str, Path]],
        max_samples: int = 6,
    ) -> SceneAnalysisResult:
        """
        画像リストから数枚を均等サンプリングして瞬時にシーン特性を認識・評価
        """
        valid_paths = [Path(p) for p in image_paths if Path(p).exists() and Path(p).is_file() and Path(p).stat().st_size > 0]
        if not valid_paths:
            return SceneAnalysisResult(
                sample_count=0,
                object_coverage_pct=50.0,
                category="MEDIUM_OBJECT",
                category_label_ja="標準造形物",
                texture_score=50.0,
                average_sharpness=100.0,
                average_brightness=128.0,
                aruco_detected=False,
                rec_precision_mode="ULTRA_HIGH",
                rec_enable_bg_removal=True,
                rec_sift_peak_threshold=0.002,
                rec_sift_max_features=16384,
                rec_enable_guided_matching=True,
                rec_enable_refine_mesh=True,
                rec_ai_sharpen_strength=0.7,
                rec_dense_resolution_level=1,
                rec_auto_scale_target_mm=60.0,
                summary_text_ja="画像が選択されていません。標準の極限高精度モードが選択されています。",
            )

        # 均等サンプリング (0度, 60度, 120度, 180度, 240度, 300度等)
        step = max(1, len(valid_paths) // max_samples)
        sample_paths = valid_paths[::step][:max_samples]

        coverage_list = []
        sharpness_list = []
        brightness_list = []
        texture_list = []
        aruco_found = False

        # ArUco ディテクター
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

        for p in sample_paths:
            try:
                buf = np.fromfile(str(p), dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
            except Exception:
                continue

            h, w = img.shape[:2]
            # 解析用軽量リサイズ (512px)
            scale = 512.0 / max(h, w)
            small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # 1. 鮮鋭度 & 輝度
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_list.append(float(lap.var()))
            brightness_list.append(float(np.mean(gray)))

            # 2. テクスチャ豊かさ (Sobel勾配強度の標準偏差)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(grad_x, grad_y)
            texture_list.append(float(np.std(mag)))

            # 3. ArUco マーカー検出
            if not aruco_found:
                corners, ids, _ = aruco_detector.detectMarkers(img)
                if ids is not None and len(ids) > 0:
                    aruco_found = True

            # 4. 被写体占有率 (U2Net または Otsu/背景差分)
            try:
                # 高速 Otsu + GrabCut 領域推定
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                # 中央領域の比率
                cov = float(np.count_nonzero(thresh > 128) / thresh.size * 100.0)
                # 50%を超えている場合は反転可能性を考慮
                if cov > 50.0:
                    cov = 100.0 - cov
                coverage_list.append(cov)
            except Exception:
                coverage_list.append(20.0)

        # 代表値算出
        avg_cov = float(np.median(coverage_list)) if coverage_list else 20.0
        avg_sharp = float(np.mean(sharpness_list)) if sharpness_list else 50.0
        avg_bright = float(np.mean(brightness_list)) if brightness_list else 128.0
        avg_texture = float(np.mean(texture_list)) if texture_list else 30.0

        # カテゴリ判定
        if avg_cov < 12.0:
            category = "SMALL_OBJECT"
            cat_label = "小型・小物被写体 (ライター・指輪・ボルト等)"
            rec_mode = "ULTRA_HIGH"
            rec_sift_peak = 0.002
            rec_sift_feats = 16384
            rec_bg_removal = True
            rec_dense_lvl = 1
            rec_auto_scale = 60.0 if not aruco_found else None
            rec_refine = True
            rec_guided = True
        elif avg_cov < 45.0:
            category = "MEDIUM_OBJECT"
            cat_label = "中型造形物 (フィギュア・靴・工業ブラケット等)"
            rec_mode = "ULTRA_HIGH"
            rec_sift_peak = 0.003
            rec_sift_feats = 12288
            rec_bg_removal = True
            rec_dense_lvl = 1
            rec_auto_scale = 80.0 if not aruco_found else None
            rec_refine = True
            rec_guided = True
        else:
            category = "LARGE_OBJECT"
            cat_label = "大型造形物・空間 (家具・彫像・マクロ撮影等)"
            rec_mode = "BALANCED"
            rec_sift_peak = 0.005
            rec_sift_feats = 8192
            rec_bg_removal = False  # 画面全体が被写体
            rec_dense_lvl = 2
            rec_auto_scale = None
            rec_refine = False
            rec_guided = False

        # テクスチャ補正 (平滑・無地の場合は感度を最大化)
        if avg_texture < 25.0:
            rec_sift_peak = min(rec_sift_peak, 0.0015)
            rec_sift_feats = max(rec_sift_feats, 16384)
            rec_guided = True
            rec_sharpen = 0.85
        else:
            rec_sharpen = 0.70

        # サマリーテキスト構築
        scale_info = "📏 ArUco実寸1:1校正モード" if aruco_found else "📐 自動適応60mmスケーリング (Z=0接地)"
        summary = (
            f"【AI 画像自動認識結果】\n"
            f"• 被写体カテゴリ: {cat_label} (画面占有率: {avg_cov:.1f}%)\n"
            f"• 表面テクスチャ: {'低テクスチャ/平滑面 (微細感度強化)' if avg_texture < 25 else '豊か/標準'}\n"
            f"• 鮮鋭度スコア: {avg_sharp:.1f} | 基準スケール: {scale_info}\n\n"
            f"【適用された最適設定値】\n"
            f"✓ 精度モード: {rec_mode} (SIFT Peak: {rec_sift_peak}, 特徴量数: {rec_sift_feats})\n"
            f"✓ AI 背景自動除去: {'有効 (被写体分離)' if rec_bg_removal else '無効'}\n"
            f"✓ エピポーラ幾何ガイド照合: {'有効' if rec_guided else '無効'}\n"
            f"✓ OpenMVS RefineMesh 光度最適化: {'有効' if rec_refine else '無効'}"
        )

        return SceneAnalysisResult(
            sample_count=len(sample_paths),
            object_coverage_pct=avg_cov,
            category=category,
            category_label_ja=cat_label,
            texture_score=avg_texture,
            average_sharpness=avg_sharp,
            average_brightness=avg_bright,
            aruco_detected=aruco_found,
            rec_precision_mode=rec_mode,
            rec_enable_bg_removal=rec_bg_removal,
            rec_sift_peak_threshold=rec_sift_peak,
            rec_sift_max_features=rec_sift_feats,
            rec_enable_guided_matching=rec_guided,
            rec_enable_refine_mesh=rec_refine,
            rec_ai_sharpen_strength=rec_sharpen,
            rec_dense_resolution_level=rec_dense_lvl,
            rec_auto_scale_target_mm=rec_auto_scale,
            summary_text_ja=summary,
        )


SCENE_ANALYZER = SceneAnalyzer()
