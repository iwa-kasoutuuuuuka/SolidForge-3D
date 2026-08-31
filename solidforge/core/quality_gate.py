"""
SolidForge 3D - Quality Gate (品質判定・前処理モジュール)
OpenCV Laplacian分散によるブレ自動検知およびSIFT/ORB特徴点抽出によるテクスチャ評価
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from solidforge.config import QualityGateConfig, CONFIG


@dataclass
class QualityEvaluationResult:
    """画像品質の評価結果"""
    is_accepted: bool
    blur_score: float
    feature_count: int
    rejection_reasons: List[str] = field(default_factory=list)
    image_path: Optional[Path] = None
    keypoints: Optional[List[cv2.KeyPoint]] = None
    width: int = 0
    height: int = 0

    @property
    def status_label(self) -> str:
        return "合格 (OK)" if self.is_accepted else "不合格 (NG)"

    @property
    def status_color(self) -> Tuple[int, int, int]:
        # BGR: 緑 (合格) / 赤 (不合格)
        return (0, 220, 100) if self.is_accepted else (50, 50, 240)


class QualityGate:
    """
    フォトグラメトリ用画像の品質ゲートウェイ
    - Laplacian分散法によるモーションブラー・ピンボケの自動検知
    - SIFT/ORBによる特徴点密度およびテクスチャ評価
    - Live View用リアルタイムHUDオーバーレイ描画
    """

    def __init__(self, config: Optional[QualityGateConfig] = None):
        self.config = config or CONFIG.quality
        self._sift = cv2.SIFT_create(nfeatures=2000)
        self._orb = cv2.ORB_create(nfeatures=2000)

    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        OpenCVのラプラシアン (Laplacian) フィルタの分散を用いて画像の鮮鋭度スコアを算出します。
        分散値が高いほど輪郭がシャープであり、低いほどピンボケ・モーションブラーが発生しています。

        Args:
            image: 入力画像 (BGR または グレースケール)

        Returns:
            float: 鮮鋭度スコア (Laplacian Variance)
        """
        if image is None or image.size == 0:
            return 0.0

        # GPU直接ラプラシアン演算 (RTX 5080)
        from solidforge.core.gpu_accelerator import GPU_ACCELERATOR
        return GPU_ACCELERATOR.calculate_blur_score_gpu(image)

    def extract_features(
        self, image: np.ndarray, method: Optional[str] = None
    ) -> Tuple[int, List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        SIFTまたはORBアルゴリズムを用いて画像から特徴点を抽出します。

        Args:
            image: 入力画像 (BGR または グレースケール)
            method: "SIFT" または "ORB" (Noneの場合は設定値を使用)

        Returns:
            Tuple[int, List[KeyPoint], Optional[np.ndarray]]:
                (検出特徴点数, キーポイントリスト, 記述子配列)
        """
        if image is None or image.size == 0:
            return 0, [], None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        extractor_type = (method or self.config.feature_extractor).upper()

        if extractor_type == "SIFT":
            keypoints, descriptors = self._sift.detectAndCompute(gray, None)
        else:
            keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        keypoint_count = len(keypoints) if keypoints is not None else 0
        return keypoint_count, (keypoints or []), descriptors

    def evaluate_image(
        self,
        image_input: Union[np.ndarray, str, Path],
        blur_threshold: Optional[float] = None,
        min_feature_count: Optional[int] = None,
    ) -> QualityEvaluationResult:
        """
        画像が3D再構築キューに投入可能な品質基準を満たしているかを判定します。

        Args:
            image_input: 画像配列 (np.ndarray) または画像ファイルパス (str / Path)
            blur_threshold: 鮮鋭度判定閾値 (Noneの場合は設定値)
            min_feature_count: 最低必要特徴点数 (Noneの場合は設定値)

        Returns:
            QualityEvaluationResult: 判定結果、スコア、不合格理由を含むオブジェクト
        """
        b_thresh = blur_threshold if blur_threshold is not None else self.config.blur_threshold
        f_thresh = min_feature_count if min_feature_count is not None else self.config.min_feature_count

        image_path: Optional[Path] = None
        image: Optional[np.ndarray] = None

        if isinstance(image_input, (str, Path)):
            image_path = Path(image_input)
            if not image_path.exists() or not image_path.is_file() or image_path.stat().st_size == 0:
                return QualityEvaluationResult(
                    is_accepted=False,
                    blur_score=0.0,
                    feature_count=0,
                    rejection_reasons=[f"画像ファイルが存在しないか、空のファイルです: {image_path.name}"],
                    image_path=image_path,
                )
            # 日本語パス安全読み込み
            try:
                buf = np.fromfile(str(image_path), dtype=np.uint8)
                if len(buf) == 0:
                    image = None
                else:
                    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception:
                image = None

            if image is None:
                return QualityEvaluationResult(
                    is_accepted=False,
                    blur_score=0.0,
                    feature_count=0,
                    rejection_reasons=[f"画像ファイルのデコードに失敗しました: {image_path.name}"],
                    image_path=image_path,
                )
        else:
            image = image_input

        if image is None or getattr(image, "size", 0) == 0:
            return QualityEvaluationResult(
                is_accepted=False,
                blur_score=0.0,
                feature_count=0,
                rejection_reasons=["画像データが空または無効です"],
                image_path=image_path,
            )

        height, width = image.shape[:2]

        # 1. 鮮鋭度 (Laplacian Variance) 測定
        blur_score = self.calculate_blur_score(image)

        # 2. 特徴量 (SIFT/ORB) 抽出
        feature_count, keypoints, _ = self.extract_features(image)

        # 3. 合否判定 & 理由収集
        rejection_reasons: List[str] = []

        if blur_score < b_thresh:
            rejection_reasons.append(
                f"ブレ・ピンボケを検知 (鮮鋭度スコア: {blur_score:.1f} < 基準値: {b_thresh:.1f})"
            )

        if feature_count < f_thresh:
            rejection_reasons.append(
                f"特徴点が不足 (特徴点数: {feature_count} < 基準値: {f_thresh})"
            )

        is_accepted = len(rejection_reasons) == 0

        return QualityEvaluationResult(
            is_accepted=is_accepted,
            blur_score=blur_score,
            feature_count=feature_count,
            rejection_reasons=rejection_reasons,
            image_path=image_path,
            keypoints=keypoints,
            width=width,
            height=height,
        )

    def draw_quality_overlay(
        self,
        frame: np.ndarray,
        eval_result: Optional[QualityEvaluationResult] = None,
        draw_grid: bool = True,
        draw_points: bool = True,
        ghost_frame: Optional[np.ndarray] = None,
        ghost_alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Live Viewストリーム用のリアルタイムHUDオーバーレイを描画します。
        - オニオンスキン (Ghost Overlay): 直前撮影フレームを半透明重畳 (60-80%オーバーラップ誘導)
        - SIFT/ORB特徴点の散布ドット表示 (緑: 良好 / 黄: 警告)
        - 3分割構図グリッド
        - 鮮鋭度・特徴点数メーターHUD
        - ステータスバッジ (合格 / 要ピント調整 / 特徴不足)

        Args:
            frame: 入力映像フレーム (BGR)
            eval_result: 直近の品質判定結果 (省略時はリアルタイム評価)
            draw_grid: 構図ガイドグリッドを描画するか
            draw_points: 特徴点ドットを描画するか
            ghost_frame: 直前撮影フレーム (オニオンスキン用)
            ghost_alpha: ゴースト透過度 (0.0 - 1.0)

        Returns:
            np.ndarray: オーバーレイが重畳された映像フレーム
        """
        if frame is None or frame.size == 0:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 0. オニオンスキン (前フレーム半透明重畳)
        if ghost_frame is not None and ghost_alpha > 0.05:
            gh, gw = ghost_frame.shape[:2]
            if (gh, gw) != (h, w):
                resized_ghost = cv2.resize(ghost_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_ghost = ghost_frame
            # アルファブレンド
            cv2.addWeighted(resized_ghost, ghost_alpha, canvas, 1.0 - ghost_alpha, 0, canvas)
            # ゴーストインジケータ表示
            cv2.putText(
                canvas,
                f"GHOST OVERLAY: {int(ghost_alpha*100)}% (Overlap 60-80%)",
                (w - 340, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 229, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        if eval_result is None:
            eval_result = self.evaluate_image(frame)

        # 1. 3分割構図グリッド (半透明)
        if draw_grid:
            grid_color = (80, 80, 80)
            # 垂直線
            cv2.line(canvas, (w // 3, 0), (w // 3, h), grid_color, 1)
            cv2.line(canvas, (2 * w // 3, 0), (2 * w // 3, h), grid_color, 1)
            # 水平線
            cv2.line(canvas, (0, h // 3), (w, h // 3), grid_color, 1)
            cv2.line(canvas, (0, 2 * h // 3), (w, 2 * h // 3), grid_color, 1)

        # 2. 特徴点ドット描画 (最大300点サンプリングで描画負荷軽減)
        if draw_points and eval_result.keypoints:
            pt_color = (0, 230, 120) if eval_result.is_accepted else (0, 200, 255)
            pts = eval_result.keypoints[:300]
            for kp in pts:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(canvas, (x, y), 2, pt_color, -1, lineType=cv2.LINE_AA)

        # 3. HUDトップパネル (半透明背景)
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (10, 10), (320, 120), (20, 24, 30), -1)
        cv2.addWeighted(hud_bg, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (320, 120), (60, 70, 85), 1)

        # 4. HUDテキスト情報
        # ステータス
        status_text = "STATUS: PASS (READY)" if eval_result.is_accepted else "STATUS: WARNING (CHECK FOCUS)"
        status_bgr = (0, 230, 120) if eval_result.is_accepted else (50, 70, 255)
        cv2.putText(
            canvas,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_bgr,
            2,
            lineType=cv2.LINE_AA,
        )

        # 鮮鋭度スコア
        blur_ratio = min(1.0, eval_result.blur_score / (self.config.blur_threshold * 2.0))
        blur_text = f"Sharpness: {eval_result.blur_score:.1f} (Min: {self.config.blur_threshold:.0f})"
        cv2.putText(
            canvas,
            blur_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        # 鮮鋭度プログレスバー
        bar_x, bar_y, bar_w, bar_h = 20, 68, 280, 6
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 50), -1)
        active_w = int(bar_w * blur_ratio)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + active_w, bar_y + bar_h), status_bgr, -1)

        # 特徴点数
        feat_text = f"Features : {eval_result.feature_count} pts (Min: {self.config.min_feature_count})"
        cv2.putText(
            canvas,
            feat_text,
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )
        feat_ratio = min(1.0, eval_result.feature_count / (self.config.min_feature_count * 2.0))
        f_bar_y = 103
        cv2.rectangle(canvas, (bar_x, f_bar_y), (bar_x + bar_w, f_bar_y + bar_h), (40, 40, 50), -1)
        f_active_w = int(bar_w * feat_ratio)
        cv2.rectangle(canvas, (bar_x, f_bar_y), (bar_x + f_active_w, f_bar_y + bar_h), (0, 200, 255), -1)

        return canvas


# シングルトンインスタンス
QUALITY_GATE = QualityGate()
