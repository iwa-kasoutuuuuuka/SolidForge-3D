# -*- coding: utf-8 -*-
"""
SolidForge 3D - AI 背景自動除去 & 被写体分離エンジン (AI Object Segmenter)
U2Net / ISNet (rembg + ONNX Runtime) および OpenCV による高精度前景マスク生成
"""

from pathlib import Path
from typing import List, Optional, Tuple, Callable
import os
import cv2
import numpy as np
from PIL import Image

try:
    import rembg
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


class AISegmenter:
    """AI を用いて撮影画像から背景を自動除去し、被写体マスクを生成するエンジン"""

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self._session = None

    def _get_session(self):
        """rembg セッションを遅延初期化"""
        if self._session is None and REMBG_AVAILABLE:
            try:
                self._session = rembg.new_session(self.model_name)
            except Exception as e:
                print(f"[AISegmenter] セッション初期化警告: {e}")
                self._session = None
        return self._session

    def generate_mask(
        self,
        image_path: Path,
        output_mask_path: Path,
        margin_pixels: int = 5,
    ) -> bool:
        """
        単一画像から被写体マスク（255: 被写体, 0: 背景）を生成し、保存する
        COLMAP & OpenMVS 互換のグレースケールPNG
        """
        try:
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            session = self._get_session()

            if session is not None:
                # rembg (U2Net) による高精度セグメンテーション
                pil_img = Image.open(str(image_path))
                mask_pil = rembg.remove(pil_img, session=session, only_mask=True)
                mask_np = np.array(mask_pil)
            else:
                # フォールバック: OpenCV GrabCut + Otsu 閾値処理
                mask_np = self._fallback_grabcut_mask(image_path)

            # 2値化
            _, binary_mask = cv2.threshold(mask_np, 128, 255, cv2.THRESH_BINARY)

            # 被写体のエッジが削れないよう、指定ピクセルだけモルフォロジー膨張
            if margin_pixels > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (margin_pixels * 2 + 1, margin_pixels * 2 + 1)
                )
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            # 保存
            cv2.imwrite(str(output_mask_path), binary_mask)
            return True
        except Exception as e:
            print(f"[AISegmenter] マスク生成エラー ({image_path.name}): {e}")
            # エラー時は全領域有効マスク (全白) を出力してパイプライン停止を防ぐ
            try:
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    white_mask = np.full((h, w), 255, dtype=np.uint8)
                    cv2.imwrite(str(output_mask_path), white_mask)
                    return True
            except Exception:
                pass
            return False

    def process_batch(
        self,
        image_paths: List[Path],
        masks_dir: Path,
        margin_pixels: int = 5,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        max_workers: Optional[int] = None,
    ) -> List[Path]:
        """
        画像リストをマルチスレッド並列処理し、COLMAP用マスク (<name>.png) および OpenMVS用マスク (<name>.mask.png) を高速生成
        """
        import shutil
        from concurrent.futures import ThreadPoolExecutor, as_completed

        masks_dir.mkdir(parents=True, exist_ok=True)
        generated_masks: List[Path] = []
        total = len(image_paths)
        if total == 0:
            return []

        workers = max_workers or min(8, os.cpu_count() or 4)

        def _process_single(item: Tuple[int, Path]) -> Tuple[int, Path, bool]:
            idx, img_path = item
            colmap_mask_path = masks_dir / f"{img_path.name}.png"
            success = self.generate_mask(img_path, colmap_mask_path, margin_pixels=margin_pixels)
            if success:
                openmvs_mask_path = masks_dir / f"{img_path.stem}.mask.png"
                if not openmvs_mask_path.exists():
                    try:
                        shutil.copyfile(str(colmap_mask_path), str(openmvs_mask_path))
                    except Exception:
                        pass
            return idx, colmap_mask_path, success

        completed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_single, (idx, img_p)) for idx, img_p in enumerate(image_paths)]
            for future in as_completed(futures):
                idx, colmap_mask_path, success = future.result()
                completed_count += 1
                if success:
                    generated_masks.append(colmap_mask_path)
                if progress_cb and total > 0:
                    pct = int(completed_count / total * 100)
                    progress_cb(pct, f"AI 背景マスク並列生成中 ({completed_count}/{total}) [{workers} Threads]...")

        return generated_masks

    def _fallback_grabcut_mask(self, image_path: Path) -> np.ndarray:
        """OpenCV GrabCut による安全フォールバックセグメンテーション"""
        img = cv2.imread(str(image_path))
        if img is None:
            return np.full((100, 100), 255, dtype=np.uint8)

        h, w = img.shape[:2]
        # 中央 80% を前景候補矩形として初期化
        rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            final_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
            return final_mask
        except Exception:
            return np.full((h, w), 255, dtype=np.uint8)


# シングルトンインスタンス
AI_SEGMENTER = AISegmenter()
