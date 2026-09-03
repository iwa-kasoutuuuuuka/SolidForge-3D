# -*- coding: utf-8 -*-
"""
SolidForge 3D - Project Bundle Manager (.sforge プロジェクト保存 & 復元エンジン)
撮影写真、設定値、3Dモデル、診断レポートを単一の .sforge 圧縮パッケージとして永続化・復元
"""

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zipfile

from solidforge.config import CONFIG, AppConfig
from solidforge.core.post_processor import PrintabilityReport


class ProjectManager:
    """SolidForge 3D プロジェクトの保存および復元マネージャ"""

    @staticmethod
    def export_project(
        dest_file_path: Path,
        project_name: str,
        image_paths: List[Path],
        model_path: Optional[Path] = None,
        report: Optional[PrintabilityReport] = None,
        config: Optional[AppConfig] = None,
    ) -> bool:
        """
        現在のスキャンプロジェクトを .sforge (ZIPアーカイブ) として保存
        """
        try:
            dest_file_path = Path(dest_file_path)
            if dest_file_path.suffix.lower() != ".sforge":
                dest_file_path = dest_file_path.with_suffix(".sforge")

            dest_file_path.parent.mkdir(parents=True, exist_ok=True)

            # メタデータ JSON 構築
            metadata = {
                "project_name": project_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "app_version": "1.0.0",
                "image_count": len(image_paths),
                "has_model": model_path is not None and model_path.exists(),
                "model_name": model_path.name if model_path and model_path.exists() else None,
            }

            if report is not None:
                metadata["report"] = {
                    "is_watertight": report.is_watertight,
                    "dimensions_mm": list(report.dimensions_mm),
                    "volume_cm3": report.volume_cm3,
                    "surface_area_cm2": report.surface_area_cm2,
                    "face_count": report.face_count,
                    "vertex_count": report.vertex_count,
                    "overhang_area_ratio_pct": report.overhang_area_ratio_pct,
                }

            with zipfile.ZipFile(dest_file_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # 1. メタデータ
                zf.writestr("project.json", json.dumps(metadata, ensure_ascii=False, indent=2))

                # 2. 撮影画像
                for img_p in image_paths:
                    p = Path(img_p)
                    if p.exists() and p.is_file():
                        zf.write(str(p), arcname=f"images/{p.name}")

                # 3. 3Dモデル & 関連ファイル
                if model_path and model_path.exists():
                    zf.write(str(model_path), arcname=f"model/{model_path.name}")
                    # 同一フォルダのテクスチャやmtlも同梱
                    parent_dir = model_path.parent
                    for related in parent_dir.glob("*.mtl"):
                        zf.write(str(related), arcname=f"model/{related.name}")
                    for related in parent_dir.glob("*.jpg"):
                        if "map_Kd" in related.name:
                            zf.write(str(related), arcname=f"model/{related.name}")
                    for related in parent_dir.glob("*.png"):
                        if "map_Kd" in related.name:
                            zf.write(str(related), arcname=f"model/{related.name}")

            return True
        except Exception as e:
            print(f"[ProjectManager] プロジェクト保存エラー: {e}")
            return False

    @staticmethod
    def import_project(
        sforge_path: Path,
        extract_root_dir: Path,
    ) -> Tuple[bool, dict, List[Path], Optional[Path]]:
        """
        .sforge ファイルからプロジェクトを解凍・復元

        Returns:
            Tuple[bool, dict, List[Path], Optional[Path]]:
            (成功フラグ, メタデータ辞書, 復元画像パス一覧, 復元モデルパス)
        """
        try:
            sforge_path = Path(sforge_path)
            if not sforge_path.exists():
                return False, {}, [], None

            target_dir = extract_root_dir / sforge_path.stem
            target_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(sforge_path, "r") as zf:
                zf.extractall(str(target_dir))

            meta_file = target_dir / "project.json"
            metadata = {}
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            images_dir = target_dir / "images"
            extracted_images = sorted(list(images_dir.glob("*.*"))) if images_dir.exists() else []

            model_dir = target_dir / "model"
            extracted_model = None
            if model_dir.exists():
                models = list(model_dir.glob("*.stl")) + list(model_dir.glob("*.obj")) + list(model_dir.glob("*.ply"))
                if models:
                    extracted_model = models[0]

            return True, metadata, extracted_images, extracted_model
        except Exception as e:
            print(f"[ProjectManager] プロジェクト復元エラー: {e}")
            return False, {}, [], None


PROJECT_MANAGER = ProjectManager()
