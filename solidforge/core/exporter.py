"""
SolidForge 3D - 3D Model Exporter & Slicer Pre-Flight
STL / OBJ / PLY / GLTF エクスポートおよびスライサー投入前検証
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import trimesh

from solidforge.core.post_processor import PrintabilityReport


class ModelExporter:
    """
    3Dメッシュのエクスポートマネージャー
    - STL: 3Dプリンタ標準 (バイナリSTL)
    - OBJ: Wavefront OBJ (テクスチャ・マテリアル付随)
    - PLY: Stanford Polygon File (頂点カラー・点群)
    - GLTF: Web/AR/3Dプレビュー用
    """

    SUPPORTED_FORMATS = {
        "stl": "STL (.stl) - 3Dプリント推奨 (バイナリ)",
        "obj": "OBJ (.obj) - テクスチャ付きメッシュ",
        "ply": "PLY (.ply) - 高密度点群・頂点カラー",
        "gltf": "GLTF (.gltf) - Web/AR用標準フォーマット",
    }

    @staticmethod
    def export_mesh(
        mesh: trimesh.Trimesh,
        destination_path: Union[str, Path],
        file_type: Optional[str] = None,
    ) -> Path:
        """
        メッシュを指定パスに出力します。

        Args:
            mesh: Trimeshオブジェクト
            destination_path: 出力先ファイルパス
            file_type: "stl", "obj", "ply", "gltf" (Noneの場合は拡張子から判定)

        Returns:
            Path: 保存先パス
        """
        dest = Path(destination_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        ext = (file_type or dest.suffix.replace(".", "")).lower()

        if ext == "stl":
            # 3Dスライサー向けにバイナリSTLとしてエクスポート
            mesh.export(str(dest), file_type="stl")
        elif ext == "obj":
            mesh.export(str(dest), file_type="obj")
        elif ext == "ply":
            mesh.export(str(dest), file_type="ply")
        elif ext in ("gltf", "glb"):
            mesh.export(str(dest), file_type=ext)
        else:
            raise ValueError(f"未対応の拡張子形式です: {ext}")

        return dest

    @staticmethod
    def launch_in_orcaslicer(stl_path: Union[str, Path]) -> bool:
        """
        生成されたSTLファイルをOrcaSlicer (またはPrusaSlicer / Bambu Studio) で自動起動し、
        ビルドプレート中央に配置して開きます。

        Args:
            stl_path: STLファイルのパス

        Returns:
            bool: 起動成功の可否
        """
        import os
        import subprocess
        from pathlib import Path

        stl_file = Path(stl_path).resolve()
        if not stl_file.exists():
            return False

        # OrcaSlicer / Bambu Studio / PrusaSlicer の標準インストールパス候補
        candidate_paths = [
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "OrcaSlicer" / "orca-slicer.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "OrcaSlicer" / "orca-slicer.exe",
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Bambu Studio" / "bambu-studio.exe",
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Prusa3D" / "PrusaSlicer" / "prusa-slicer.exe",
            Path("C:\\Program Files\\OrcaSlicer\\orca-slicer.exe"),
            Path("C:\\Program Files\\Bambu Lab\\Bambu Studio\\bambu-studio.exe"),
        ]

        # 1. 検出された専用スライサーを実行
        for exe_path in candidate_paths:
            if exe_path.exists():
                try:
                    subprocess.Popen([str(exe_path), str(stl_file)])
                    return True
                except Exception:
                    pass

        # 2. システムの既定3Dスライサー (Windows Association) でフォールバック起動
        try:
            os.startfile(str(stl_file))
            return True
        except Exception:
            return False


EXPORTER = ModelExporter()
