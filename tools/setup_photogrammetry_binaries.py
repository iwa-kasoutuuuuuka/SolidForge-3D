# -*- coding: utf-8 -*-
"""
SolidForge 3D - Photogrammetry Engine Automated Setup
Downloads and configures official COLMAP (Windows CUDA) and OpenMVS (Windows x64)
"""

import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = BASE_DIR / "bin"
COLMAP_DIR = BIN_DIR / "colmap"
OPENMVS_DIR = BIN_DIR / "openmvs"

COLMAP_URL = "https://github.com/colmap/colmap/releases/download/3.9.1/COLMAP-3.9.1-windows-cuda.zip"
OPENMVS_URL = "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_Windows_x64.zip"


def download_with_progress(url: str, dest_path: Path, label: str):
    """プログレス表示付きダウンローダー"""
    print(f"[{label}] ダウンロード開始: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1MB
        
        while True:
            chunk = resp.read(block_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100.0
                mb_cur = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r[{label}] 進捗: {percent:.1f}% ({mb_cur:.1f} MB / {mb_tot:.1f} MB)")
                sys.stdout.flush()
    print(f"\n[{label}] ダウンロード完了 -> {dest_path.name}")


def extract_zip(zip_path: Path, target_dir: Path, label: str):
    """ZIP解凍"""
    print(f"[{label}] 解凍中 -> {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)
    print(f"[{label}] 解凍完了！")


def setup_colmap():
    """COLMAP (Windows CUDA) のダウンロード & 配置"""
    COLMAP_DIR.mkdir(parents=True, exist_ok=True)
    
    # 既存チェック
    if (COLMAP_DIR / "colmap.bat").exists() or (COLMAP_DIR / "COLMAP.bat").exists() or (COLMAP_DIR / "colmap.exe").exists():
        print("[COLMAP] 既にインストール済みです。スキップします。")
        return

    zip_file = BIN_DIR / "colmap_cuda.zip"
    try:
        download_with_progress(COLMAP_URL, zip_file, "COLMAP CUDA")
        extract_zip(zip_file, COLMAP_DIR, "COLMAP")
        
        # 解凍後のネストフォルダを整理
        nested = list(COLMAP_DIR.glob("COLMAP-*"))
        if nested and nested[0].is_dir():
            for item in nested[0].iterdir():
                shutil.move(str(item), str(COLMAP_DIR))
            nested[0].rmdir()
            
        if zip_file.exists():
            zip_file.unlink()
        print("[COLMAP] セットアップが完了しました！")
    except Exception as e:
        print(f"[COLMAP エラー] {e}")


def setup_openmvs():
    """OpenMVS (Windows x64) のダウンロード & 配置"""
    OPENMVS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 既存チェック
    if (OPENMVS_DIR / "InterfaceCOLMAP.exe").exists() or (OPENMVS_DIR / "InterfaceCOLMAP").exists():
        print("[OpenMVS] 既にインストール済みです。スキップします。")
        return

    zip_file = BIN_DIR / "openmvs_win64.zip"
    try:
        download_with_progress(OPENMVS_URL, zip_file, "OpenMVS Win64")
        extract_zip(zip_file, OPENMVS_DIR, "OpenMVS")
        
        # 解凍後のネストフォルダを整理
        nested = list(OPENMVS_DIR.glob("OpenMVS*")) + list(OPENMVS_DIR.glob("openMVS*")) + list(OPENMVS_DIR.glob("bin*"))
        for d in nested:
            if d.is_dir():
                for item in d.rglob("*.exe"):
                    dest = OPENMVS_DIR / item.name
                    if not dest.exists():
                        shutil.copy2(item, dest)
                for item in d.rglob("*.dll"):
                    dest = OPENMVS_DIR / item.name
                    if not dest.exists():
                        shutil.copy2(item, dest)
                        
        if zip_file.exists():
            zip_file.unlink()
        print("[OpenMVS] セットアップが完了しました！")
    except Exception as e:
        print(f"[OpenMVS エラー] {e}")


if __name__ == "__main__":
    print("========================================================")
    print("  ⚡ SolidForge 3D - 3D再構築エンジン自動セットアップ")
    print("========================================================")
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    setup_colmap()
    setup_openmvs()
    print("\n[SUCCESS] すべてのエンジンのセットアップが完了しました！")
