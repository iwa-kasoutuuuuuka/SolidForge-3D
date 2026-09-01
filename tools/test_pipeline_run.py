# -*- coding: utf-8 -*-
"""
End-to-end pipeline verification test for SolidForge 3D
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solidforge.core.pipeline import PIPELINE
from solidforge.config import CONFIG

# Force AI Enhancer & Background Removal ON
CONFIG.ai_enhancement.enable_ai_enhancer = True
CONFIG.ai_enhancement.enable_ai_background_removal = True

images = sorted(list(Path(r"E:\SolidForge 3D\workspace\reconstruction_1788246891\images").glob("*.jpg")))
print(f"[TEST] Starting full pipeline verification on {len(images)} images...", flush=True)

PIPELINE.log_emitted.connect(lambda msg: print(f"[LOG] {msg}", flush=True))
PIPELINE.progress_updated.connect(lambda p, s: print(f"[{p}%] {s}", flush=True))

finished_result = {}

def on_finished(success, mesh_path, report):
    finished_result["success"] = success
    finished_result["path"] = mesh_path
    finished_result["report"] = report
    print(f"\n[TEST COMPLETE] Success: {success}, Path: {mesh_path}", flush=True)
    if report:
        print(f"[TEST REPORT]\n{report.summary_text_ja}", flush=True)

PIPELINE.reconstruction_finished.connect(on_finished)

# Run pipeline synchronously in main thread for testing
PIPELINE._run_pipeline(images, output_format="stl")

if not finished_result.get("success"):
    print("[TEST FAILED] Pipeline did not finish successfully.")
    sys.exit(1)
else:
    print("[TEST PASSED] Pipeline finished 100% successfully!")
    sys.exit(0)
