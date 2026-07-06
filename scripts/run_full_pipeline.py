#!/usr/bin/env python3
"""CLI: Run the entire ALPR dataset engineering pipeline end-to-end.

Order: inspection -> EDA -> quality -> harmonization -> preprocessing ->
statistics -> split. Each stage is also runnable independently via its own
script in this directory, which is useful while iterating on one stage.

Usage:
    python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STAGES = [
    "run_inspection.py",
    "run_eda.py",
    "run_quality_check.py",
    "run_harmonization.py",
    "run_preprocessing.py",
    "run_statistics.py",
    "run_split.py",
]


def main() -> None:
    for stage in STAGES:
        stage_path = SCRIPT_DIR / stage
        print(f"\n{'=' * 70}\nRunning stage: {stage}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(stage_path)], cwd=SCRIPT_DIR.parent)
        if result.returncode != 0:
            print(f"Stage '{stage}' failed with exit code {result.returncode}. Stopping pipeline.")
            sys.exit(result.returncode)
    print("\nFull pipeline complete. See reports/ for all generated outputs.")


if __name__ == "__main__":
    main()
