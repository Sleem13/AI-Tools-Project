"""GET /api/pipeline/status — pipeline stages and last run info."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

STAGES = [
    {"name": "Inspection", "script": "scripts/run_inspection.py"},
    {"name": "EDA Figures", "script": "scripts/run_eda.py"},
    {"name": "Quality Check", "script": "scripts/run_quality_check.py"},
    {"name": "Harmonization", "script": "scripts/run_harmonization.py"},
    {"name": "Preprocessing", "script": "scripts/run_preprocessing.py"},
    {"name": "Statistics", "script": "scripts/run_statistics.py"},
    {"name": "Train/Val/Test Split", "script": "scripts/run_split.py"},
]


@router.get("/pipeline/status")
def pipeline_status():
    reports_dir = PROJECT_ROOT / "reports"
    datasets_dir = PROJECT_ROOT / "data" / "raw"

    # Detect datasets
    dataset_names = []
    if datasets_dir.is_dir():
        for d in sorted(datasets_dir.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                dataset_names.append(d.name)

    # Count images
    total_images = 0
    for name in dataset_names:
        ds_path = datasets_dir / name
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            total_images += len(list(ds_path.rglob(ext)))

    # Check which stages have output
    report_dirs = {
        "eda": reports_dir / "eda",
        "quality": reports_dir / "quality",
        "harmonization": reports_dir / "harmonization",
        "splits": reports_dir / "splits",
    }

    stages = []
    for i, stage in enumerate(STAGES):
        status = "pending"
        last_run = None

        # Check if relevant output exists
        if i == 0:
            for name in dataset_names:
                if (datasets_dir / name).is_dir():
                    count = sum(1 for _ in (datasets_dir / name).rglob("*") if _.is_file())
                    if count > 0:
                        status = "completed"
                        break
        elif i <= 2 and report_dirs["eda"].is_dir():
            status = "completed"
        elif i == 3 and report_dirs["harmonization"].is_dir():
            status = "completed"
        elif i == 4 and (PROJECT_ROOT / "data" / "processed").is_dir():
            status = "completed"
        elif i == 5 and report_dirs.get("eda") and any((report_dirs["eda"]).glob("dataset_statistics.*")):
            status = "completed"
        elif i == 6 and report_dirs["splits"].is_dir():
            status = "completed"

        # Find last modified report for last_run estimate
        if status == "completed":
            for rdir in report_dirs.values():
                if rdir.is_dir():
                    files = list(rdir.rglob("*"))
                    if files:
                        newest = max(f.stat().st_mtime for f in files if f.is_file())
                        from datetime import datetime

                        last_run = datetime.fromtimestamp(newest).isoformat()
                        break

        stages.append({
            **stage,
            "status": status,
            "last_run": last_run,
        })

    return {
        "stages": stages,
        "total_images": total_images,
        "datasets": dataset_names,
    }
