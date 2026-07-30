"""GET /api/datasets — dataset info from raw data directory."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.get("/datasets")
def list_datasets():
    raw_dir = PROJECT_ROOT / "data" / "raw"
    datasets = []

    if raw_dir.is_dir():
        for d in sorted(raw_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue

            image_count = 0
            annotation_count = 0
            for f in d.rglob("*"):
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    image_count += 1
                elif ext in (".txt", ".xml", ".json", ".csv"):
                    annotation_count += 1

            # Detect format from annotations
            fmt = "unknown"
            for _ in d.rglob("*.xml"):
                fmt = "voc_xml"
                break
            if fmt == "unknown":
                for f in d.rglob("*.txt"):
                    content = f.read_text(encoding="utf-8", errors="ignore").strip()
                    if content and len(content.split()) == 5:
                        fmt = "yolo"
                        break
            if fmt == "unknown":
                for _ in d.rglob("*.json"):
                    fmt = "coco_json"
                    break
            if fmt == "unknown":
                for _ in d.rglob("*.csv"):
                    fmt = "csv"
                    break

            datasets.append({
                "name": d.name,
                "format": fmt,
                "image_count": image_count,
                "annotation_count": annotation_count,
                "issues": 0,
            })

    return {"datasets": datasets}
