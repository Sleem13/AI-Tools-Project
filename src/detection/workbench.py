"""Business logic for the Master Plate dataset training workbench."""

from __future__ import annotations

import csv
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
SPLITS = ("train", "val", "test")
SUPPORTED_PYTHON_VERSIONS = frozenset({(3, 11), (3, 12)})

METRIC_COLUMNS = {
    "epoch": "epoch",
    "train_box_loss": "train/box_loss",
    "train_cls_loss": "train/cls_loss",
    "train_dfl_loss": "train/dfl_loss",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "val_box_loss": "val/box_loss",
    "val_cls_loss": "val/cls_loss",
    "val_dfl_loss": "val/dfl_loss",
}


def resolve_master_dataset_root(project_root: Path) -> Path:
    """Resolve the dataset root from the environment or repository convention."""
    configured = os.environ.get("MASTER_PLATE_DATASET_ROOT")
    if configured:
        path = Path(configured).expanduser()
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return (project_root / "data" / "processed" / "Master_Plate_Dataset").resolve()


def inspect_runtime() -> dict[str, Any]:
    """Return the exact model runtime visible to the API process."""
    import torch
    import ultralytics

    cuda_available = torch.cuda.is_available()
    runtime: dict[str, Any] = {
        "python": platform.python_version(),
        "python_supported": sys.version_info[:2] in SUPPORTED_PYTHON_VERSIONS,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count(),
        "gpu_name": None,
        "gpu_memory_gib": None,
        "compute_capability": None,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        runtime.update(
            gpu_name=properties.name,
            gpu_memory_gib=round(properties.total_memory / 1024**3, 1),
            compute_capability=f"{properties.major}.{properties.minor}",
        )
    runtime["ready"] = bool(runtime["python_supported"] and cuda_available)
    return runtime


def inspect_dataset(dataset_root: Path) -> dict[str, Any]:
    """Summarize the Ultralytics dataset without decoding every image."""
    split_stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        split_stats[split] = {
            "images": sum(1 for _ in iter_images(image_dir)),
            "labels": sum(1 for path in label_dir.rglob("*.txt") if path.is_file()) if label_dir.is_dir() else 0,
        }
    data_yaml = dataset_root / "data.yaml"
    return {
        "root": str(dataset_root),
        "exists": dataset_root.is_dir(),
        "data_yaml": str(data_yaml),
        "data_yaml_exists": data_yaml.is_file(),
        "splits": split_stats,
        "total_images": sum(item["images"] for item in split_stats.values()),
        "total_labels": sum(item["labels"] for item in split_stats.values()),
        "ready": data_yaml.is_file() and split_stats["train"]["images"] > 0 and split_stats["val"]["images"] > 0,
    }


def iter_images(image_dir: Path):
    if not image_dir.is_dir():
        return
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def sample_images(dataset_root: Path, split: str, count: int, seed: int) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    images = list(iter_images(dataset_root / "images" / split))
    if not images:
        return []
    return random.Random(seed).sample(images, min(count, len(images)))


def resolve_dataset_image(dataset_root: Path, split: str, relative_path: str) -> Path:
    """Resolve a browser-requested image while preventing path traversal."""
    if split not in SPLITS:
        raise ValueError(f"Unsupported split: {split}")
    image_root = (dataset_root / "images" / split).resolve()
    candidate = (image_root / relative_path).resolve()
    if candidate != image_root and image_root not in candidate.parents:
        raise ValueError("Image path escapes the selected dataset split")
    if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise FileNotFoundError(relative_path)
    return candidate


def relative_image_path(dataset_root: Path, split: str, image_path: Path) -> str:
    return image_path.relative_to(dataset_root / "images" / split).as_posix()


def natural_high_resolution_plate(cropped_image: np.ndarray, scale_factor: int = 4) -> np.ndarray:
    """Replicate the notebook's natural Lanczos plate enhancement."""
    if cropped_image.size == 0:
        raise ValueError("cropped_image must not be empty")
    if not 1 <= scale_factor <= 8:
        raise ValueError("scale_factor must be between 1 and 8")
    height, width = cropped_image.shape[:2]
    return cv2.resize(
        cropped_image,
        (width * scale_factor, height * scale_factor),
        interpolation=cv2.INTER_LANCZOS4,
    )


def read_training_history(results_csv: Path) -> dict[str, Any]:
    """Parse Ultralytics ``results.csv`` into UI-friendly metric names."""
    if not results_csv.is_file():
        return {"history": [], "latest": None}
    history: list[dict[str, float | int | None]] = []
    with results_csv.open(encoding="utf-8", newline="") as csv_file:
        for raw_row in csv.DictReader(csv_file):
            row = {key.strip(): value.strip() for key, value in raw_row.items() if key is not None and value is not None}
            parsed: dict[str, float | int | None] = {}
            for output_name, source_name in METRIC_COLUMNS.items():
                value = _number(row.get(source_name))
                parsed[output_name] = int(value) + 1 if output_name == "epoch" and value is not None else value
            history.append(parsed)
    return {"history": history, "latest": history[-1] if history else None}


def discover_latest_run(output_root: Path) -> Path | None:
    if not output_root.is_dir():
        return None
    result_files = [path for path in output_root.rglob("results.csv") if path.is_file()]
    return max(result_files, key=lambda path: path.stat().st_mtime).parent if result_files else None


def discover_best_completed_run(output_root: Path) -> Path | None:
    """Select the strongest finalized run for production inference.

    A checkpoint is written after every epoch, so ``best.pt`` alone does not
    prove that training finished. Ultralytics writes ``results.png`` during its
    final evaluation; requiring it prevents interrupted/OOM runs from silently
    replacing a stable production model. Among finalized runs, prefer the best
    observed mAP50-95 and use modification time only as a tie-breaker.
    """
    if not output_root.is_dir():
        return None

    candidates = []
    for results_path in output_root.rglob("results.csv"):
        run_dir = results_path.parent
        if not results_path.is_file():
            continue
        if not (run_dir / "weights" / "best.pt").is_file() or not (run_dir / "results.png").is_file():
            continue
        history = read_training_history(results_path).get("history", [])
        scores = [row["map50_95"] for row in history if row.get("map50_95") is not None]
        if scores:
            candidates.append((max(scores), results_path.stat().st_mtime, run_dir))
    return max(candidates, default=(None, None, None))[-1]


def training_artifacts(run_dir: Path | None) -> list[dict[str, Any]]:
    definitions = (
        ("results.png", "Training curves"),
        ("confusion_matrix.png", "Confusion matrix"),
        ("confusion_matrix_normalized.png", "Normalized confusion matrix"),
        ("labels.jpg", "Dataset labels"),
        ("val_batch0_pred.jpg", "Validation predictions"),
    )
    return [
        {"name": name, "label": label, "exists": bool(run_dir and (run_dir / name).is_file())}
        for name, label in definitions
    ]


def load_reference_run(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None
