"""Master Plate dataset training, artifact, and evaluation APIs."""

from __future__ import annotations

import base64
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import cv2
import torch
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.detection.workbench import (
    discover_latest_run,
    inspect_dataset,
    inspect_runtime,
    load_reference_run,
    natural_high_resolution_plate,
    read_training_history,
    relative_image_path,
    resolve_dataset_image,
    resolve_master_dataset_root,
    sample_images,
    training_artifacts,
)

from api.training_jobs import training_job_manager

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "models" / "detection"
REFERENCE_RUN_PATH = PROJECT_ROOT / "reports" / "training" / "master_plate_notebook_run.json"
ARTIFACT_NAMES = {
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "labels.jpg",
    "val_batch0_pred.jpg",
}

_evaluation_model = None
_evaluation_model_path: Path | None = None
_evaluation_model_lock = threading.Lock()


class StartTrainingRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=1000)
    imgsz: int = Field(default=640, ge=320, le=2048, multiple_of=32)
    batch: int = Field(default=16, ge=1, le=256)
    device: str = Field(default="0", pattern=r"^(cpu|\d+)$")


class EvaluationRequest(BaseModel):
    count: int = Field(default=3, ge=1, le=5)
    confidence: float = Field(default=0.4, ge=0.01, le=1.0)
    scale_factor: int = Field(default=4, ge=1, le=8)
    seed: int = 42


@router.get("/training/workbench")
def training_workbench() -> dict[str, Any]:
    dataset_root = resolve_master_dataset_root(PROJECT_ROOT)
    run_dir = _active_or_latest_run()
    metrics = read_training_history(run_dir / "results.csv") if run_dir else {"history": [], "latest": None}
    job = training_job_manager.snapshot()
    job["metrics"] = metrics
    return {
        "runtime": inspect_runtime(),
        "dataset": inspect_dataset(dataset_root),
        "job": job,
        "run": _run_payload(run_dir, metrics),
        "reference_run": load_reference_run(REFERENCE_RUN_PATH),
        "defaults": {"epochs": 50, "imgsz": 640, "batch": 16, "device": "0"},
    }


@router.get("/training/status")
def training_status() -> dict[str, Any]:
    workbench = training_workbench()
    job = workbench["job"]
    latest = job["metrics"].get("latest") or workbench["run"].get("latest") or {}
    epoch = int(latest.get("epoch") or 0)
    reference_epochs = ((workbench.get("reference_run") or {}).get("training") or {}).get("epochs", 50)
    return {
        "training": [
            {
                "model": "YOLO11 Plate Detection (Master Plate)",
                "status": job["status"] if job["status"] != "idle" else ("completed" if workbench["run"]["best_model_exists"] else "idle"),
                "epoch": epoch,
                "total_epochs": int(job.get("epochs") or reference_epochs),
                "loss": latest.get("train_box_loss") or 0,
                "metrics": {key: latest[key] for key in ("precision", "recall", "map50", "map50_95") if latest.get(key) is not None},
            }
        ],
        "job": job,
    }


@router.post("/training/start")
def start_training(request: StartTrainingRequest) -> dict[str, Any]:
    dataset_root = resolve_master_dataset_root(PROJECT_ROOT)
    dataset = inspect_dataset(dataset_root)
    if not dataset["ready"]:
        raise HTTPException(status_code=409, detail=f"Master Plate dataset is not ready at {dataset_root}")
    if request.device != "cpu" and not torch.cuda.is_available():
        raise HTTPException(status_code=409, detail="CUDA is unavailable to the API process; use the Python 3.12 GPU environment")

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"master_plate_yolo11_{timestamp}"
    run_dir = OUTPUT_ROOT / run_name
    log_path = PROJECT_ROOT / "reports" / "training" / f"{run_name}.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_detection.py"),
        "--stage",
        "plate",
        "--config",
        str(PROJECT_ROOT / "configs" / "model" / "master_plate_detection.yaml"),
        "--data",
        str(dataset_root / "data.yaml"),
        "--epochs",
        str(request.epochs),
        "--imgsz",
        str(request.imgsz),
        "--batch",
        str(request.batch),
        "--device",
        request.device,
        "--output",
        str(OUTPUT_ROOT),
        "--name",
        run_name,
    ]
    try:
        job = training_job_manager.start(
            command=command,
            cwd=PROJECT_ROOT,
            run_name=run_name,
            run_dir=run_dir,
            log_path=log_path,
            epochs=request.epochs,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"job": job}


@router.get("/training/dataset/samples")
def dataset_samples(
    split: Annotated[str, Query(pattern=r"^(train|val|test)$")] = "train",
    count: Annotated[int, Query(ge=1, le=12)] = 4,
    seed: int = 42,
) -> dict[str, Any]:
    dataset_root = resolve_master_dataset_root(PROJECT_ROOT)
    images = sample_images(dataset_root, split, count, seed)
    items = []
    for image_path in images:
        relative_path = relative_image_path(dataset_root, split, image_path)
        label_path = dataset_root / "labels" / split / Path(relative_path).with_suffix(".txt")
        items.append(
            {
                "name": image_path.name,
                "path": relative_path,
                "label_exists": label_path.is_file(),
                "url": f"/api/training/dataset/image?split={split}&path={quote(relative_path)}",
            }
        )
    return {"split": split, "count": len(items), "images": items}


@router.get("/training/dataset/image")
def dataset_image(
    split: Annotated[str, Query(pattern=r"^(train|val|test)$")],
    path: str,
) -> FileResponse:
    dataset_root = resolve_master_dataset_root(PROJECT_ROOT)
    try:
        image_path = resolve_dataset_image(dataset_root, split, path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Dataset image not found") from exc
    return FileResponse(image_path)


@router.get("/training/artifact")
def training_artifact(name: str) -> FileResponse:
    if name not in ARTIFACT_NAMES:
        raise HTTPException(status_code=404, detail="Unknown training artifact")
    run_dir = _active_or_latest_run()
    artifact = run_dir / name if run_dir else None
    if artifact is None or not artifact.is_file():
        raise HTTPException(status_code=404, detail="Training artifact not found")
    return FileResponse(artifact)


@router.post("/training/evaluate")
def evaluate_validation_samples(request: EvaluationRequest) -> dict[str, Any]:
    dataset_root = resolve_master_dataset_root(PROJECT_ROOT)
    image_paths = sample_images(dataset_root, "val", request.count, request.seed)
    if not image_paths:
        raise HTTPException(status_code=409, detail="No validation images are available")
    model_path = _best_model_path()
    if model_path is None:
        raise HTTPException(status_code=409, detail="No trained plate model is available")

    model = _load_evaluation_model(model_path)
    predictions = model.predict(
        source=[str(path) for path in image_paths],
        conf=request.confidence,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    samples = []
    for image_path, prediction in zip(image_paths, predictions, strict=True):
        original = cv2.imread(str(image_path))
        annotated = prediction.plot()
        crops = []
        if original is not None and prediction.boxes is not None:
            image_height, image_width = original.shape[:2]
            for box in prediction.boxes:
                x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_width, x2), min(image_height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = natural_high_resolution_plate(original[y1:y2, x1:x2], request.scale_factor)
                crops.append(
                    {
                        "image": _image_data_url(crop),
                        "confidence": float(box.conf[0]),
                        "bbox": [x1, y1, x2, y2],
                    }
                )
        samples.append(
            {
                "name": image_path.name,
                "annotated_image": _image_data_url(annotated),
                "detections": crops,
            }
        )
    return {"model": str(model_path), "samples": samples}


def _active_or_latest_run() -> Path | None:
    job = training_job_manager.snapshot()
    if job.get("run_dir"):
        active = Path(job["run_dir"])
        if active.is_dir():
            return active
    return discover_latest_run(OUTPUT_ROOT)


def _run_payload(run_dir: Path | None, metrics: dict[str, Any]) -> dict[str, Any]:
    artifacts = training_artifacts(run_dir)
    for artifact in artifacts:
        artifact["url"] = f"/api/training/artifact?name={artifact['name']}" if artifact["exists"] else None
    best_model = run_dir / "weights" / "best.pt" if run_dir else None
    return {
        "directory": str(run_dir) if run_dir else None,
        "best_model": str(best_model) if best_model else None,
        "best_model_exists": bool(best_model and best_model.is_file()),
        "latest": metrics.get("latest"),
        "history": metrics.get("history", []),
        "artifacts": artifacts,
    }


def _best_model_path() -> Path | None:
    run_dir = _active_or_latest_run()
    candidate = run_dir / "weights" / "best.pt" if run_dir else None
    return candidate if candidate and candidate.is_file() else None


def _load_evaluation_model(model_path: Path):
    global _evaluation_model, _evaluation_model_path
    with _evaluation_model_lock:
        if _evaluation_model is None or _evaluation_model_path != model_path:
            from ultralytics import YOLO

            _evaluation_model = YOLO(str(model_path))
            _evaluation_model_path = model_path
        return _evaluation_model


def _image_data_url(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode inference image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
