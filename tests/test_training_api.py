"""FastAPI integration tests for the Master Plate workbench."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from api.main import app
from api.routes import training
from fastapi.testclient import TestClient


def _create_dataset(root: Path) -> None:
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        cv2.imwrite(str(image_dir / f"{split}.jpg"), np.zeros((12, 20, 3), dtype=np.uint8))
        (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\nnames: [license_plate]\n", encoding="utf-8")


def test_workbench_and_dataset_sample_endpoints(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Master_Plate_Dataset"
    _create_dataset(dataset_root)
    monkeypatch.setenv("MASTER_PLATE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(training, "OUTPUT_ROOT", tmp_path / "models")
    client = TestClient(app)

    workbench = client.get("/api/training/workbench")
    samples = client.get("/api/training/dataset/samples", params={"split": "train", "count": 4, "seed": 7})
    image = client.get("/api/training/dataset/image", params={"split": "train", "path": "train.jpg"})

    assert workbench.status_code == 200
    assert workbench.json()["dataset"]["ready"] is True
    assert workbench.json()["dataset"]["splits"]["train"] == {"images": 1, "labels": 1}
    assert samples.status_code == 200
    assert samples.json()["images"][0]["label_exists"] is True
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_start_training_builds_reproducible_command(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Master_Plate_Dataset"
    _create_dataset(dataset_root)
    monkeypatch.setenv("MASTER_PLATE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(training, "OUTPUT_ROOT", tmp_path / "models")
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"status": "training", "run_name": kwargs["run_name"], "log_tail": ""}

    monkeypatch.setattr(training.training_job_manager, "start", fake_start)
    client = TestClient(app)

    response = client.post(
        "/api/training/start",
        json={"epochs": 50, "imgsz": 640, "batch": 16, "device": "cpu"},
    )

    assert response.status_code == 200
    command = captured["command"]
    assert command[command.index("--data") + 1] == str(dataset_root / "data.yaml")
    assert command[command.index("--config") + 1].endswith("master_plate_detection.yaml")
    assert command[command.index("--epochs") + 1] == "50"
    assert command[command.index("--name") + 1].startswith("master_plate_yolo11_")


def test_start_training_can_use_external_gpu_python(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "Master_Plate_Dataset"
    _create_dataset(dataset_root)
    training_python = tmp_path / "gpu-venv" / "python.exe"
    training_python.parent.mkdir()
    training_python.touch()
    monkeypatch.setenv("MASTER_PLATE_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(training, "_training_python", lambda: training_python)
    monkeypatch.setattr(
        training,
        "_training_runtime",
        lambda: {"ready": True, "cuda_available": True},
    )
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"status": "training", "run_name": kwargs["run_name"], "log_tail": ""}

    monkeypatch.setattr(training.training_job_manager, "start", fake_start)
    client = TestClient(app)

    response = client.post(
        "/api/training/start",
        json={"epochs": 1, "imgsz": 320, "batch": 2, "device": "0"},
    )

    assert response.status_code == 200
    assert captured["command"][0] == str(training_python)


def test_start_character_training_uses_segmented_dataset(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "characters"
    for split in ("train", "valid", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        cv2.imwrite(str(image_dir / f"{split}.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
        (label_dir / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.4", encoding="utf-8")
    (dataset_root / "data.yaml").write_text(
        "train: train/images\nval: valid/images\nnames: ['0']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CHARACTER_DATASET_ROOT", str(dataset_root))
    monkeypatch.setattr(training, "CHARACTER_OUTPUT_ROOT", tmp_path / "character-models")
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"status": "training", "run_name": kwargs["run_name"], "log_tail": ""}

    monkeypatch.setattr(training.training_job_manager, "start", fake_start)
    client = TestClient(app)

    response = client.post(
        "/api/training/start",
        json={
            "stage": "character",
            "epochs": 100,
            "imgsz": 640,
            "batch": 32,
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    command = captured["command"]
    assert command[command.index("--stage") + 1] == "character"
    assert command[command.index("--data") + 1] == str(dataset_root / "data.yaml")
    assert command[command.index("--config") + 1].endswith("character_detection.yaml")
    assert captured["stage"] == "character"
