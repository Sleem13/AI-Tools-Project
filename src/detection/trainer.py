"""Config-driven Ultralytics training for YOLO11 detection stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.detection.data import create_dataset_yaml


def train_detector(
    config: dict[str, Any],
    dataset_yaml_path: str | Path,
    output_dir: str | Path | None = None,
):
    """Train one detection stage from a prepared Ultralytics dataset YAML."""
    from ultralytics import YOLO

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    model_name = model_cfg.get("name", "yolo11n")
    if model_cfg.get("pretrained", True):
        model_source = model_cfg.get("weights", f"{model_name}.pt")
    else:
        model_source = model_cfg.get("architecture", f"{model_name}.yaml")

    model = YOLO(str(model_source))
    train_args = _build_training_args(config)
    train_args["data"] = str(dataset_yaml_path)
    train_args["project"] = str(output_dir or training_cfg.get("project", "models/detection"))
    return model.train(**train_args)


def train_yolo(
    config: dict[str, Any],
    train_image_dir: str | Path,
    val_image_dir: str | Path,
    output_dir: str | Path = "models/detection",
    dataset_yaml_path: str | Path = "configs/model/dataset.yaml",
    test_image_dir: str | Path | None = None,
    names: dict[int, str] | None = None,
):
    """Backward-compatible helper that creates a dataset YAML and trains YOLO11."""
    configured_names = config.get("data", {}).get("names")
    class_names = names or _normalise_names(configured_names) or {0: "license_plate"}
    create_dataset_yaml(
        train_path=train_image_dir,
        val_path=val_image_dir,
        test_path=test_image_dir,
        names=class_names,
        output_path=dataset_yaml_path,
    )
    return train_detector(config, dataset_yaml_path, output_dir)


def _normalise_names(names: dict[int, str] | list[str] | None) -> dict[int, str] | None:
    if names is None:
        return None
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(class_id): class_name for class_id, class_name in names.items()}


def _build_training_args(config: dict[str, Any]) -> dict[str, Any]:
    hp = config.get("hyperparameters", {})
    training = config.get("training", {})
    return {
        "epochs": hp.get("epochs", 100),
        "batch": hp.get("batch", 16),
        "imgsz": hp.get("imgsz", 640),
        "optimizer": hp.get("optimizer", "AdamW"),
        "lr0": hp.get("lr0", 0.001),
        "lrf": hp.get("lrf", 0.01),
        "momentum": hp.get("momentum", 0.937),
        "weight_decay": hp.get("weight_decay", 0.0005),
        "warmup_epochs": hp.get("warmup_epochs", 3.0),
        "box": hp.get("box", 7.5),
        "cls": hp.get("cls", 0.5),
        "dfl": hp.get("dfl", 1.5),
        "hsv_h": hp.get("hsv_h", 0.015),
        "hsv_s": hp.get("hsv_s", 0.7),
        "hsv_v": hp.get("hsv_v", 0.4),
        "degrees": hp.get("degrees", 0.0),
        "translate": hp.get("translate", 0.1),
        "scale": hp.get("scale", 0.5),
        "shear": hp.get("shear", 0.0),
        "perspective": hp.get("perspective", 0.0),
        "flipud": hp.get("flipud", 0.0),
        "fliplr": hp.get("fliplr", 0.5),
        "mosaic": hp.get("mosaic", 1.0),
        "mixup": hp.get("mixup", 0.0),
        "copy_paste": hp.get("copy_paste", 0.0),
        "patience": hp.get("patience", 20),
        "device": training.get("device", 0),
        "workers": training.get("workers", 4),
        "name": training.get("name", "exp"),
        "exist_ok": training.get("exist_ok", False),
        "save_period": training.get("save_period", 10),
        "deterministic": training.get("deterministic", True),
        "seed": training.get("seed", 42),
        "plots": training.get("plots", True),
        "cache": training.get("cache", False),
        "amp": training.get("amp", True),
    }
