"""Programmatic YOLOv8 training entry point for license plate detection."""

from __future__ import annotations

from pathlib import Path

from src.detection.data import create_dataset_yaml


def train_yolo(
    config: dict,
    train_image_dir: str | Path,
    val_image_dir: str | Path,
    output_dir: str | Path = "models/detection",
    dataset_yaml_path: str | Path = "configs/model/dataset.yaml",
) -> dict:
    """Train a YOLOv8 model for license plate detection.

    Args:
        config: Detection configuration dictionary (from detection.yaml).
        train_image_dir: Directory containing training images (YOLO labels
                         expected in sibling ``labels/`` directory).
        val_image_dir: Directory containing validation images.
        output_dir: Where YOLO saves training artifacts.
        dataset_yaml_path: Where to write the generated data YAML.

    Returns:
        Training results dictionary from ultralytics.
    """
    from ultralytics import YOLO

    model_cfg = config.get("model", {})
    hp = config.get("hyperparameters", {})
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})

    nc = 1
    names = {0: "license_plate"}

    create_dataset_yaml(
        train_path=str(train_image_dir),
        val_path=str(val_image_dir),
        names=names,
        output_path=dataset_yaml_path,
    )

    model_name = model_cfg.get("name", "yolov8n")
    pretrained = model_cfg.get("pretrained", True)

    if pretrained:
        model = YOLO(f"{model_name}.pt")
    else:
        model = YOLO(f"{model_name}.yaml")

    results = model.train(
        data=str(dataset_yaml_path),
        epochs=hp.get("epochs", 100),
        batch=hp.get("batch", 16),
        imgsz=hp.get("imgsz", 640),
        optimizer=hp.get("optimizer", "AdamW"),
        lr0=hp.get("lr0", 0.001),
        lrf=hp.get("lrf", 0.01),
        momentum=hp.get("momentum", 0.937),
        weight_decay=hp.get("weight_decay", 0.0005),
        warmup_epochs=hp.get("warmup_epochs", 3),
        warmup_momentum=hp.get("warmup_momentum", 0.8),
        warmup_bias_lr=hp.get("warmup_bias_lr", 0.1),
        box=hp.get("box", 7.5),
        cls=hp.get("cls", 0.5),
        dfl=hp.get("dfl", 1.5),
        hsv_h=hp.get("hsv_h", 0.015),
        hsv_s=hp.get("hsv_s", 0.7),
        hsv_v=hp.get("hsv_v", 0.4),
        degrees=hp.get("degrees", 0.0),
        translate=hp.get("translate", 0.1),
        scale=hp.get("scale", 0.5),
        shear=hp.get("shear", 0.0),
        perspective=hp.get("perspective", 0.0),
        flipud=hp.get("flipud", 0.0),
        fliplr=hp.get("fliplr", 0.5),
        mosaic=hp.get("mosaic", 1.0),
        mixup=hp.get("mixup", 0.0),
        copy_paste=hp.get("copy_paste", 0.0),
        device=train_cfg.get("device", 0),
        workers=train_cfg.get("workers", 4),
        project=str(output_dir),
        name=train_cfg.get("name", "exp"),
        exist_ok=train_cfg.get("exist_ok", False),
        patience=hp.get("patience", 20),
        save_period=train_cfg.get("save_period", 10),
        val_period=train_cfg.get("val_period", 1),
        deterministic=train_cfg.get("deterministic", True),
        seed=train_cfg.get("seed", 42),
    )

    return results
