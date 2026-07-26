"""YOLO-format dataset loader and data configuration for license plate detection."""

from __future__ import annotations

from pathlib import Path

import yaml


def create_dataset_yaml(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path | None = None,
    names: dict[int, str] | None = None,
    output_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> dict:
    """Generate a YOLO data.yaml dict and optionally persist it.

    Args:
        train_path: Path or glob pattern for training images.
        val_path: Path or glob pattern for validation images.
        test_path: Optional path for test images.
        names: Class mapping {0: "license_plate"}.
        output_path: If provided, write the YAML here.
        dataset_root: Root used to resolve relative split paths.

    Returns:
        Dictionary suitable for ultralytics YOLO training.
    """
    if names is None:
        names = {0: "license_plate"}

    cfg = {
        "path": str(Path(dataset_root or Path.cwd()).resolve()),
        "train": str(train_path),
        "val": str(val_path),
        "nc": len(names),
        "names": names,
    }
    if test_path is not None:
        cfg["test"] = str(test_path)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    return cfg


class LicensePlateDataset:
    """Thin wrapper referencing the YOLO dataset YAML for downstream use.

    The actual data loading is handled internally by ultralytics.YOLO.
    """

    def __init__(self, data_yaml: str | Path | dict) -> None:
        if isinstance(data_yaml, dict):
            self.config = data_yaml
        else:
            with open(data_yaml, encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

    @property
    def nc(self) -> int:
        return self.config.get("nc", 1)

    @property
    def names(self) -> dict[int, str]:
        return self.config.get("names", {0: "license_plate"})
