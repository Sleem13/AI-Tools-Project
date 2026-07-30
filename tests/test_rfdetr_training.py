from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from src.detection.rfdetr import convert_yolo_dataset_to_rfdetr_coco


def test_convert_yolo_dataset_to_rfdetr_coco_layout(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        image_path = dataset / "images" / split / f"{split}_plate.jpg"
        Image.new("RGB", (200, 100), color=(10, 20, 30)).save(image_path)
        (dataset / "labels" / split / f"{split}_plate.txt").write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: license_plate",
            ]
        ),
        encoding="utf-8",
    )

    summary = convert_yolo_dataset_to_rfdetr_coco(data_yaml, tmp_path / "rfdetr")

    assert summary.images == {"train": 1, "valid": 1}
    assert summary.annotations == {"train": 1, "valid": 1}
    train_coco = json.loads((tmp_path / "rfdetr" / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert train_coco["categories"] == [{"id": 1, "name": "license_plate"}]
    assert train_coco["images"][0]["file_name"] == "train_plate.jpg"
    assert train_coco["annotations"][0]["category_id"] == 1
    assert train_coco["annotations"][0]["bbox"] == [60.0, 40.0, 80.0, 20.0]
    assert (tmp_path / "rfdetr" / "train" / "train_plate.jpg").is_file()


def test_convert_yolo_character_classes_to_coco_categories(tmp_path: Path) -> None:
    dataset = tmp_path / "characters"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (100, 100), color=(30, 20, 10)).save(dataset / "images" / split / f"{split}.jpg")
        (dataset / "labels" / split / f"{split}.txt").write_text(
            "0 0.2 0.5 0.1 0.3\n1 0.8 0.5 0.1 0.3\n",
            encoding="utf-8",
        )
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: alif",
                "  1: 5",
            ]
        ),
        encoding="utf-8",
    )

    convert_yolo_dataset_to_rfdetr_coco(data_yaml, tmp_path / "rfdetr_chars")

    train_coco = json.loads((tmp_path / "rfdetr_chars" / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    assert train_coco["categories"] == [{"id": 1, "name": "alif"}, {"id": 2, "name": "5"}]
    assert [annotation["category_id"] for annotation in train_coco["annotations"]] == [1, 2]
