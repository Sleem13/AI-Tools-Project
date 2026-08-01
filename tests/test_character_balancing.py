from __future__ import annotations

from pathlib import Path

import yaml
from scripts.build_balanced_character_dataset import build_balanced_character_dataset


def _write_sample(root: Path, split: str, stem: str, classes: list[int]) -> None:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.jpg").write_bytes(b"image")
    (label_dir / f"{stem}.txt").write_text(
        "\n".join(f"{class_id} 0.5 0.5 0.1 0.1" for class_id in classes), encoding="utf-8"
    )


def test_balancing_repeats_rare_training_images_only_and_is_deterministic(tmp_path: Path) -> None:
    dataset = tmp_path / "source"
    for index in range(4):
        _write_sample(dataset, "train", f"common-{index}", [0])
    _write_sample(dataset, "train", "rare", [1])
    _write_sample(dataset, "valid", "validation", [1])
    source_yaml = dataset / "data.yaml"
    source_yaml.write_text(
        yaml.safe_dump(
            {"train": "train/images", "val": "valid/images", "nc": 2, "names": ["common", "rare"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    output = tmp_path / "balanced"
    first = build_balanced_character_dataset(source_yaml, output, target_instances=2)
    manifest_first = (output / "train.txt").read_text(encoding="utf-8")
    second = build_balanced_character_dataset(source_yaml, output, target_instances=2)

    assert first["classes"]["rare"]["before"] == 1
    assert first["classes"]["rare"]["effective_after"] == 2
    assert first["balanced_train_entries"] == 6
    assert manifest_first == (output / "train.txt").read_text(encoding="utf-8")
    assert second["classes"] == first["classes"]
    balanced = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert balanced["val"].endswith("valid/images")
    assert balanced["train"].endswith("balanced/train.txt")
