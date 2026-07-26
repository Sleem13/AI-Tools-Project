#!/usr/bin/env python3
"""Run the vehicle -> plate -> optional character cascade on images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.inference import build_two_stage_detector  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO11 vehicle-to-plate detection.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "model" / "two_stage.yaml")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Annotated image output directory")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON results file")
    parser.add_argument("--device", type=str, default=None, help="Override both stage devices")
    parser.add_argument("--vehicle-conf", type=float, default=None)
    parser.add_argument("--plate-conf", type=float, default=None)
    parser.add_argument("--plate-weights", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if args.device is not None:
        config.setdefault("vehicle", {})["device"] = args.device
        config.setdefault("plate", {})["device"] = args.device
    if args.vehicle_conf is not None:
        config.setdefault("vehicle", {})["conf_threshold"] = args.vehicle_conf
    if args.plate_conf is not None:
        config.setdefault("plate", {})["conf_threshold"] = args.plate_conf
    if args.plate_weights is not None:
        config.setdefault("plate", {})["weights"] = str(args.plate_weights.resolve())

    detector = build_two_stage_detector(config, PROJECT_ROOT)
    image_paths = _image_paths(args.input)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        detections = detector.predict(image)
        serialised = [detection.to_dict() for detection in detections]
        all_results.append({"image": str(image_path), "detections": serialised})
        print(f"{image_path.name}: {len(detections)} plate(s)")
        for detection in detections:
            print(
                f"  {detection.vehicle.class_name} -> plate {detection.plate.bbox} "
                f"plate_conf={detection.plate.confidence:.3f} "
                f"cascade_conf={detection.combined_confidence:.3f}"
            )
        if args.output:
            annotated = _draw_cascade(image, detections)
            cv2.imwrite(str(args.output / image_path.name), annotated)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")


def _image_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return [input_path]


def _draw_cascade(image, detections):
    annotated = image.copy()
    drawn_vehicles: set[tuple[float, float, float, float]] = set()
    for detection in detections:
        if detection.vehicle.bbox not in drawn_vehicles:
            vx1, vy1, vx2, vy2 = (int(value) for value in detection.vehicle.bbox)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 128, 0), 2)
            cv2.putText(
                annotated,
                f"{detection.vehicle.class_name} {detection.vehicle.confidence:.2f}",
                (vx1, max(15, vy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 128, 0),
                1,
            )
            drawn_vehicles.add(detection.vehicle.bbox)
        px1, py1, px2, py2 = (int(value) for value in detection.plate.bbox)
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"plate {detection.plate.confidence:.2f}",
            (px1, max(15, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return annotated


if __name__ == "__main__":
    main()
