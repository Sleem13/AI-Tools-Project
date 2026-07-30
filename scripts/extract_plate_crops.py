#!/usr/bin/env python3
"""Extract license plate crops using detection model to build OCR training set.

Usage:
    python scripts/extract_plate_crops.py --weights models/weights/detection_best.pt
    python scripts/extract_plate_crops.py --weights models/weights/detection_best.pt --input-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.alpr_dataset.io_utils import list_images
from src.detection.inference import LicensePlateDetector
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plate crops for OCR training.")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to trained detection model weights")
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed" / "unified" / "images",
                        help="Directory of input images")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "ocr",
                        help="Output directory for crops + labels")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = LicensePlateDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )

    images = list_images(args.input_dir)
    if not images:
        print(f"No images found in {args.input_dir}")
        return

    records: list[dict] = []
    for img_path in tqdm(images, desc="Extracting crops"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        crops = detector.crop_plates(image, min_confidence=args.conf)
        for i, (crop, conf) in enumerate(crops):
            stem = f"{img_path.stem}_crop_{i}"
            crop_path = crops_dir / f"{stem}.jpg"
            cv2.imwrite(str(crop_path), crop)
            records.append({
                "image_path": str(crop_path),
                "source_image": str(img_path),
                "confidence": round(conf, 4),
                "text": "",
            })

    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Extracted {len(records)} plate crops -> {crops_dir}")
    print(f"Labels (fill in 'text' field manually or via annotation) -> {labels_path}")


if __name__ == "__main__":
    main()
