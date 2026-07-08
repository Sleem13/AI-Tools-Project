#!/usr/bin/env python3
"""CLI: Run license plate detection inference on images or directories.

Usage:
    python scripts/run_detection_inference.py --weights models/weights/detection_best.pt --input image.jpg
    python scripts/run_detection_inference.py --weights models/weights/detection_best.pt --input data/test/ --output results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from src.detection.inference import LicensePlateDetector
from src.visualization.detection_viz import draw_detections


def main() -> None:
    parser = argparse.ArgumentParser(description="Run license plate detection inference.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    detector = LicensePlateDetector(
        weights_path=args.weights,
        conf_threshold=args.conf,
        device=args.device,
    )

    if args.input.is_dir():
        image_paths = sorted(args.input.rglob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    else:
        image_paths = [args.input]

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping unreadable: {img_path}")
            continue

        detections = detector.predict(image)
        print(f"{img_path.name}: {len(detections)} plate(s) detected")
        for det in detections:
            print(f"  {det.class_name}: {det.bbox} conf={det.confidence:.3f}")

        if args.output:
            det_dicts = [
                {"bbox": d.bbox, "confidence": d.confidence, "plate_text": ""}
                for d in detections
            ]
            annotated = draw_detections(image, det_dicts)
            out_path = args.output / img_path.name
            cv2.imwrite(str(out_path), annotated)
            print(f"  -> annotated saved to {out_path}")


if __name__ == "__main__":
    main()
