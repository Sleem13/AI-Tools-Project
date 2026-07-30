#!/usr/bin/env python3
"""CLI: Run OCR inference on cropped plate images.

Usage:
    python scripts/run_ocr_inference.py --weights models/weights/ocr_best.pth --input crop.jpg
    python scripts/run_ocr_inference.py --weights models/weights/ocr_best.pth --input crops/ --output results.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.ocr.inference import PlateReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on plate crop images.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Output text file with results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    reader = PlateReader(
        weights_path=args.weights,
        device=args.device,
    )

    if args.input.is_dir():
        image_paths = sorted(args.input.rglob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    else:
        image_paths = [args.input]

    results: list[str] = []
    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping unreadable: {img_path}")
            continue
        text = reader.read_plate(image)
        line = f"{img_path.name}: {text}"
        results.append(line)
        print(line)

    if args.output:
        args.output.write_text("\n".join(results), encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
