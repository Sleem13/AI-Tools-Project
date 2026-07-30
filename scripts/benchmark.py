#!/usr/bin/env python3
"""CLI: Benchmark the full ALPR pipeline on a test set.

Usage:
    python scripts/benchmark.py --det-weights models/weights/detection_best.pt --ocr-weights models/weights/ocr_best.pth
    python scripts/benchmark.py --det-weights models/weights/detection_best.pt --ocr-weights models/weights/ocr_best.pth --test-list reports/splits/test.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from src.detection.inference import LicensePlateDetector
from src.evaluation.pipeline import ALPRPipeline
from src.ocr.inference import PlateReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ALPR pipeline.")
    parser.add_argument("--det-weights", type=Path, required=True)
    parser.add_argument("--ocr-weights", type=Path, required=True)
    parser.add_argument("--test-list", type=Path,
                        default=PROJECT_ROOT / "reports" / "splits" / "test.txt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detector = LicensePlateDetector(
        weights_path=args.det_weights,
        conf_threshold=args.conf,
        device=args.device,
    )
    reader = PlateReader(
        weights_path=args.ocr_weights,
        device=args.device,
    )
    pipeline = ALPRPipeline(detector=detector, reader=reader, min_confidence=args.conf)

    test_images = []
    if args.test_list.exists():
        with open(args.test_list, encoding="utf-8") as f:
            test_images = [Path(line.strip()) for line in f if line.strip()]
    else:
        print(f"Test list not found: {args.test_list}")
        return

    print(f"Benchmarking on {len(test_images)} images...")
    total_time = 0.0
    plate_count = 0
    for img_path in test_images:
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        t0 = time.perf_counter()
        results = pipeline.run(image)
        elapsed = (time.perf_counter() - t0) * 1000.0
        total_time += elapsed
        plate_count += len(results)

        for r in results:
            print(f"  {img_path.name}: {r['plate_text']} (conf={r['confidence']:.2f})")

    avg_time = total_time / len(test_images) if test_images else 0.0
    print("\nResults:")
    print(f"  Images: {len(test_images)}")
    print(f"  Plates detected: {plate_count}")
    print(f"  Avg inference time: {avg_time:.1f} ms/image")
    print(f"  Total inference time: {total_time:.1f} ms")


if __name__ == "__main__":
    main()
