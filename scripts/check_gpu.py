#!/usr/bin/env python3
"""Fail-fast validation for the CUDA environment used by YOLO training."""

from __future__ import annotations

import platform
import sys

import torch
import ultralytics


def main() -> int:
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"Ultralytics: {ultralytics.__version__}")

    if sys.version_info[:2] != (3, 12):
        print("ERROR: use Python 3.12 for the supported Windows CUDA environment.")
        return 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable. Check the NVIDIA driver and install requirements-gpu.txt in a fresh venv.")
        return 1

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    total_memory_gib = properties.total_memory / 1024**3
    print(f"GPU: {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"VRAM: {total_memory_gib:.1f} GiB")

    first = torch.rand((1024, 1024), device=device)
    second = torch.rand((1024, 1024), device=device)
    result = first @ second
    torch.cuda.synchronize()
    print(f"CUDA matrix test: OK ({result.shape[0]}x{result.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
