# NVIDIA GPU environment for YOLO training

This project targets an NVIDIA RTX 2000 Ada Generation GPU with 16 GB VRAM. Use a dedicated Python 3.12 environment for training; do not reuse a CPU-only or Python 3.14 environment.

## 1. Install the NVIDIA driver

Install the current NVIDIA RTX Enterprise or Studio driver for the RTX 2000 Ada, restart Windows, and confirm that the driver can see the GPU:

```powershell
nvidia-smi
```

The PyTorch wheel contains the CUDA runtime needed by this project. A separate CUDA Toolkit installation is not required unless compiling custom CUDA extensions.

## 2. Create the Python 3.12 environment

From the repository root:

```powershell
py -3.12 -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-gpu\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` installs matching PyTorch 2.13 and torchvision 0.28 CUDA 12.6 wheels. The normal `requirements.txt` remains suitable for CPU development and CI.

## 3. Verify CUDA before training

```powershell
.\.venv-gpu\Scripts\python.exe scripts\check_gpu.py
```

The command must report:

- Python 3.12
- a PyTorch CUDA runtime
- `NVIDIA RTX 2000 Ada Generation`
- approximately 16 GB VRAM
- a successful CUDA matrix test

Do not begin a long training run if this check fails.

## 4. Train the plate detector

First prepare the actual dataset so the configured split manifests point to existing images and YOLO label files. Then run:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py --stage plate --device 0
```

The default plate configuration uses `imgsz: 960`, `batch: 16`, and automatic mixed precision. This should be a reasonable starting point for 16 GB VRAM. If CUDA runs out of memory, reduce the batch first:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py --stage plate --device 0 --batch 8
```

If memory remains comfortably below 16 GB, try `--batch 24` or `--batch 32`. Monitor utilization with:

```powershell
nvidia-smi -l 2
```

## 5. Run two-stage inference

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_detection_inference.py `
  --input C:\path\to\actual\images `
  --output reports\two_stage `
  --json reports\two_stage\detections.json `
  --device 0
```

Model checkpoints, datasets, W&B artifacts, and training runs are ignored by Git. Keep trained weights in `models/detection/.../weights/` and back them up separately or publish them as versioned release assets.
