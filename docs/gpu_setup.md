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

The backend may run from a separate CPU environment. To make the React
Training page launch jobs through an already-installed GPU environment, set:

```powershell
$env:ALPR_TRAINING_PYTHON = "D:\path\to\gpu-venv\Scripts\python.exe"
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## 4. Validate the training inputs

Run both preflights before allocating GPU time:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py `
  --stage plate `
  --config configs/model/master_plate_detection.yaml `
  --device 0 `
  --check

.\.venv-gpu\Scripts\python.exe scripts\train_detection.py `
  --stage character `
  --device 0 `
  --check
```

The plate check requires `data/processed/Master_Plate_Dataset/data.yaml` (or a
YAML supplied with `--data`). The character check uses the included segmented
plate dataset.

## 5. Train the plate detector

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

## 6. Train the character detector

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py `
  --stage character `
  --device 0 `
  --epochs 100 `
  --imgsz 640 `
  --batch 32
```

If the GPU runs out of memory, reduce `--batch` to 16 or 8. The resulting
`models/character/<run>/weights/best.pt` is discovered automatically by the
backend.

Stage one uses the COCO-pretrained YOLO11 vehicle detector and does not require
training unless a separate domain-specific vehicle dataset is prepared.

## 7. Run three-stage inference

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_detection_inference.py `
  --input C:\path\to\actual\images `
  --output reports\two_stage `
  --json reports\two_stage\detections.json `
  --device 0
```

Model checkpoints, datasets, W&B artifacts, and training runs are ignored by Git. Keep trained weights in `models/detection/.../weights/` and back them up separately or publish them as versioned release assets.
