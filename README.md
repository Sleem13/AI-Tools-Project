# ALPR Dataset Pipeline

Production-grade dataset engineering and ALPR (Automatic License Plate Recognition) pipeline for Egyptian license plates.

Covers the full lifecycle: raw dataset ingestion, inspection, EDA, quality assessment, harmonization, preprocessing, statistics, train/val/test splitting, YOLO11 vehicle and plate detection, YOLO26 character detection, optional CRNN OCR, and end-to-end evaluation.

## Architecture

```
src/
  alpr_dataset/         # Core 7-stage dataset pipeline
    annotations/        # Unified schema + parsers (YOLO, VOC XML, COCO, CSV)
    inspection/         # Part 1: scanner, hashing, image stats
    eda/                # Part 2-3: 22 plot types, quality assessment
    harmonization/      # Part 4: normalize to unified YOLO format
    preprocessing/      # Part 5: 14 configurable transforms
    statistics/         # Part 6: stats generation
    splitting/          # Part 7: stratified train/val/test split
  dataset/              # Inspector/reporter (used by main.py CLI)
  detection/            # Vehicle -> plate -> character cascade
  ocr/                  # CRNN + CTC text recognition
  evaluation/           # End-to-end ALPR evaluation
  postprocessing/       # Egyptian plate text formatting
  visualization/        # Detection result visualization
scripts/                # 14 CLI entry points
configs/                # YAML-driven pipeline configuration
api/                    # FastAPI backend
frontend/               # Vite React application
tests/                  # Unit tests
```

## Prerequisites

- Python 3.12 is recommended. The GPU setup is pinned to Python 3.12; do not
  reuse an incompatible Python 3.13/3.14 environment.
- Node.js 20.19+ or 22.12+.
- An NVIDIA CUDA environment is optional for using the app, but recommended
  for training.

```powershell
# Windows, from the repository root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

cd frontend
npm install
cd ..
```

For CUDA training, follow [docs/gpu_setup.md](docs/gpu_setup.md) and use its
`.venv-gpu` environment instead of the CPU environment above.

If FastAPI runs from the normal CPU environment while training uses another
existing CUDA environment, point the backend at that interpreter before
starting it:

```powershell
$env:ALPR_TRAINING_PYTHON = "D:\path\to\gpu-venv\Scripts\python.exe"
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

The Training page will probe that interpreter and launch both plate and
character jobs through it; the backend itself does not need CUDA.

Before any long run, validate the selected dataset and device without
downloading a model or starting an epoch:

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

The check reports split sizes and class count. It exits with an actionable
error if a YAML, image, split entry, or CUDA runtime is missing.

## Choose a Run Mode

### 1. Frontend preview only

Use this mode to inspect the interface without starting Python:

```powershell
cd frontend
npm run dev
```

Open `http://localhost:5173`. Routes and static UI will work, while live
datasets, training state, model health, and inference will show the backend as
offline. Detection and training require the FastAPI backend.

### 2. Use the complete app without training a new model

You do not need to retrain if a compatible plate detector checkpoint already
exists. Put `best.pt` in one of these supported locations:

1. `models/detection/plate_yolo11/weights/best.pt`
2. `models/weights/detection_best.pt`
3. `models/detection/<run-name>/weights/best.pt`

OCR is optional. If available, put its checkpoint at
`models/weights/ocr_best.pth`. Without OCR weights, plate boxes still work but
recognized plate text will be empty.

Start the backend:

```powershell
# Terminal 1, repository root, with the Python environment activated
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend:

```powershell
# Terminal 2
cd frontend
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to
`http://localhost:8000`. Confirm the connection with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-RestMethod http://127.0.0.1:5173/api/pipeline/status
```

If no supported plate checkpoint exists, the application still exposes
datasets, workflow, settings, and training pages, but image/video detection
returns HTTP 503 until training finishes or weights are supplied.

### 3. Train the plate detector, then use the app

The Master Plate dataset must contain at least:

```text
data/processed/Master_Plate_Dataset/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

To rebuild that structure from the Egyptian vehicle and motorcycle sources
already under `data/raw`, run:

```powershell
python scripts/run_harmonization.py
python scripts/run_split.py
```

The split command now materializes
`data/processed/Master_Plate_Dataset/data.yaml` plus its image and label
folders. Use `python scripts/run_full_pipeline.py` when you also want the
inspection, EDA, quality, preprocessing, and statistics reports; that complete
engineering pass is substantially slower.

If the dataset is elsewhere, set its root before starting the backend:

```powershell
$env:MASTER_PLATE_DATASET_ROOT = "D:\datasets\Master_Plate_Dataset"
```

Stage one normally does **not** need project-specific training. It uses the
COCO-pretrained `yolo11n.pt` vehicle classes for cars, motorcycles, buses, and
trucks. Only train `--stage vehicle` after preparing the optional domain
dataset referenced by `configs/model/vehicle_detection.yaml`.

Recommended workflow:

1. Activate the Python 3.12 CUDA environment.
2. Start FastAPI and the frontend using the commands from mode 2.
3. Open `http://localhost:5173/training`.
4. Verify that the dataset and CUDA runtime are ready.
5. Configure epochs, image size, batch size, and device, then select
   **Start training**.

FastAPI launches the training process, reports live epoch metrics and logs, and
writes runs under `models/detection/master_plate_yolo11_<timestamp>/`. The
backend automatically selects the newest `weights/best.pt` after training, so
restart is not required.

To train directly from the command line instead:

```powershell
python scripts/train_detection.py `
  --stage plate `
  --config configs/model/master_plate_detection.yaml `
  --data data/processed/Master_Plate_Dataset/data.yaml `
  --epochs 50 `
  --imgsz 640 `
  --batch 16 `
  --device 0 `
  --output models/detection `
  --name master_plate_yolo11_manual
```

Use `--device cpu` when CUDA is unavailable. CPU training is supported but can
be considerably slower.

Train stage three on the included segmented-character dataset:

```powershell
python scripts/train_detection.py `
  --stage character `
  --epochs 100 `
  --imgsz 640 `
  --batch 32 `
  --device 0 `
  --output models/character `
  --name yolo26_characters
```

The complete model-training order is therefore:

1. Run the plate preflight and train stage two.
2. Run the character preflight and train stage three.
3. Keep the pretrained stage-one vehicle model unless a separate annotated
   vehicle dataset is available.
4. Start FastAPI from the same GPU environment, then start the frontend.

For a quick command-path smoke test, use one epoch with a unique name. This is
not a useful trained checkpoint:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py `
  --stage character --epochs 1 --imgsz 320 --batch 2 --device 0 `
  --name character_smoke_test
```

The Character tab at `http://localhost:5173/training` launches the same
training workflow. At inference time, the backend automatically discovers the
newest `models/character/<run>/weights/best.pt`, runs it on every stage-two
plate crop, orders detections right-to-left by row, and maps the 38 dataset
classes to Arabic glyphs and digits. CRNN remains an optional fallback when a
character checkpoint is unavailable.

For final benchmark reporting, regroup the character dataset by original plate
identity before splitting. The current Roboflow export contains nine base
image names shared across train/validation or train/test; leaving them in place
can slightly inflate validation metrics.

The raw dataset engineering and optional OCR commands remain available:

```powershell
python scripts/run_full_pipeline.py
python scripts/train_ocr.py
python scripts/run_detection_inference.py --input test_images/ --output reports/two_stage/
python scripts/run_ocr_inference.py --weights models/weights/ocr_best.pth --input plate_crops/
```

Human review corrections from the Vite media lab can be imported into a
reviewed feedback folder:

```powershell
python scripts/import_review_feedback.py path\to\media-review.json
```

This writes corrected text, model output, detection boxes, diagnostics, and
plate crop images under `data/reviewed/human_feedback/`. Use those artifacts
for annotation and fine-tuning; YOLO weights are updated by retraining on
reviewed labels, not by directly editing a `.pt` file.

## Frontend and Backend Configuration

During local development, keep `VITE_API_BASE_URL` empty and use the Vite
proxy. Copy the example only when configuration is needed:

```powershell
Copy-Item frontend\.env.example frontend\.env
```

For a separately hosted backend, set the public API origin at frontend build
time:

```dotenv
VITE_API_BASE_URL=https://api.example.com
```

Allow the frontend origin on FastAPI:

```powershell
$env:ALPR_CORS_ORIGINS = "https://app.example.com,http://localhost:5173"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

`VITE_API_PROXY_TARGET` changes the local Vite proxy target. Both API requests
and backend-returned video, dataset-image, and training-artifact URLs use the
configured API base.

## Recent Frontend/Backend Integration Work

- Added `VITE_API_BASE_URL` support for separate frontend/backend deployments.
- Fixed Vite environment loading for `VITE_API_PROXY_TARGET`.
- Normalized relative backend media URLs for videos, dataset samples, and
  training artifacts.
- Added actionable connection and API validation errors in the frontend.
- Added configurable FastAPI CORS origins through `ALPR_CORS_ORIGINS`.
- Added the FastAPI, Uvicorn, and multipart packages to project metadata.
- Added typed API responses and explicit offline states for Datasets and
  Workflow.

The integration was verified with a production frontend build, frontend lint,
127 backend tests, real one-epoch CUDA smoke runs for both trainable stages,
and live FastAPI health/runtime requests.

## Pipeline Stages

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `run_inspection.py` | Scan datasets, detect formats, compute image metadata |
| 2 | `run_eda.py` | Generate 22 EDA figure types per dataset |
| 3 | `run_quality_check.py` | Detect corruption, duplicates, missing labels |
| 4 | `run_harmonization.py` | Normalize all datasets to unified YOLO format |
| 5 | `run_preprocessing.py` | Apply configurable transform chain |
| 6 | `run_statistics.py` | Generate resolution/quality statistics |
| 7 | `run_split.py` | Stratified train/val/test splitting |

## Configuration

All settings are YAML-driven in `configs/`:

- `pipeline_config.yaml` — Global paths, thresholds, seeds
- `datasets.yaml` — Per-dataset format, class maps, subdirectories
- `preprocessing_config.yaml` — Transform chain and split ratios
- `model/detection.yaml` — YOLO11 plate training
- `model/vehicle_detection.yaml` — optional YOLO11 vehicle fine-tuning
- `model/character_detection.yaml` — YOLO26 character training on segmented plates
- `model/two_stage.yaml` — three-stage cascade inference and decoding thresholds
- `model/ocr.yaml` — CRNN architecture and training config

## Testing

```bash
pytest tests/ -v
```

See [Vehicle, plate, and character detection](docs/two_stage_yolo.md) for dataset layout, training, inference, and stage-three decoding.

For NVIDIA training, follow the [RTX 2000 Ada GPU environment setup](docs/gpu_setup.md). It uses a dedicated Python 3.12 environment and pinned CUDA-enabled PyTorch wheels.

The [Master Plate workbench](docs/master_plate_workbench.md) moves the complete `Master_Plate_Dataset.ipynb` workflow into the React/FastAPI Training page, including runtime readiness, dataset samples, YOLO11 launch and monitoring, training artifacts, random validation inference, and enhanced plate crops.

The [interactive workflow and media lab](docs/interactive_workflow.md) visualize the complete learning and inference lifecycle and support human-reviewed detection on real images and asynchronously processed videos.

## License

MIT
