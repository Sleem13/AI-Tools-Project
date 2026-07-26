# ALPR Dataset Pipeline

Production-grade dataset engineering and ALPR (Automatic License Plate Recognition) pipeline for Egyptian license plates.

Covers the full lifecycle: raw dataset ingestion, inspection, EDA, quality assessment, harmonization, preprocessing, statistics, train/val/test splitting, two-stage YOLO11 vehicle-to-plate detection, text recognition, and end-to-end evaluation.

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
  detection/            # YOLO11 vehicle -> plate cascade
  ocr/                  # CRNN + CTC text recognition
  evaluation/           # End-to-end ALPR evaluation
  postprocessing/       # Egyptian plate text formatting
  visualization/        # Detection result visualization
scripts/                # 14 CLI entry points
configs/                # YAML-driven pipeline configuration
app/                    # Streamlit dashboard
tests/                  # Unit tests
```

## Quick Start

```bash
# Clone and install
git clone <repo-url> && cd AI-Tools-Project
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt -r requirements-dev.txt

# Download datasets from Kaggle
kaggle datasets download -d mahmoudkhater99/car-plate -p data/raw/dataset_A --unzip
kaggle datasets download -d mahmoudeldebase/egyptian-cars-plates -p data/raw/dataset_B --unzip

# Run the full 7-stage pipeline
python scripts/run_full_pipeline.py

# Train the custom stage-two plate detector (stage one uses COCO YOLO11)
python scripts/train_detection.py --stage plate

# Train OCR model
python scripts/train_ocr.py

# Run inference
python scripts/run_detection_inference.py --input test_images/ --output reports/two_stage/
python scripts/run_ocr_inference.py --weights models/weights/ocr_best.pt --input plate_crops/

# Launch dashboard
streamlit run app/streamlit_app.py
```

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
- `model/two_stage.yaml` — cascade inference and future YOLO26 character-stage settings
- `model/ocr.yaml` — CRNN architecture and training config

## Testing

```bash
pytest tests/ -v
```

See [Two-stage YOLO11 vehicle and plate detection](docs/two_stage_yolo.md) for dataset layout, training, inference, and the future YOLO26 character-stage contract.

For NVIDIA training, follow the [RTX 2000 Ada GPU environment setup](docs/gpu_setup.md). It uses a dedicated Python 3.12 environment and pinned CUDA-enabled PyTorch wheels.

## License

MIT
