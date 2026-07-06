# Execution Guide

## 0. Environment setup

```bash
cd alpr_dataset_pipeline
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --break-system-packages   # drop the flag on a normal venv
```

## 1. Get the data in place

```bash
# Using the Kaggle CLI (requires ~/.kaggle/kaggle.json credentials):
kaggle datasets download -d mahmoudkhater99/car-plate -p data/raw/dataset_A --unzip
kaggle datasets download -d mahmoudeldebase/egyptian-cars-plates -p data/raw/dataset_B --unzip
```

Or download manually from the Kaggle web UI and extract into
`data/raw/dataset_A/` and `data/raw/dataset_B/` respectively.

## 2. Inspect (Part 1) — always run this first

```bash
python scripts/run_inspection.py
```

Open `reports/eda/dataset_summary.md`. Check:
- `image_formats` / `annotation_formats` — confirms what's actually on disk.
- `n_images_missing_annotations` / `n_orphan_annotations` — flags mismatches early.

Then edit `configs/datasets.yaml`:
- Set `annotation_format` per dataset (`yolo`, `voc_xml`, `coco_json`, or `csv`).
- Set `image_subdir` / `annotation_subdir` if images and labels live in separate folders.
- Set `class_map` to the dataset's real `{id: name}` mapping.
- If a CSV format, you'll also need to check the column names used by
  `alpr_dataset.annotations.parsers.parse_csv_annotations` (`filename_col`,
  `box_cols`, `class_col` — defaults are `filename`, `(xmin,ymin,xmax,ymax)`, `class`)
  and adjust the call in `annotations/loader.py` or pass overrides if the
  real CSV uses different headers.

## 3. EDA figures (Part 2)

```bash
python scripts/run_eda.py
```

Outputs land in `reports/figures/<dataset_name>/` as matched `.png` + `.svg`
pairs — 19 figure types per dataset covering size, resolution, bbox geometry,
class balance, sample galleries, photometric quality, color, entropy,
duplicates, and outliers.

## 4. Quality assessment (Part 3)

```bash
python scripts/run_quality_check.py
```

Read `reports/quality/<dataset>_quality_report.md` for the summary table,
and the paired `.json` for full file-level listings (capped at 200 entries
per category to keep the file manageable — increase the caps in
`eda/quality.py` if you need the full lists).

## 5. Harmonization (Part 4)

```bash
python scripts/run_harmonization.py
```

Produces `data/processed/unified/images/` and `.../labels/` with every
image renamed `<dataset>__<original_stem>.jpg` and every annotation
converted to normalized YOLO txt against the `unified_class_map` in
`configs/datasets.yaml`. Full audit trail in
`reports/harmonization/harmonization_metadata.json`.

## 6. Preprocessing (Part 5)

```bash
python scripts/run_preprocessing.py
```

Runs the step chain configured in `configs/preprocessing_config.yaml`
against the harmonized image set, writing to
`data/processed/preprocessed/` plus before/after comparison PNGs in
`reports/figures/before_after/`. Edit the YAML to enable/disable/reorder
steps — no code changes needed for parameter tuning.

To preprocess a different image set instead of the harmonized one:

```bash
python scripts/run_preprocessing.py --input-dir data/raw/dataset_A
```

## 7. Statistics (Part 6)

```bash
python scripts/run_statistics.py
```

Writes `reports/statistics/<dataset>_statistics.json`,
`<dataset>_resolution_table.csv`, `combined_statistics.csv`, and
`statistics_report.md`.

## 8. Train / val / test split (Part 7)

```bash
# Default 70/15/15 from preprocessing_config.yaml
python scripts/run_split.py

# 80/10/10 custom ratio, physically copying files into split folders
python scripts/run_split.py --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --copy-files
```

Splitting operates on the **harmonized** dataset (`data/processed/unified/`)
so the train/val/test partition is consistent across both original source
datasets. Manifests land in `reports/splits/`.

## Running everything at once

```bash
python scripts/run_full_pipeline.py
```

Runs stages 2-8 above in order (stage 1's data-download step is manual and
one-time). Stops immediately if any stage exits non-zero, so you can fix
config and re-run just that stage rather than restarting from scratch.

## Logs

Every script writes to `reports/logs/alpr_dataset.log` in addition to the
Rich-formatted console output, so a full-pipeline run leaves a complete,
timestamped audit trail even if you only watched the console for one stage.
