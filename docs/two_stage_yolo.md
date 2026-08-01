# Vehicle, plate, and character detection

The detection system is a cascade:

1. **Vehicle detection:** COCO-pretrained YOLO11 detects `car`, `motorcycle`, `bus`, and `truck`.
2. **Plate detection:** a custom YOLO11 model runs on each padded vehicle crop.
3. **Character detection:** YOLO26 detects and classifies digits and Arabic characters inside every stage-two plate crop.

Stage one works without custom training because `yolo11n.pt` is pretrained on COCO. Stage two must be trained on this project's license-plate annotations before the full cascade can run.

## Dataset layout

Ultralytics detection datasets use matching `images` and `labels` trees:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each label is a YOLO text file containing one row per object:

```text
class_id x_center y_center width height
```

Coordinates are normalized to the image dimensions. The plate dataset uses one class:

```yaml
names:
  0: license_plate
```

The default plate workflow uses the prepared Master Plate dataset at
`data/processed/Master_Plate_Dataset`. Its `data.yaml` must contain populated
train and validation splits before training.

## Train stage two: plates

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py --stage plate --device 0
```

By default this uses:

- model: `yolo11n.pt`
- config: `configs/model/master_plate_detection.yaml`
- dataset: `data/processed/Master_Plate_Dataset/data.yaml`
- output: `models/detection/master_plate_yolo11/weights/best.pt`

The older harmonized/unified workflow remains available explicitly with
`--config configs/model/detection.yaml`.

For an independently prepared Ultralytics dataset:

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --stage plate `
  --data C:\datasets\plates\data.yaml `
  --epochs 150 `
  --imgsz 960 `
  --device 0
```

## Optional stage-one fine-tuning

The default stage-one checkpoint already detects COCO vehicle classes. Fine-tune only when the target camera domain needs it:

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --stage vehicle `
  --data C:\datasets\egyptian_traffic\data.yaml `
  --device 0
```

If the custom vehicle dataset uses class IDs `0..3`, update `vehicle.class_ids` in `configs/model/two_stage.yaml`. The default IDs `[2, 3, 5, 7]` are specifically the COCO IDs.

## Run the cascade

After stage-two training:

```powershell
.\.venv\Scripts\python.exe scripts\run_detection_inference.py `
  --input C:\images\traffic `
  --output reports\two_stage `
  --json reports\two_stage\detections.json `
  --device 0
```

To use a checkpoint outside the configured location:

```powershell
.\.venv\Scripts\python.exe scripts\run_detection_inference.py `
  --input image.jpg `
  --plate-weights C:\models\plate_yolo11\best.pt
```

Blue boxes identify vehicles; green boxes identify plates. JSON results include the vehicle box, globally mapped plate box, per-stage confidences, and a combined confidence used for duplicate suppression.

## Train stage three: characters

The included `data/raw/dataset_Charcters_ready_plates` dataset contains cropped
plates with YOLO boxes for 38 digit and Arabic-character classes.

```powershell
.\.venv-gpu\Scripts\python.exe scripts\train_detection.py `
  --stage character `
  --epochs 100 `
  --imgsz 640 `
  --batch 32 `
  --device 0 `
  --output models\character `
  --name yolo26_characters
```

The backend discovers the newest character checkpoint automatically. During
inference, `CharacterDetector` consumes each stage-two crop, clusters boxes
into rows, sorts each row right-to-left, separates letter and digit groups,
and maps the dataset's transliterated class names to Arabic glyphs. Character
results are included in the API response with class, glyph, confidence, row,
order, and crop-relative bounding box.

CRNN OCR remains an optional fallback when stage-three weights are unavailable.

Before publishing final metrics, create identity-grouped splits. The current
export has nine original base names shared between train and validation/test,
so its supplied split is suitable for development but not the final unbiased
benchmark.
