# Two-stage YOLO11 vehicle and plate detection

The detection system is a cascade:

1. **Vehicle detection:** COCO-pretrained YOLO11 detects `car`, `motorcycle`, `bus`, and `truck`.
2. **Plate detection:** a custom YOLO11 model runs on each padded vehicle crop.
3. **Character detection (future):** the cascade exposes full-resolution plate crops for a custom YOLO26 character model.

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

The repository's harmonization and splitting pipeline already produces YOLO plate labels and split manifests. Run it before training if `data/processed/unified` is not present.

## Train stage two: plates

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py --stage plate --device 0
```

By default this uses:

- model: `yolo11n.pt`
- config: `configs/model/detection.yaml`
- split manifests: `reports/splits/{train,val,test}.txt`
- output: `models/detection/plate_yolo11/weights/best.pt`

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

## Future YOLO26 character stage

`TwoStageDetector.crop_plates()` returns `(plate_crop, detection_metadata)` pairs. A future character detector can consume each crop directly, sort detected characters from left to right (or by configured plate row), and attach the decoded text to the existing metadata. The reserved `character` section in `configs/model/two_stage.yaml` already defines the intended YOLO26 checkpoint and thresholds.
