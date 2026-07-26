# Workflow visualization and human review media lab

The application now has two complementary visual surfaces:

- `/workflow` shows the complete live ALPR lifecycle from dataset engineering through model training, three-stage detection, and human feedback.
- `/` accepts real images and videos for auditable inference and human review.

## Workflow visualization

The workflow page reads actual API state. Nodes change between waiting, ready, complete, and planned based on dataset outputs, the Master Plate workbench, trained checkpoints, and model health.

The learning lane represents:

```text
Raw datasets -> inspect and harmonize -> Master Plate splits -> YOLO11 plate training
```

The inference lane represents:

```text
Image/video -> YOLO11 vehicles -> YOLO11 plates -> YOLO26 characters -> human verification
```

Human decisions form a feedback loop for future annotation and retraining.

## Image review

Upload an image from the Dashboard and the API returns:

- the full annotated image;
- vehicle and plate boxes with per-stage confidence;
- OCR and formatted text when the reader is available;
- a 4x Lanczos-enhanced crop for every plate.

Reviewers can accept, reject, or flag every detection. **Export review JSON** downloads the model output and decisions as an auditable record.

## Video review

The media lab accepts MP4, MOV, AVI, MKV, WebM, and M4V files up to 512 MB. Video processing runs asynchronously so the UI can poll and display:

- processed and total frames;
- percentage progress;
- frames containing detections;
- total plate observations;
- timestamped detection and OCR events;
- the completed annotated video for playback or download.

The frame-stride control trades temporal precision for speed. A stride of `1` runs the cascade on every frame; the default `3` reuses the most recent boxes between inference frames.

Only one video job runs per API process at a time to protect GPU memory and model state. Generated uploads and videos are stored under `reports/video_jobs/` and ignored by Git.
