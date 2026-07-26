# Master Plate workbench

The Training page productionizes every workflow in `Master_Plate_Dataset.ipynb` inside the existing React and FastAPI application.

## Notebook-to-app mapping

| Notebook workflow | App implementation |
|---|---|
| Install and verify PyTorch, CUDA, and Ultralytics | Live runtime readiness card from the API process |
| Count and preview random training images | Dataset view with split counts, label presence, and seeded samples |
| Train `yolo11n.pt` for 50 epochs at 640px and batch 16 | Validated training form backed by `master_plate_detection.yaml` |
| Display `results.png` and confusion matrices | Training artifact gallery |
| Load `best.pt` and test random validation images | Evaluate view using the latest completed checkpoint |
| Draw detections and crop plates | Annotated full-image and per-detection results |
| Enlarge crops with Lanczos interpolation | Configurable 2x–8x natural plate enhancement |

The notebook's inconsistent `Plate_Detection/...` and `runs/detect/...` paths are replaced by one run directory under `models/detection/`. Its undefined final `test_natural_plate_detection()` call is replaced by the tested evaluation API.

## Dataset location

The default expected structure is:

```text
data/processed/Master_Plate_Dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

To keep the dataset elsewhere, set its root before starting FastAPI:

```powershell
$env:MASTER_PLATE_DATASET_ROOT = "D:\datasets\Master_Plate_Dataset"
```

The app never serves arbitrary filesystem paths. Dataset image requests are constrained to the selected split directory, and training parameters are validated by the API.

## Run the application

Start the backend with the GPU environment so training and evaluation see CUDA:

```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn api.main:app --reload --port 8000
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/training` and use the four workbench views:

1. **Overview** — confirm Python, CUDA, dataset, and imported notebook evidence.
2. **Dataset** — inspect train, validation, or test images and matching labels.
3. **Train & results** — launch YOLO11, follow epochs/logs, and inspect artifacts.
4. **Evaluate** — run seeded random validation inference and inspect enhanced crops.

The API permits one training process at a time. Completed checkpoints are discovered automatically, and the main detection API selects the newest available plate checkpoint after training.
