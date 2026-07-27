"""Human-in-the-loop image and video ALPR inference APIs."""

from __future__ import annotations

import base64
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from src.detection.workbench import natural_high_resolution_plate

from api.deps import get_detector, get_reader
from api.video_jobs import video_job_manager

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIDEO_JOB_ROOT = PROJECT_ROOT / "reports" / "video_jobs"
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
MAX_VIDEO_BYTES = 512 * 1024 * 1024


@router.post("/detect")
async def detect(
    file: Annotated[UploadFile, File()],
    conf: Annotated[float, Form(ge=0.01, le=1.0)] = 0.25,
):
    content = await file.read()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    detector = get_detector()
    reader = get_reader()
    if detector is None:
        raise HTTPException(status_code=503, detail="Plate model not loaded; train stage two or configure its weights")

    started = time.perf_counter()
    detector.plate_detector.conf_threshold = conf
    cascade_detections = detector.predict(image)
    annotated = _draw_cascade(image, cascade_detections)
    detections = []
    for index, cascade_detection in enumerate(cascade_detections):
        plate_box = _clip_box(cascade_detection.plate.bbox, image.shape[1], image.shape[0])
        if plate_box is None:
            continue
        x1, y1, x2, y2 = plate_box
        crop = image[y1:y2, x1:x2]
        text = getattr(cascade_detection, "character_text", "")
        text_source = "character_detector" if text else ""
        if not text and reader is not None:
            text = reader.read_plate(crop)
            text_source = "crnn"
        formatted = ""
        if text:
            from src.postprocessing.plate_formatter import format_plate

            formatted = format_plate(text)
        item = cascade_detection.to_dict()
        item.update(
            {
                "id": f"plate-{index + 1}",
                "plate_text": text,
                "formatted_text": formatted,
                "text_source": text_source or None,
                "plate_crop": _image_data_url(natural_high_resolution_plate(crop, 4)),
            }
        )
        detections.append(item)

    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "detections": detections,
        "annotated_image": _image_data_url(annotated),
        "processing_time_ms": round(elapsed, 1),
    }


@router.post("/detect/video")
async def detect_video(
    file: Annotated[UploadFile, File()],
    conf: Annotated[float, Form(ge=0.01, le=1.0)] = 0.25,
    frame_stride: Annotated[int, Form(ge=1, le=30)] = 3,
) -> dict[str, Any]:
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=422, detail="Unsupported video format")
    job_id = uuid.uuid4().hex
    job_dir = VIDEO_JOB_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / f"input{suffix}"
    output_path = job_dir / "annotated.mp4"

    size = 0
    with input_path.open("wb") as output_file:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                output_file.close()
                input_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video exceeds the 512 MB limit")
            output_file.write(chunk)
    if size == 0:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="Uploaded video is empty")

    try:
        job = video_job_manager.create(job_id, file.filename or input_path.name)
    except RuntimeError as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    worker = threading.Thread(
        target=_process_video,
        args=(job_id, input_path, output_path, conf, frame_stride),
        daemon=True,
    )
    worker.start()
    return {"job": job}


@router.get("/detect/video/{job_id}")
def video_status(job_id: str) -> dict[str, Any]:
    job = video_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video job not found")
    return {"job": job}


@router.get("/detect/video/{job_id}/result")
def video_result(job_id: str) -> FileResponse:
    job = video_job_manager.get(job_id)
    if job is None or job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Annotated video is not available")
    result_path = VIDEO_JOB_ROOT / job_id / "annotated.mp4"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Annotated video file is missing")
    return FileResponse(result_path, media_type="video/mp4", filename=f"{Path(job['filename']).stem}-annotated.mp4")


@router.get("/detect/video/{job_id}/frame/{frame_index}")
def video_frame(job_id: str, frame_index: int):
    job = video_job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video job not found")
    job_dir = VIDEO_JOB_ROOT / job_id
    input_files = list(job_dir.glob("input.*"))
    if not input_files:
        raise HTTPException(status_code=404, detail="Source video not found")
    capture = cv2.VideoCapture(str(input_files[0]))
    if not capture.isOpened():
        raise HTTPException(status_code=500, detail="Could not open source video")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise HTTPException(status_code=404, detail="Frame not found")
    detector = get_detector()
    if detector is not None:
        conf = job.get("confidence", 0.25)
        detector.plate_detector.conf_threshold = float(conf) if isinstance(conf, (int, float)) else 0.25
        detections = detector.predict(frame)
        labels = [getattr(cd, "character_text", "") for cd in detections]
        frame = _draw_cascade(frame, detections, labels)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    from fastapi.responses import Response
    return Response(content=encoded.tobytes(), media_type="image/jpeg")


def _process_video(job_id: str, input_path: Path, output_path: Path, confidence: float, frame_stride: int) -> None:
    capture = None
    writer = None
    try:
        detector = get_detector()
        reader = get_reader()
        if detector is None:
            raise RuntimeError("Plate model is not loaded")
        detector.plate_detector.conf_threshold = confidence
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError("Could not open uploaded video")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("Uploaded video has invalid dimensions")
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated video")
        video_job_manager.update(job_id, status="processing", total_frames=total_frames)

        frame_index = 0
        frames_with_detections = 0
        total_detections = 0
        events: list[dict[str, Any]] = []
        current_detections = []
        current_labels: list[str] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                current_detections = detector.predict(frame)
                current_labels = []
                for cascade_detection in current_detections:
                    plate_box = _clip_box(cascade_detection.plate.bbox, width, height)
                    if plate_box is None:
                        current_labels.append("")
                        continue
                    x1, y1, x2, y2 = plate_box
                    crop = frame[y1:y2, x1:x2]
                    label = getattr(cascade_detection, "character_text", "")
                    if not label and reader is not None:
                        label = reader.read_plate(crop)
                    current_labels.append(label)
                if current_detections:
                    frames_with_detections += 1
                    total_detections += len(current_detections)
                    if len(events) < 30:
                        events.append(
                            {
                                "frame": frame_index,
                                "time_seconds": round(frame_index / fps, 2),
                                "detections": len(current_detections),
                                "plates": [label for label in current_labels if label],
                                "frame_url": f"/api/detect/video/{job_id}/frame/{frame_index}",
                            }
                        )
            annotated = _draw_cascade(frame, current_detections, current_labels)
            writer.write(annotated)
            frame_index += 1
            if frame_index % 10 == 0 or frame_index == total_frames:
                progress = frame_index / total_frames if total_frames > 0 else 0.0
                video_job_manager.update(
                    job_id,
                    progress=min(1.0, progress),
                    processed_frames=frame_index,
                    frames_with_detections=frames_with_detections,
                    total_detections=total_detections,
                    events=events,
                )
        writer.release()
        writer = None
        video_job_manager.update(
            job_id,
            status="completed",
            progress=1.0,
            processed_frames=frame_index,
            frames_with_detections=frames_with_detections,
            total_detections=total_detections,
            events=events,
            result_url=f"/api/detect/video/{job_id}/result",
        )
    except Exception as exc:
        video_job_manager.update(job_id, status="error", error=str(exc))
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()


def _draw_cascade(image, detections, labels: list[str] | None = None):
    annotated = image.copy()
    labels = labels or []
    drawn_vehicles: set[tuple[float, float, float, float]] = set()
    for index, detection in enumerate(detections):
        if detection.vehicle.bbox not in drawn_vehicles:
            vx1, vy1, vx2, vy2 = (int(value) for value in detection.vehicle.bbox)
            cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 128, 0), 2)
            cv2.putText(
                annotated,
                f"{detection.vehicle.class_name} {detection.vehicle.confidence:.2f}",
                (vx1, max(18, vy1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 128, 0),
                2,
            )
            drawn_vehicles.add(detection.vehicle.bbox)
        px1, py1, px2, py2 = (int(value) for value in detection.plate.bbox)
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 230, 118), 2)
        label = labels[index] if index < len(labels) and labels[index] else "plate"
        cv2.putText(
            annotated,
            f"{label} {detection.plate.confidence:.2f}",
            (px1, max(18, py1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 118),
            2,
        )
    return annotated


def _clip_box(bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _image_data_url(image) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
