"""Tests for annotated image responses and asynchronous video job setup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from api.main import app
from api.routes import detection
from api.video_jobs import VideoJobManager
from fastapi.testclient import TestClient
from src.detection.inference import DetectionResult, TwoStageDetection


class FakeCascadeDetector:
    def __init__(self) -> None:
        self.plate_detector = SimpleNamespace(conf_threshold=0.25)

    def predict(self, _image):
        return [
            TwoStageDetection(
                vehicle=DetectionResult((2, 2, 38, 28), 0.9, 2, "car"),
                plate=DetectionResult((10, 12, 30, 22), 0.8, 0, "license_plate"),
                plate_bbox_in_vehicle=(8, 10, 28, 20),
            )
        ]


class FakeReader:
    def read_plate(self, _crop):
        return "ABC123"


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_image_detection_returns_annotated_image_and_crop(monkeypatch) -> None:
    monkeypatch.setattr(detection, "get_detector", lambda: FakeCascadeDetector())
    monkeypatch.setattr(detection, "get_reader", lambda: FakeReader())
    client = TestClient(app)

    response = client.post(
        "/api/detect",
        files={"file": ("car.jpg", _jpeg_bytes(), "image/jpeg")},
        data={"conf": "0.25"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")
    assert payload["detections"][0]["plate_crop"].startswith("data:image/jpeg;base64,")
    assert payload["detections"][0]["vehicle"]["class_name"] == "car"
    assert payload["detections"][0]["plate_text"] == "ABC123"


def test_video_upload_creates_pollable_job(tmp_path: Path, monkeypatch) -> None:
    manager = VideoJobManager()
    monkeypatch.setattr(detection, "VIDEO_JOB_ROOT", tmp_path)
    monkeypatch.setattr(detection, "video_job_manager", manager)

    class FakeThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(detection.threading, "Thread", FakeThread)
    client = TestClient(app)

    response = client.post(
        "/api/detect/video",
        files={"file": ("traffic.mp4", b"not-a-real-video", "video/mp4")},
        data={"conf": "0.25", "frame_stride": "3"},
    )

    assert response.status_code == 200
    job = response.json()["job"]
    assert job["status"] == "queued"
    status = client.get(f"/api/detect/video/{job['id']}")
    assert status.status_code == 200
    assert status.json()["job"]["filename"] == "traffic.mp4"


def test_video_job_manager_allows_only_one_active_job() -> None:
    manager = VideoJobManager()

    manager.create("first", "first.mp4")

    try:
        manager.create("second", "second.mp4")
    except RuntimeError as exc:
        assert "already" in str(exc)
    else:
        raise AssertionError("Expected a second active video job to be rejected")


def test_video_worker_produces_annotated_output(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "input.avi"
    output_path = tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (48, 32))
    assert writer.isOpened()
    for _ in range(6):
        writer.write(np.zeros((32, 48, 3), dtype=np.uint8))
    writer.release()

    manager = VideoJobManager()
    manager.create("video-job", "input.avi")
    monkeypatch.setattr(detection, "video_job_manager", manager)
    monkeypatch.setattr(detection, "get_detector", lambda: FakeCascadeDetector())
    monkeypatch.setattr(detection, "get_reader", lambda: FakeReader())

    detection._process_video("video-job", input_path, output_path, 0.25, 2)

    job = manager.get("video-job")
    assert job is not None
    assert job["status"] == "completed"
    assert job["processed_frames"] == 6
    assert job["frames_with_detections"] == 3
    assert job["total_detections"] == 3
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
