"""In-memory state manager for asynchronous local video inference."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from typing import Any


class VideoJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, job_id: str, filename: str) -> dict[str, Any]:
        with self._lock:
            if any(job["status"] in {"queued", "processing"} for job in self._jobs.values()):
                raise RuntimeError("Another video is already being processed")
            self._jobs[job_id] = {
                "id": job_id,
                "filename": filename,
                "status": "queued",
                "progress": 0.0,
                "processed_frames": 0,
                "total_frames": 0,
                "frames_with_detections": 0,
                "total_detections": 0,
                "events": [],
                "error": None,
                "result_url": None,
                "created_at": datetime.now(UTC).isoformat(),
            }
            self._prune()
            return dict(self._jobs[job_id])

    def update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            self._jobs[job_id].update(changes)
            return dict(self._jobs[job_id])

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _prune(self) -> None:
        completed = [job_id for job_id, job in self._jobs.items() if job["status"] in {"completed", "error"}]
        for job_id in completed[:-9]:
            self._jobs.pop(job_id, None)

    def cleanup(self) -> None:
        """Mark all active jobs as errored on server shutdown."""
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in {"queued", "processing"}:
                    job["status"] = "error"
                    job["error"] = "Server shutting down"


video_job_manager = VideoJobManager()
