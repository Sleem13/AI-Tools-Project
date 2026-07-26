"""Single-process manager for local Ultralytics training jobs."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any


class TrainingJobManager:
    """Launch and observe at most one training process per API instance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._state: dict[str, Any] = {
            "status": "idle",
            "run_name": None,
            "run_dir": None,
            "log_path": None,
            "epochs": 0,
            "return_code": None,
        }

    def start(
        self,
        command: list[str],
        cwd: Path,
        run_name: str,
        run_dir: Path,
        log_path: Path,
        epochs: int,
    ) -> dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("A training job is already running")
            self._close_log()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = log_path.open("w", encoding="utf-8")
            creation_flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self._process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
            )
            self._state = {
                "status": "training",
                "run_name": run_name,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "epochs": epochs,
                "return_code": None,
            }
            snapshot = dict(self._state)
        snapshot["log_tail"] = ""
        return snapshot

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._process is not None:
                return_code = self._process.poll()
                if return_code is not None and self._state["status"] == "training":
                    self._state["return_code"] = return_code
                    self._state["status"] = "completed" if return_code == 0 else "error"
                    self._close_log()
            snapshot = dict(self._state)
        snapshot["log_tail"] = self._read_log_tail(snapshot.get("log_path"))
        return snapshot

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    @staticmethod
    def _read_log_tail(log_path: str | None, max_chars: int = 6000) -> str:
        if not log_path:
            return ""
        path = Path(log_path)
        if not path.is_file():
            return ""
        with path.open(encoding="utf-8", errors="replace") as log_file:
            content = log_file.read()
        return content[-max_chars:]


training_job_manager = TrainingJobManager()
