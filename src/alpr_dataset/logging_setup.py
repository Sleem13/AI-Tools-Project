"""Centralized logging configuration for the ALPR dataset pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(
    log_dir: Path,
    name: str = "alpr_dataset",
    level: int = logging.INFO,
    log_filename: str | None = None,
) -> logging.Logger:
    """Configure a logger that writes to both console (rich) and a log file.

    Args:
        log_dir: Directory where the log file will be written. Created if missing.
        name: Logger name / namespace.
        level: Logging level.
        log_filename: Optional explicit log file name. Defaults to f"{name}.log".

    Returns:
        Configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging is called more than once.
        return logger

    console_handler = RichHandler(rich_tracebacks=True, show_path=False)
    console_handler.setLevel(level)

    file_path = log_dir / (log_filename or f"{name}.log")
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
