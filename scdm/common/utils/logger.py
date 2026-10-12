"""Logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(name: str, output_dir: str | Path) -> logging.Logger:
    """Create a console and file logger for one experiment run."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    while logger.handlers:
        handler = logger.handlers.pop()
        handler.close()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
