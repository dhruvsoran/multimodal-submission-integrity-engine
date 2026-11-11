"""Shared helpers: logging, ffmpeg discovery, misc utilities."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    """Console + optional file logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
        )
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_ffmpeg_exe() -> str | None:
    """Locate an ffmpeg binary.

    Tries PATH first, then falls back to the static binary bundled with the
    ``imageio-ffmpeg`` package (installed with the project requirements).
    Returns None when no binary can be found.
    """
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def truncate(text: str, limit: int = 220) -> str:
    """Trim a passage for console display."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"
