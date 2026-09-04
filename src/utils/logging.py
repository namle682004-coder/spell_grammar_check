"""Logging setup using loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Configure loguru: stderr + optional file sink."""
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_file), level=level, rotation="50 MB", retention="7 days",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    return logger
