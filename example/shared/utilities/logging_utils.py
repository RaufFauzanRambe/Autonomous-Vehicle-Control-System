"""Logging helpers shared by every example module.

One uniform format for console + optional file output so that logs from
different modules can be compared side by side in research runs.

Usage::

    from shared.utilities.logging_utils import get_logger

    logger = get_logger("01_basic_simulation", log_file="outputs/run.log")

Extension point: swap the formatting for ``structlog``/JSON logging when
integrating with larger experiment-tracking pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
_CONFIGURED: set[str] = set()


def get_logger(name: str, log_file: str | Path | None = None,
               level: str = "INFO") -> logging.Logger:
    """Return a configured logger with a uniform format.

    Calling this repeatedly with the same ``name`` is idempotent: handlers
    are only attached once per logger name.

    Args:
        name: Logger name, conventionally ``<module_folder>`` (e.g.
            ``"01_basic_simulation"``).
        log_file: Optional file path; parent directories are created
            automatically.
        level: Logging level name (``"DEBUG"``, ``"INFO"``, ...).

    Returns:
        A :class:`logging.Logger` ready to use.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if name not in _CONFIGURED:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(console)
        logger.propagate = False
        _CONFIGURED.add(name)

    if log_file is not None:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        key = f"{name}:{file_path}"
        if key not in _CONFIGURED:
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(file_handler)
            _CONFIGURED.add(key)

    return logger
