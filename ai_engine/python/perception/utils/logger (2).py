"""
Logging utility for the Autonomous Vehicle Control System.

Provides colored console logging, file rotation, structured JSON logging,
and a dedicated performance logger for timing-critical subsystems.

Usage:
    from utils.logger import get_logger, get_perf_logger

    logger = get_logger("perception")
    logger.info("Object detected", extra={"object_type": "vehicle", "distance": 12.4})

    perf_logger = get_perf_logger("inference")
    perf_logger.timer("model_forward", 0.032)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Union

# ---------------------------------------------------------------------------
# Colour definitions for ANSI-compatible terminals
# ---------------------------------------------------------------------------

class AnsiColors:
    """ANSI escape code constants for coloured terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"


# Map logging levels to colours
_LEVEL_COLOUR_MAP: Dict[int, str] = {
    logging.DEBUG: AnsiColors.CYAN,
    logging.INFO: AnsiColors.GREEN,
    logging.WARNING: AnsiColors.YELLOW,
    logging.ERROR: AnsiColors.RED,
    logging.CRITICAL: AnsiColors.BG_RED + AnsiColors.WHITE + AnsiColors.BOLD,
}


# ---------------------------------------------------------------------------
# Custom formatters
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    """Formatter that injects ANSI colour codes based on log level."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        use_color: bool = True,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            colour = _LEVEL_COLOUR_MAP.get(record.levelno, AnsiColors.RESET)
            record.levelname = f"{colour}{record.levelname:<8}{AnsiColors.RESET}"
            record.name = f"{AnsiColors.MAGENTA}{record.name}{AnsiColors.RESET}"
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter suitable for machine parsing and log aggregators."""

    def __init__(
        self,
        include_extras: bool = True,
        timestamp_key: str = "timestamp",
        level_key: str = "level",
        message_key: str = "message",
        logger_key: str = "logger",
    ) -> None:
        super().__init__()
        self.include_extras = include_extras
        self.timestamp_key = timestamp_key
        self.level_key = level_key
        self.message_key = message_key
        self.logger_key = logger_key

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            self.timestamp_key: datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            self.level_key: record.levelname,
            self.logger_key: record.name,
            self.message_key: record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        if self.include_extras:
            # Capture any extra fields passed via logger.info(..., extra={})
            standard_attrs = set(
                logging.LogRecord(
                    "", 0, "", 0, "", (), None
                ).__dict__.keys()
            )
            extras = {
                k: v
                for k, v in record.__dict__.items()
                if k not in standard_attrs and not k.startswith("_")
            }
            if extras:
                log_entry["extras"] = extras

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        try:
            return json.dumps(log_entry, default=_json_default, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps(
                {self.message_key: record.getMessage()}, default=str
            )


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for non-standard types in JSON logs."""
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def _make_console_handler(
    level: int = logging.DEBUG,
    use_color: bool = True,
    stream: TextIO = sys.stderr,
) -> logging.StreamHandler:
    """Create a coloured (or plain) console stream handler."""
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    fmt = "%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(ColoredFormatter(fmt=fmt, datefmt=datefmt, use_color=use_color))
    return handler


def _make_file_handler(
    path: Union[str, Path],
    level: int = logging.DEBUG,
    max_bytes: int = 50 * 1024 * 1024,  # 50 MB
    backup_count: int = 10,
    encoding: str = "utf-8",
) -> RotatingFileHandler:
    """Create a rotating file handler (size-based rotation)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding=encoding,
    )
    handler.setLevel(level)
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


def _make_timed_file_handler(
    path: Union[str, Path],
    level: int = logging.DEBUG,
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 30,
    encoding: str = "utf-8",
) -> TimedRotatingFileHandler:
    """Create a time-based rotating file handler (daily rotation by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        str(path),
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding=encoding,
    )
    handler.setLevel(level)
    fmt = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    return handler


def _make_json_file_handler(
    path: Union[str, Path],
    level: int = logging.DEBUG,
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 5,
    encoding: str = "utf-8",
) -> RotatingFileHandler:
    """Create a rotating file handler that writes JSON-structured log entries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding=encoding,
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    return handler


# ---------------------------------------------------------------------------
# Logger manager – singleton-like registry
# ---------------------------------------------------------------------------

_logger_cache: Dict[str, logging.Logger] = {}
_perf_logger_cache: Dict[str, "PerformanceLogger"] = {}

# Defaults (can be overridden via environment variables)
_DEFAULT_LOG_DIR = os.environ.get(
    "AV_LOG_DIR", "/var/log/autonomous_vehicle"
)
_DEFAULT_CONSOLE_LEVEL = os.environ.get(
    "AV_CONSOLE_LOG_LEVEL", "INFO"
).upper()
_DEFAULT_FILE_LEVEL = os.environ.get(
    "AV_FILE_LOG_LEVEL", "DEBUG"
).upper()


def get_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    console_level: Optional[Union[str, int]] = None,
    file_level: Optional[Union[str, int]] = None,
    enable_json: bool = True,
    enable_file: bool = True,
    enable_timed: bool = False,
) -> logging.Logger:
    """Retrieve or create a configured logger.

    Args:
        name: Logger name (typically module or subsystem identifier).
        log_dir: Directory for log files. Defaults to ``AV_LOG_DIR`` env var
            or ``/var/log/autonomous_vehicle``.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.
        enable_json: Whether to also create a JSON-formatted log file.
        enable_file: Whether to create a plain-text rotating log file.
        enable_timed: If True, use time-based rotation instead of size-based.

    Returns:
        A fully configured :class:`logging.Logger` instance.
    """
    if name in _logger_cache:
        return _logger_cache[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # handlers control actual granularity

    # Resolve levels
    _console_lvl = (
        logging.getLevelName(console_level)
        if isinstance(console_level, str)
        else console_level
    ) if console_level is not None else logging.getLevelName(_DEFAULT_CONSOLE_LEVEL)

    _file_lvl = (
        logging.getLevelName(file_level)
        if isinstance(file_level, str)
        else file_level
    ) if file_level is not None else logging.getLevelName(_DEFAULT_FILE_LEVEL)

    # Console handler
    use_color = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    logger.addHandler(_make_console_handler(level=_console_lvl, use_color=use_color))

    # File handlers
    _log_dir = Path(log_dir) if log_dir else Path(_DEFAULT_LOG_DIR)
    if enable_file:
        if enable_timed:
            logger.addHandler(
                _make_timed_file_handler(
                    _log_dir / f"{name}.log", level=_file_lvl
                )
            )
        else:
            logger.addHandler(
                _make_file_handler(
                    _log_dir / f"{name}.log", level=_file_lvl
                )
            )
    if enable_json:
        logger.addHandler(
            _make_json_file_handler(
                _log_dir / f"{name}.json.log", level=_file_lvl
            )
        )

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    _logger_cache[name] = logger
    return logger


# ---------------------------------------------------------------------------
# Performance logger
# ---------------------------------------------------------------------------

class PerformanceLogger:
    """Dedicated logger for performance / timing measurements.

    Captures wall-clock durations, counts, and custom metrics, then
    periodically flushes a summary to the standard logging pipeline.

    Example::

        perf = get_perf_logger("perception")
        with perf.measure("lane_detection"):
            result = detect_lanes(image)
        perf.increment("frames_processed")
        perf.flush()
    """

    def __init__(
        self,
        name: str,
        flush_interval: float = 10.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.name = name
        self._logger = logger or get_logger(f"perf.{name}")
        self._flush_interval = flush_interval
        self._last_flush: float = time.monotonic()

        self._timers: Dict[str, list] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._active_spans: Dict[str, float] = {}

    # -- Timers --

    def timer(self, label: str, duration: float) -> None:
        """Record a timing measurement in seconds."""
        self._timers.setdefault(label, []).append(duration)

    def start_timer(self, label: str) -> None:
        """Start a named timer span."""
        self._active_spans[label] = time.perf_counter()

    def stop_timer(self, label: str) -> float:
        """Stop a previously started timer span and record the duration.

        Returns:
            The elapsed wall-clock seconds.

        Raises:
            KeyError: If no timer with *label* was started.
        """
        start = self._active_spans.pop(label, None)
        if start is None:
            raise KeyError(f"No active timer for label '{label}'")
        duration = time.perf_counter() - start
        self.timer(label, duration)
        self._maybe_flush()
        return duration

    def measure(self, label: str):
        """Context manager / decorator for timing a code block.

        Usage::

            with perf.measure("inference"):
                output = model(input_tensor)
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self.start_timer(label)
            try:
                yield
            finally:
                self.stop_timer(label)

        return _ctx()

    # -- Counters --

    def increment(self, label: str, value: int = 1) -> None:
        """Increment a named counter."""
        self._counters[label] = self._counters.get(label, 0) + value

    # -- Gauges --

    def gauge(self, label: str, value: float) -> None:
        """Set a named gauge to an arbitrary value."""
        self._gauges[label] = value

    # -- Flush --

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> Dict[str, Any]:
        """Flush all collected metrics to the logger and return a summary dict."""
        summary: Dict[str, Any] = {}

        for label, values in self._timers.items():
            if not values:
                continue
            stats = {
                "count": len(values),
                "total": sum(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            stats["p50"] = sorted_vals[n // 2]
            stats["p95"] = sorted_vals[int(n * 0.95)]
            stats["p99"] = sorted_vals[int(n * 0.99)]
            summary[label] = stats
            self._logger.info(
                "Timer stats",
                extra={"metric_type": "timer", "label": label, **stats},
            )

        for label, count in self._counters.items():
            summary[f"counter:{label}"] = count
            self._logger.info(
                "Counter",
                extra={"metric_type": "counter", "label": label, "count": count},
            )

        for label, value in self._gauges.items():
            summary[f"gauge:{label}"] = value
            self._logger.info(
                "Gauge",
                extra={"metric_type": "gauge", "label": label, "value": value},
            )

        # Reset collections
        self._timers.clear()
        self._counters.clear()
        self._gauges.clear()
        self._last_flush = time.monotonic()
        return summary

    def get_summary(self) -> Dict[str, Any]:
        """Return the current metrics without flushing."""
        result: Dict[str, Any] = {}
        for label, values in self._timers.items():
            if values:
                result[label] = {
                    "count": len(values),
                    "total": sum(values),
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        result.update({f"counter:{k}": v for k, v in self._counters.items()})
        result.update({f"gauge:{k}": v for k, v in self._gauges.items()})
        return result


def get_perf_logger(
    name: str,
    flush_interval: float = 10.0,
) -> PerformanceLogger:
    """Retrieve or create a :class:`PerformanceLogger` instance."""
    if name not in _perf_logger_cache:
        _perf_logger_cache[name] = PerformanceLogger(
            name=name, flush_interval=flush_interval
        )
    return _perf_logger_cache[name]


# ---------------------------------------------------------------------------
# Convenience: module-level helpers
# ---------------------------------------------------------------------------

def configure_root_logger(level: Union[str, int] = "WARNING") -> None:
    """Set the root logger level to suppress noisy third-party output."""
    logging.getLogger().setLevel(
        level if isinstance(level, int) else logging.getLevelName(level)
    )


def silence_logger(name: str) -> None:
    """Silence a specific logger (e.g., noisy libraries)."""
    logging.getLogger(name).setLevel(logging.CRITICAL + 1)


def list_active_loggers() -> Dict[str, int]:
    """Return a mapping of logger name → effective level for all known loggers."""
    return {
        name: logging.getLogger(name).getEffectiveLevel()
        for name in logging.root.manager.loggerDict  # type: ignore[attr-defined]
    }
