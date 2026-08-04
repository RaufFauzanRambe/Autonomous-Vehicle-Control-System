"""
File:        python/telemetry.py
Brief:       Telemetry — aggregates subsystem state, serializes it to
             JSON, and publishes over MQTT with rate limiting and a
             rolling statistics buffer.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

from loguru import logger

from mqtt_client import MqttClient


# ----------------------------------------------------------------------
# Per-key statistics
# ----------------------------------------------------------------------
@dataclass(slots=True)
class NumericStat:
    """Rolling statistics for a single numeric telemetry field."""

    window: int = 60
    _samples: Deque[float] = field(default_factory=lambda: deque(maxlen=60))

    def push(self, value: float) -> None:
        if not isinstance(value, (int, float)) or value != value:  # NaN check
            return
        self._samples.append(float(value))

    def snapshot(self) -> dict:
        if not self._samples:
            return {"count": 0}
        return {
            "count": len(self._samples),
            "min": min(self._samples),
            "max": max(self._samples),
            "mean": statistics.fmean(self._samples),
            "stdev": (statistics.pstdev(self._samples)
                      if len(self._samples) > 1 else 0.0),
            "last": self._samples[-1],
        }


# ----------------------------------------------------------------------
# Telemetry module
# ----------------------------------------------------------------------
class Telemetry:
    """Aggregates and publishes telemetry data over MQTT.

    Maintains a rolling buffer of numeric samples per key, and a
    rate-limited JSON publisher. Callers push dicts via :meth:`publish`;
    the JSON payload is also queued to a local ring buffer for the
    dashboard to consume on reconnect.
    """

    DEFAULT_RATE_HZ: float = 5.0          # publish rate
    DEFAULT_BUFFER_SIZE: int = 100        # local ring buffer

    def __init__(self, config: dict,
                 mqtt_client: Optional[MqttClient] = None) -> None:
        self.rate_hz: float = float(config.get("rate_hz", self.DEFAULT_RATE_HZ))
        self.buffer_size: int = int(config.get("buffer_size",
                                               self.DEFAULT_BUFFER_SIZE))
        self.topic_prefix: str = config.get("topic_prefix", "vehicle/av-00/telemetry")
        self.qos: int = int(config.get("qos", 1))
        self.retain: bool = bool(config.get("retain", False))
        self.collect_stats: bool = bool(config.get("collect_stats", True))

        self.mqtt: Optional[MqttClient] = mqtt_client
        self._buffer: Deque[dict] = deque(maxlen=self.buffer_size)
        self._stats: Dict[str, NumericStat] = defaultdict(
            lambda: NumericStat(window=60)
        )
        self._lock = threading.RLock()
        self._last_publish: float = 0.0
        self._publish_count: int = 0
        self._drop_count: int = 0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        if self.mqtt is None:
            logger.warning("Telemetry has no MQTT client — data stays local")

    # ------------------------------------------------------------------
    # Background publisher
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background publisher thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop,
                                        name="telemetry-pub", daemon=True)
        self._thread.start()
        logger.info("Telemetry publisher started ({} Hz)", self.rate_hz)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _publish_loop(self) -> None:
        period = 1.0 / self.rate_hz
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                self._flush_one()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Telemetry publish failed: {}", exc)
            elapsed = time.monotonic() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Push telemetry data
    # ------------------------------------------------------------------
    def publish(self, data: Dict[str, Any]) -> None:
        """Push a telemetry dict into the buffer.

        The data is published on the next tick of the rate-limited
        publisher thread. Numeric fields are also fed into the
        statistics collector.
        """
        payload = {
            "ts": time.time(),
            "data": data,
        }
        with self._lock:
            self._buffer.append(payload)
            if self.collect_stats:
                self._collect_statistics(data)
        # If start() wasn't called, publish immediately (rate-limited)
        if self._thread is None:
            self._flush_one()

    def _collect_statistics(self, data: Dict[str, Any], prefix: str = "") -> None:
        """Recursively collect numeric statistics from the payload."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self._collect_statistics(value, full_key)
            elif isinstance(value, (int, float)):
                self._stats[full_key].push(value)

    # ------------------------------------------------------------------
    # Flush
    # ------------------------------------------------------------------
    def _flush_one(self) -> bool:
        """Publish the oldest buffered payload, respecting rate limit."""
        now = time.monotonic()
        if now - self._last_publish < 1.0 / self.rate_hz:
            return False
        with self._lock:
            if not self._buffer:
                return False
            payload = self._buffer.popleft()
        self._last_publish = now
        try:
            self._publish_payload(payload)
            self._publish_count += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self._drop_count += 1
            logger.warning("Telemetry publish failed ({}): {}",
                           self._drop_count, exc)
            # Re-queue at the back
            with self._lock:
                self._buffer.append(payload)
            return False

    def _publish_payload(self, payload: dict) -> None:
        """Serialize and publish one payload over MQTT."""
        if self.mqtt is None:
            return
        topic = f"{self.topic_prefix}/state"
        try:
            body = json.dumps(payload, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            logger.warning("JSON serialization failed: {}", exc)
            return
        self.mqtt.publish(topic, body, qos=self.qos, retain=self.retain)

    # ------------------------------------------------------------------
    # Statistics access
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, dict]:
        """Snapshot of all collected numeric statistics."""
        with self._lock:
            return {k: v.snapshot() for k, v in self._stats.items()}

    def reset_statistics(self) -> None:
        with self._lock:
            self._stats.clear()

    # ------------------------------------------------------------------
    # Buffer access (for the dashboard)
    # ------------------------------------------------------------------
    def snapshot(self, n: int = 10) -> list[dict]:
        """Return the last ``n`` buffered payloads."""
        with self._lock:
            return list(self._buffer)[-n:]

    def flush(self) -> int:
        """Flush all buffered payloads synchronously.

        Returns the number of payloads successfully published.
        """
        published = 0
        while True:
            if not self._flush_one():
                break
            published += 1
        return published

    # ------------------------------------------------------------------
    # Stats / cleanup
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "publish_count": self._publish_count,
            "drop_count": self._drop_count,
            "buffered": len(self._buffer),
            "rate_hz": self.rate_hz,
            "stat_keys": len(self._stats),
        }

    def close(self) -> None:
        self.stop()
        self.flush()
        logger.info("Telemetry closed (published={}, dropped={})",
                    self._publish_count, self._drop_count)
