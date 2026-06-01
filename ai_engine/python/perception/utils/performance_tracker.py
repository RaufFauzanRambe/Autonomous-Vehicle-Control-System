"""
Performance tracking utility for the Autonomous Vehicle Control System.

Provides FPS measurement, latency percentile tracking, throughput
calculation, and resource usage monitoring for real-time control loops.

Usage:
    from utils.performance_tracker import PerformanceTracker

    tracker = PerformanceTracker("perception_pipeline")
    tracker.start()

    for frame in camera_stream:
        with tracker.measure("total"):
            detections = detect(frame)
            with tracker.measure("postprocess"):
                filtered = postprocess(detections)
        tracker.record_frame()

    print(tracker.summary())
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    """Track frames-per-second using a sliding window of frame timestamps.

    Args:
        window_size: Number of recent frames to consider.
    """

    def __init__(self, window_size: int = 60) -> None:
        self.window_size = window_size
        self._timestamps: Deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def tick(self) -> None:
        """Record a frame event (call once per processed frame)."""
        with self._lock:
            self._timestamps.append(time.perf_counter())

    @property
    def fps(self) -> float:
        """Current FPS estimate based on the recent window."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    @property
    def frame_count(self) -> int:
        with self._lock:
            return len(self._timestamps)

    @property
    def mean_frame_time_ms(self) -> float:
        """Average time between frames in milliseconds."""
        fps = self.fps
        return 1000.0 / fps if fps > 0 else 0.0

    def reset(self) -> None:
        with self._lock:
            self._timestamps.clear()


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Track latency measurements and compute percentiles.

    Args:
        max_samples: Maximum number of samples to retain.
    """

    def __init__(self, max_samples: int = 10000) -> None:
        self.max_samples = max_samples
        self._samples: Deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._total_count: int = 0

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement in milliseconds."""
        with self._lock:
            self._samples.append(latency_ms)
            self._total_count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._total_count

    def percentile(self, p: float) -> float:
        """Compute the *p*-th percentile (0–100) of recorded latencies."""
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_vals = sorted(self._samples)
            idx = int(len(sorted_vals) * p / 100.0)
            idx = min(idx, len(sorted_vals) - 1)
            return sorted_vals[idx]

    def statistics(self) -> Dict[str, float]:
        """Return a comprehensive statistics dict."""
        with self._lock:
            if not self._samples:
                return {"count": 0}
            vals = list(self._samples)
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            mean = sum(vals) / n
            variance = sum((v - mean) ** 2 for v in vals) / n
            return {
                "count": self._total_count,
                "current_samples": n,
                "mean_ms": round(mean, 3),
                "std_ms": round(variance ** 0.5, 3),
                "min_ms": round(sorted_vals[0], 3),
                "max_ms": round(sorted_vals[-1], 3),
                "p50_ms": round(sorted_vals[n // 2], 3),
                "p90_ms": round(sorted_vals[int(n * 0.90)], 3),
                "p95_ms": round(sorted_vals[int(n * 0.95)], 3),
                "p99_ms": round(sorted_vals[int(n * 0.99)], 3),
                "p999_ms": round(sorted_vals[min(int(n * 0.999), n - 1)], 3),
            }

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._total_count = 0


# ---------------------------------------------------------------------------
# Resource usage tracker
# ---------------------------------------------------------------------------

class ResourceTracker:
    """Track system resource usage (CPU, memory) via ``/proc`` and ``os``."""

    def __init__(self) -> None:
        self._last_cpu_time: Optional[float] = None
        self._last_proc_time: Optional[float] = None
        self._cpu_percent: float = 0.0

    def update(self) -> Dict[str, Any]:
        """Sample current resource usage. Call periodically.

        Returns:
            Dict with CPU percent, RSS memory, and thread count.
        """
        # CPU usage (process-level)
        try:
            import resource as res_module
            usage = res_module.getrusage(res_module.RUSAGE_SELF)
            proc_time = usage.ru_utime + usage.ru_stime
        except (ImportError, AttributeError):
            proc_time = time.process_time()

        now = time.perf_counter()
        if self._last_cpu_time is not None and self._last_proc_time is not None:
            elapsed = now - self._last_cpu_time
            if elapsed > 0:
                self._cpu_percent = ((proc_time - self._last_proc_time) / elapsed) * 100.0

        self._last_cpu_time = now
        self._last_proc_time = proc_time

        # Memory (RSS in MB)
        memory_rss_mb = self._get_memory_mb()

        # Thread count
        thread_count = threading.active_count()

        return {
            "cpu_percent": round(self._cpu_percent, 1),
            "memory_rss_mb": round(memory_rss_mb, 1),
            "thread_count": thread_count,
        }

    def _get_memory_mb(self) -> float:
        """Get process RSS memory in MB from /proc/self/status."""
        try:
            with open("/proc/self/status", "r") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        # Value is in kB
                        return int(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return 0.0


# ---------------------------------------------------------------------------
# Throughput tracker
# ---------------------------------------------------------------------------

class ThroughputTracker:
    """Track throughput of items processed per unit time.

    Useful for monitoring data ingestion rates, message processing rates, etc.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()
        self._total_items: int = 0

    def record(self, count: int = 1) -> None:
        """Record *count* items processed at the current time."""
        now = time.monotonic()
        with self._lock:
            self._events.append(now)
            self._total_items += count

    @property
    def throughput(self) -> float:
        """Items per second over the recent window."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self.window_seconds
            while self._events and self._events[0] < cutoff:
                self._events.popleft()
            if not self._events:
                return 0.0
            elapsed = now - self._events[0]
            return len(self._events) / elapsed if elapsed > 0 else 0.0

    @property
    def total_items(self) -> int:
        with self._lock:
            return self._total_items

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._total_items = 0


# ---------------------------------------------------------------------------
# PerformanceTracker – unified interface
# ---------------------------------------------------------------------------

class PerformanceTracker:
    """Unified performance tracker combining FPS, latency, throughput,
    and resource monitoring.

    Args:
        name: Identifier for this tracker (e.g., subsystem name).
        fps_window: Number of frames for FPS calculation.
        latency_max_samples: Max latency samples to retain.
    """

    def __init__(
        self,
        name: str,
        fps_window: int = 60,
        latency_max_samples: int = 10000,
    ) -> None:
        self.name = name
        self.fps = FPSTracker(window_size=fps_window)
        self._latency_trackers: Dict[str, LatencyTracker] = {}
        self._latency_max_samples = latency_max_samples
        self.resource = ResourceTracker()
        self.throughput = ThroughputTracker()

        self._start_time: Optional[float] = None
        self._active_spans: Dict[str, float] = {}
        self._lock = threading.Lock()

    # -- Lifecycle --

    def start(self) -> None:
        """Mark the start of tracking."""
        self._start_time = time.perf_counter()

    @property
    def uptime(self) -> float:
        """Seconds since :meth:`start` was called."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    # -- Frame recording --

    def record_frame(self) -> None:
        """Record a processed frame (for FPS tracking)."""
        self.fps.tick()
        self.throughput.record()

    # -- Latency measurement --

    def latency_tracker(self, label: str) -> LatencyTracker:
        """Get or create a named latency tracker."""
        if label not in self._latency_trackers:
            self._latency_trackers[label] = LatencyTracker(max_samples=self._latency_max_samples)
        return self._latency_trackers[label]

    def record_latency(self, label: str, latency_ms: float) -> None:
        """Record a latency measurement under *label*."""
        self.latency_tracker(label).record(latency_ms)

    def start_span(self, label: str) -> None:
        """Start a named timing span."""
        self._active_spans[label] = time.perf_counter()

    def end_span(self, label: str) -> float:
        """End a timing span and record the latency.

        Returns:
            Elapsed milliseconds.
        """
        start = self._active_spans.pop(label, None)
        if start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.record_latency(label, elapsed_ms)
        return elapsed_ms

    def measure(self, label: str):
        """Context manager for timing a code block.

        Usage::

            with tracker.measure("inference"):
                result = model(input)
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self.start_span(label)
            try:
                yield
            finally:
                self.end_span(label)

        return _ctx()

    # -- Summary --

    def summary(self) -> Dict[str, Any]:
        """Generate a comprehensive performance summary."""
        result: Dict[str, Any] = {
            "name": self.name,
            "uptime_seconds": round(self.uptime, 1),
            "fps": round(self.fps.fps, 2),
            "mean_frame_time_ms": round(self.fps.mean_frame_time_ms, 2),
            "total_frames": self.fps.frame_count,
            "throughput_per_second": round(self.throughput.throughput, 2),
            "total_items_processed": self.throughput.total_items,
        }

        # Latency summaries
        latencies: Dict[str, Dict[str, float]] = {}
        for label, tracker in self._latency_trackers.items():
            latencies[label] = tracker.statistics()
        if latencies:
            result["latencies"] = latencies

        # Resource usage
        result["resource"] = self.resource.update()

        return result

    def latency_summary(self, label: str) -> Dict[str, float]:
        """Get latency statistics for a specific label."""
        if label in self._latency_trackers:
            return self._latency_trackers[label].statistics()
        return {"count": 0}
