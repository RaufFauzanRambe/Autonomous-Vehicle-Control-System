"""
Latency Monitor Module for Autonomous Vehicle AI.

Comprehensive latency monitoring and SLA enforcement:
- Percentile statistics (P50, P95, P99, P999)
- Jitter detection and analysis
- SLA tracking with configurable thresholds
- Alerting on performance degradation
- Rolling window statistics with multiple time horizons
- Latency budget tracking across inference pipeline stages

Designed for real-time autonomous driving systems where sub-millisecond
latency tracking is safety-critical.
"""

import time
import threading
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DegradationType(Enum):
    """Types of performance degradation."""
    LATENCY_SPIKE = "latency_spike"
    SUSTAINED_DEGRADATION = "sustained_degradation"
    JITTER_ANOMALY = "jitter_anomaly"
    SLA_BREACH = "sla_breach"
    TREND_DEGRADATION = "trend_degradation"


@dataclass
class LatencySLA:
    """SLA definition for latency thresholds."""
    name: str = "default"
    p50_max_ms: float = 15.0
    p95_max_ms: float = 30.0
    p99_max_ms: float = 50.0
    p999_max_ms: float = 100.0
    max_latency_ms: float = 200.0  # Hard maximum
    max_jitter_ms: float = 10.0
    min_throughput_fps: float = 10.0


@dataclass
class LatencyAlert:
    """An alert triggered by latency degradation."""
    timestamp: float = 0.0
    severity: AlertSeverity = AlertSeverity.WARNING
    degradation_type: DegradationType = DegradationType.LATENCY_SPIKE
    message: str = ""
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class StageLatency:
    """Latency tracking for a pipeline stage."""
    name: str = ""
    latency_ms: float = 0.0
    timestamp: float = 0.0


class RollingWindowStats:
    """
    Rolling window statistics for latency tracking.

    Maintains multiple time horizons (1s, 5s, 30s, 60s) for
    real-time percentile computation without storing all history.
    Uses a deque-based circular buffer per window.
    """

    def __init__(
        self,
        window_sizes: Optional[List[float]] = None,
        max_samples: int = 10000,
    ) -> None:
        """
        Args:
            window_sizes: List of window durations in seconds.
            max_samples: Maximum number of samples to retain.
        """
        self._window_sizes = window_sizes or [1.0, 5.0, 30.0, 60.0]
        self._max_samples = max_samples
        self._samples: deque = deque(maxlen=max_samples)
        self._timestamps: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def record(self, value: float, timestamp: Optional[float] = None) -> None:
        """
        Record a latency sample.

        Args:
            value: Latency value in milliseconds.
            timestamp: Optional timestamp (defaults to now).
        """
        ts = timestamp or time.time()
        with self._lock:
            self._samples.append(value)
            self._timestamps.append(ts)

    def get_stats(self, window_sec: float = 5.0) -> Dict[str, float]:
        """
        Compute statistics over a rolling time window.

        Args:
            window_sec: Window duration in seconds.

        Returns:
            Dictionary of percentile and aggregate statistics.
        """
        with self._lock:
            if not self._samples:
                return self._empty_stats()

            now = time.time()
            cutoff = now - window_sec

            # Filter samples within window
            filtered = []
            for ts, val in zip(self._timestamps, self._samples):
                if ts >= cutoff:
                    filtered.append(val)

            if not filtered:
                return self._empty_stats()

            arr = np.array(filtered)
            return {
                "count": len(arr),
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "p50_ms": float(np.percentile(arr, 50)),
                "p90_ms": float(np.percentile(arr, 90)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "p999_ms": float(np.percentile(arr, 99.9)),
                "window_sec": window_sec,
            }

    def get_all_windows(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all configured window sizes."""
        result = {}
        for ws in self._window_sizes:
            key = f"window_{ws}s"
            result[key] = self.get_stats(ws)
        return result

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        """Return empty statistics dictionary."""
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "p999_ms": 0.0,
            "window_sec": 0.0,
        }


class JitterDetector:
    """
    Jitter detection and analysis for latency measurements.

    Detects latency jitter using multiple techniques:
    - Inter-arrival time variance
    - Running coefficient of variation
    - Sustained jitter episodes
    """

    def __init__(
        self,
        jitter_threshold_ms: float = 10.0,
        cv_threshold: float = 0.3,
        episode_window: int = 20,
    ) -> None:
        self._jitter_threshold_ms = jitter_threshold_ms
        self._cv_threshold = cv_threshold
        self._episode_window = episode_window
        self._prev_latency: Optional[float] = None
        self._jitter_history: deque = deque(maxlen=1000)
        self._recent_latencies: deque = deque(maxlen=episode_window)
        self._jitter_count = 0
        self._total_samples = 0

    def check(self, latency_ms: float) -> Optional[LatencyAlert]:
        """
        Check a latency sample for jitter.

        Args:
            latency_ms: Current latency measurement.

        Returns:
            Alert if jitter detected, None otherwise.
        """
        self._total_samples += 1
        self._recent_latencies.append(latency_ms)

        alert = None

        # Check for sudden spike (delta jitter)
        if self._prev_latency is not None:
            delta = abs(latency_ms - self._prev_latency)
            self._jitter_history.append(delta)

            if delta > self._jitter_threshold_ms:
                self._jitter_count += 1
                alert = LatencyAlert(
                    severity=AlertSeverity.WARNING,
                    degradation_type=DegradationType.JITTER_ANOMALY,
                    message=f"Latency jitter spike: {delta:.2f} ms delta "
                            f"(threshold: {self._jitter_threshold_ms:.1f} ms)",
                    metric_name="jitter_delta",
                    current_value=delta,
                    threshold=self._jitter_threshold_ms,
                )

        # Check for sustained high coefficient of variation
        if len(self._recent_latencies) >= self._episode_window:
            arr = np.array(self._recent_latencies)
            mean = np.mean(arr)
            if mean > 0:
                cv = np.std(arr) / mean
                if cv > self._cv_threshold:
                    if alert is None:
                        alert = LatencyAlert(
                            severity=AlertSeverity.WARNING,
                            degradation_type=DegradationType.JITTER_ANOMALY,
                            message=f"Sustained jitter detected: CV={cv:.3f} "
                                    f"(threshold: {self._cv_threshold:.3f})",
                            metric_name="jitter_cv",
                            current_value=cv,
                            threshold=self._cv_threshold,
                        )

        self._prev_latency = latency_ms
        return alert

    @property
    def jitter_rate(self) -> float:
        """Rate of jitter occurrences."""
        return self._jitter_count / max(self._total_samples, 1)

    @property
    def avg_jitter_ms(self) -> float:
        """Average jitter magnitude."""
        if not self._jitter_history:
            return 0.0
        return float(np.mean(list(self._jitter_history)))


class SLATracker:
    """
    SLA compliance tracker for latency requirements.

    Monitors latency against defined SLA thresholds and generates
    alerts when SLAs are breached. Tracks compliance percentage
    over time for reporting and safety certification.
    """

    def __init__(self, sla: LatencySLA) -> None:
        self._sla = sla
        self._total_checks = 0
        self._breaches: Dict[str, int] = {
            "p50": 0, "p95": 0, "p99": 0, "p999": 0, "max": 0, "jitter": 0,
        }
        self._compliance_history: deque = deque(maxlen=1000)
        self._last_breach_time: float = 0.0
        self._consecutive_breaches = 0

    def check(self, stats: Dict[str, float], jitter_ms: float = 0.0) -> List[LatencyAlert]:
        """
        Check current statistics against SLA thresholds.

        Args:
            stats: Current latency statistics dictionary.
            jitter_ms: Current jitter measurement.

        Returns:
            List of alerts for any SLA breaches.
        """
        self._total_checks += 1
        alerts = []

        # Check each percentile threshold
        checks = [
            ("p50", stats.get("p50_ms", 0.0), self._sla.p50_max_ms),
            ("p95", stats.get("p95_ms", 0.0), self._sla.p95_max_ms),
            ("p99", stats.get("p99_ms", 0.0), self._sla.p99_max_ms),
            ("p999", stats.get("p999_ms", 0.0), self._sla.p999_max_ms),
            ("max", stats.get("max_ms", 0.0), self._sla.max_latency_ms),
        ]

        for name, value, threshold in checks:
            if value > threshold:
                self._breaches[name] += 1
                severity = AlertSeverity.CRITICAL if name in ("p99", "max") else AlertSeverity.WARNING
                alerts.append(LatencyAlert(
                    severity=severity,
                    degradation_type=DegradationType.SLA_BREACH,
                    message=f"SLA breach on {name}: {value:.2f} ms > {threshold:.2f} ms",
                    metric_name=f"sla_{name}",
                    current_value=value,
                    threshold=threshold,
                ))

        # Check jitter SLA
        if jitter_ms > self._sla.max_jitter_ms:
            self._breaches["jitter"] += 1
            alerts.append(LatencyAlert(
                severity=AlertSeverity.WARNING,
                degradation_type=DegradationType.JITTER_ANOMALY,
                message=f"Jitter SLA breach: {jitter_ms:.2f} ms > {self._sla.max_jitter_ms:.2f} ms",
                metric_name="sla_jitter",
                current_value=jitter_ms,
                threshold=self._sla.max_jitter_ms,
            ))

        # Track compliance
        is_compliant = len(alerts) == 0
        self._compliance_history.append(is_compliant)

        if not is_compliant:
            self._last_breach_time = time.time()
            self._consecutive_breaches += 1
        else:
            self._consecutive_breaches = 0

        # Alert on sustained degradation
        if self._consecutive_breaches >= 5:
            alerts.append(LatencyAlert(
                severity=AlertSeverity.CRITICAL,
                degradation_type=DegradationType.SUSTAINED_DEGRADATION,
                message=f"Sustained SLA degradation: {self._consecutive_breaches} consecutive breaches",
                metric_name="consecutive_breaches",
                current_value=float(self._consecutive_breaches),
                threshold=5.0,
            ))

        return alerts

    @property
    def compliance_rate(self) -> float:
        """Overall SLA compliance rate (0.0 - 1.0)."""
        if not self._compliance_history:
            return 1.0
        return sum(1 for c in self._compliance_history if c) / len(self._compliance_history)

    @property
    def total_breaches(self) -> int:
        """Total number of SLA breaches."""
        return sum(self._breaches.values())

    def get_report(self) -> Dict[str, Any]:
        """Generate an SLA compliance report."""
        return {
            "sla_name": self._sla.name,
            "total_checks": self._total_checks,
            "total_breaches": self.total_breaches,
            "compliance_rate": self.compliance_rate,
            "breach_breakdown": dict(self._breaches),
            "consecutive_breaches": self._consecutive_breaches,
            "last_breach_time": self._last_breach_time,
        }


class LatencyMonitor:
    """
    Unified latency monitor for autonomous vehicle inference pipeline.

    Combines rolling window statistics, jitter detection, and SLA
    tracking into a single monitoring interface. Supports pipeline
    stage-level latency budgeting and configurable alerting.

    Example:
        >>> monitor = LatencyMonitor(LatencySLA(p99_max_ms=50.0))
        >>> with monitor.track("inference"):
        ...     result = model(input_data)
        >>> stats = monitor.get_statistics()
    """

    def __init__(
        self,
        sla: Optional[LatencySLA] = None,
        alert_callback: Optional[Callable[[LatencyAlert], None]] = None,
        window_sizes: Optional[List[float]] = None,
    ) -> None:
        self._sla = sla or LatencySLA()
        self._alert_callback = alert_callback
        self._rolling_stats = RollingWindowStats(window_sizes=window_sizes)
        self._jitter_detector = JitterDetector(
            jitter_threshold_ms=self._sla.max_jitter_ms
        )
        self._sla_tracker = SLATracker(self._sla)
        self._alerts: deque = deque(maxlen=1000)
        self._stage_latencies: Dict[str, float] = {}
        self._stage_timestamps: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._total_inferences = 0

    def record(self, latency_ms: float, stage: Optional[str] = None) -> None:
        """
        Record a latency measurement.

        Args:
            latency_ms: Measured latency in milliseconds.
            stage: Optional pipeline stage name.
        """
        with self._lock:
            self._total_inferences += 1

            # Record to rolling stats
            self._rolling_stats.record(latency_ms)

            # Check jitter
            jitter_alert = self._jitter_detector.check(latency_ms)
            if jitter_alert:
                self._fire_alert(jitter_alert)

            # Track stage latency
            if stage:
                self._stage_latencies[stage] = latency_ms

            # Periodic SLA check (every 10 inferences)
            if self._total_inferences % 10 == 0:
                stats = self._rolling_stats.get_stats(5.0)
                sla_alerts = self._sla_tracker.check(
                    stats, self._jitter_detector.avg_jitter_ms
                )
                for alert in sla_alerts:
                    self._fire_alert(alert)

    def track(self, stage_name: str) -> "LatencyContext":
        """
        Create a context manager for tracking a pipeline stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            Context manager that records latency on exit.
        """
        return LatencyContext(self, stage_name)

    def record_stage(self, stages: List[StageLatency]) -> None:
        """
        Record latency for multiple pipeline stages.

        Args:
            stages: List of stage latency measurements.
        """
        total = sum(s.latency_ms for s in stages)
        for stage in stages:
            self._stage_latencies[stage.name] = stage.latency_ms
            self._stage_timestamps[stage.name] = stage.timestamp
        self.record(total, stage="total_pipeline")

    def _fire_alert(self, alert: LatencyAlert) -> None:
        """Fire a latency alert."""
        self._alerts.append(alert)
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                print(f"[LatencyMonitor] Alert callback error: {e}")

    def get_statistics(self, window_sec: float = 5.0) -> Dict[str, Any]:
        """
        Get comprehensive latency statistics.

        Args:
            window_sec: Time window for statistics computation.

        Returns:
            Dictionary containing all monitoring statistics.
        """
        stats = self._rolling_stats.get_stats(window_sec)
        all_windows = self._rolling_stats.get_all_windows()

        return {
            "current_window": stats,
            "all_windows": all_windows,
            "jitter": {
                "jitter_rate": self._jitter_detector.jitter_rate,
                "avg_jitter_ms": self._jitter_detector.avg_jitter_ms,
            },
            "sla": self._sla_tracker.get_report(),
            "stage_latencies": dict(self._stage_latencies),
            "total_inferences": self._total_inferences,
            "recent_alerts_count": len(self._alerts),
        }

    def get_alerts(self, severity: Optional[AlertSeverity] = None, limit: int = 50) -> List[LatencyAlert]:
        """
        Get recent alerts.

        Args:
            severity: Filter by severity level.
            limit: Maximum number of alerts to return.

        Returns:
            List of recent alerts.
        """
        alerts = list(self._alerts)
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def get_latency_budget(self, total_budget_ms: float) -> Dict[str, float]:
        """
        Calculate latency budget allocation across pipeline stages.

        Args:
            total_budget_ms: Total available latency budget in ms.

        Returns:
            Dictionary mapping stage names to allocated budget.
        """
        if not self._stage_latencies:
            return {"total": total_budget_ms}

        total_measured = sum(self._stage_latencies.values())
        if total_measured == 0:
            return self._stage_latencies

        budget = {}
        for name, latency in self._stage_latencies.items():
            proportion = latency / total_measured
            budget[name] = proportion * total_budget_ms

        return budget

    def reset(self) -> None:
        """Reset all monitoring state."""
        with self._lock:
            self._rolling_stats = RollingWindowStats()
            self._jitter_detector = JitterDetector(self._sla.max_jitter_ms)
            self._sla_tracker = SLATracker(self._sla)
            self._alerts.clear()
            self._stage_latencies.clear()
            self._stage_timestamps.clear()
            self._total_inferences = 0


class LatencyContext:
    """Context manager for tracking latency of a code block."""

    def __init__(self, monitor: LatencyMonitor, stage_name: str) -> None:
        self._monitor = monitor
        self._stage_name = stage_name
        self._start_time = 0.0
        self._latency_ms = 0.0

    def __enter__(self) -> "LatencyContext":
        self._start_time = time.time()
        return self

    def __exit__(self, *args) -> None:
        self._latency_ms = (time.time() - self._start_time) * 1000
        self._monitor.record(self._latency_ms, stage=self._stage_name)

    @property
    def latency_ms(self) -> float:
        """Get the measured latency (only valid after context exit)."""
        return self._latency_ms
