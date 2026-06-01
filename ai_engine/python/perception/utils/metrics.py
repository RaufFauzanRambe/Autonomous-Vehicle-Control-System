"""Metrics tracking and reporting for Thunders AI.

Provides latency, throughput, and accuracy computation with
export support for JSON and Prometheus formats.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class MetricRecord:
    """A single metric data point.

    Attributes:
        name: Metric name.
        value: Recorded value.
        timestamp: Time of recording.
        tags: Dimensional tags.
    """

    def __init__(
        self,
        name: str,
        value: float,
        timestamp: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        self.value = value
        self.timestamp = timestamp or time.time()
        self.tags = tags or {}

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the record."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


class Metrics:
    """Centralised metrics tracking and reporting.

    Tracks named metrics with optional tags, computes aggregate
    statistics, and exports results in JSON or Prometheus format.

    Attributes:
        prefix: Optional prefix for all metric names.
        records: Stored metric records keyed by name.
    """

    def __init__(
        self,
        prefix: str = "",
        retention: int = 10000,
        default_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.prefix = prefix
        self.retention = retention
        self.default_tags = default_tags or {}
        self.records: Dict[str, List[MetricRecord]] = defaultdict(list)
        self._latency_records: Dict[str, List[float]] = defaultdict(list)
        self._throughput_counters: Dict[str, Dict[str, Any]] = {}
        self._accuracy_records: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        logger.info(
            "Metrics initialised: prefix='%s', retention=%d", prefix, retention
        )

    def track(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricRecord:
        """Record a metric data point.

        Args:
            name: Metric name.
            value: Observed value.
            tags: Optional dimensional tags.

        Returns:
            The created MetricRecord.
        """
        full_name = f"{self.prefix}{name}" if self.prefix else name
        merged_tags = {**self.default_tags, **(tags or {})}

        record = MetricRecord(
            name=full_name,
            value=value,
            tags=merged_tags,
        )
        self.records[full_name].append(record)

        # Enforce retention limit
        if len(self.records[full_name]) > self.retention:
            self.records[full_name] = self.records[full_name][-self.retention:]

        logger.debug("Tracked metric: %s=%.4f", full_name, value)
        return record

    def compute_latency(
        self,
        operation: str,
        window: Optional[int] = None,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Compute latency statistics for a tracked operation.

        Args:
            operation: Operation name.
            window: Number of recent samples to consider (None = all).
            percentiles: Percentile values to compute (default [50, 90, 95, 99]).

        Returns:
            Dictionary with min, max, avg, p50, p90, p95, p99 in seconds.
        """
        percentiles = percentiles or [50, 90, 95, 99]
        latencies = self._latency_records.get(operation, [])

        if not latencies:
            return {"operation": operation, "error": "no data"}

        samples = latencies[-window:] if window else latencies
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        result: Dict[str, float] = {
            "operation": operation,
            "count": float(n),
            "min": sorted_samples[0],
            "max": sorted_samples[-1],
            "avg": sum(sorted_samples) / n,
        }

        for p in percentiles:
            index = min(int(n * p / 100), n - 1)
            result[f"p{p}"] = sorted_samples[index]

        return result

    def compute_throughput(
        self,
        operation: str,
        window_seconds: float = 60.0,
    ) -> Dict[str, float]:
        """Compute throughput for a tracked operation.

        Args:
            operation: Operation name.
            window_seconds: Time window for rate calculation.

        Returns:
            Dictionary with operations_per_second and total_operations.
        """
        counter = self._throughput_counters.get(operation)
        if not counter:
            return {"operation": operation, "error": "no data"}

        elapsed = time.time() - counter["start_time"]
        total = counter["count"]
        elapsed = max(elapsed, 0.001)  # avoid division by zero

        return {
            "operation": operation,
            "total_operations": float(total),
            "elapsed_seconds": round(elapsed, 4),
            "operations_per_second": round(total / elapsed, 4),
        }

    def compute_accuracy(
        self,
        model_name: str,
        window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute accuracy statistics for a model.

        Args:
            model_name: Model identifier.
            window: Number of recent predictions to consider.

        Returns:
            Dictionary with accuracy, total, and correct counts.
        """
        records = self._accuracy_records.get(model_name, [])
        if not records:
            return {"model": model_name, "error": "no data"}

        samples = records[-window:] if window else records
        total = len(samples)
        correct = sum(1 for _, is_correct in samples if is_correct)

        return {
            "model": model_name,
            "accuracy": round(correct / total, 6) if total > 0 else 0.0,
            "total_predictions": float(total),
            "correct_predictions": float(correct),
        }

    def record_latency(self, operation: str, duration_seconds: float) -> None:
        """Record a latency measurement.

        Args:
            operation: Operation name.
            duration_seconds: Measured duration.
        """
        self._latency_records[operation].append(duration_seconds)
        self.track(f"latency.{operation}", duration_seconds)

    def record_throughput(self, operation: str, count: int = 1) -> None:
        """Record a throughput event.

        Args:
            operation: Operation name.
            count: Number of operations completed.
        """
        if operation not in self._throughput_counters:
            self._throughput_counters[operation] = {
                "count": 0,
                "start_time": time.time(),
            }
        self._throughput_counters[operation]["count"] += count

    def record_accuracy(
        self, model_name: str, prediction_id: int, is_correct: bool
    ) -> None:
        """Record an accuracy measurement.

        Args:
            model_name: Model identifier.
            prediction_id: Prediction identifier.
            is_correct: Whether the prediction was correct.
        """
        self._accuracy_records[model_name].append(
            (prediction_id, int(is_correct))
        )
        self.track(
            f"accuracy.{model_name}",
            float(is_correct),
            tags={"model": model_name},
        )

    def export(
        self,
        format: str = "json",
        metric_names: Optional[List[str]] = None,
    ) -> Any:
        """Export tracked metrics in the specified format.

        Args:
            format: 'json' or 'prometheus'.
            metric_names: Optional filter for specific metrics.

        Returns:
            Formatted metrics string.

        Raises:
            ValueError: If format is unsupported.
        """
        names = metric_names or list(self.records.keys())

        if format == "json":
            return self._export_json(names)
        elif format == "prometheus":
            return self._export_prometheus(names)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # -- Internal helpers ---------------------------------------------------

    def _export_json(self, names: List[str]) -> str:
        """Export metrics as JSON."""
        data: Dict[str, Any] = {}
        for name in names:
            records = self.records.get(name, [])
            if not records:
                continue
            values = [r.value for r in records]
            data[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": round(sum(values) / len(values), 6),
                "latest": values[-1],
                "tags": records[-1].tags,
            }
        return json.dumps(data, indent=2)

    def _export_prometheus(self, names: List[str]) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines: List[str] = []
        for name in names:
            records = self.records.get(name, [])
            if not records:
                continue
            safe_name = name.replace(".", "_")
            lines.append(f"# TYPE {safe_name} gauge")
            latest = records[-1]
            labels = ""
            if latest.tags:
                parts = [f'{k}="{v}"' for k, v in latest.tags.items()]
                labels = "{" + ",".join(parts) + "}"
            lines.append(f"{safe_name}{labels} {latest.value}")
        return "\n".join(lines) + "\n"
