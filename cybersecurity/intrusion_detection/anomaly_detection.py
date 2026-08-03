"""Anomaly detection for time-series vehicle / network / sensor metrics.

Combines lightweight statistical detectors (z-score, IQR) with optional
scikit-learn based models (Isolation Forest, one-class SVM, autoencoder stub)
to flag anomalous metric values and streams.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scikit-learn is the recommended backend
    from sklearn.ensemble import IsolationForest  # type: ignore
    from sklearn.neighbors import LocalOutlierFactor  # type: ignore
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn optional
    _HAS_SKLEARN = False

from .constants import (
    DEFAULT_ANOMALY_IQR_MULTIPLIER,
    DEFAULT_ANOMALY_ZSCORE_THRESHOLD,
    DEFAULT_BASELINE_TRAINING_SAMPLES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    """Outcome of evaluating a single observation against the baseline."""

    metric: str
    value: float
    score: float  # higher = more anomalous
    is_anomalous: bool
    method: str
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricBaseline:
    """Statistical baseline for a single metric."""

    name: str
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0
    _samples: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))

    def add_sample(self, value: float) -> None:
        self._samples.append(float(value))
        self.sample_count = len(self._samples)
        self._recompute()

    def _recompute(self) -> None:
        if not self._samples:
            return
        arr = np.asarray(self._samples, dtype=float)
        self.mean = float(arr.mean())
        self.std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        self.min = float(arr.min())
        self.max = float(arr.max())
        self.q1 = float(np.percentile(arr, 25))
        self.q3 = float(np.percentile(arr, 75))
        self.iqr = self.q3 - self.q1
        self.last_updated = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Statistical + ML anomaly detector for streaming metrics.

    Two layers operate concurrently:

    * **Statistical layer**: maintains per-metric rolling baselines and flags
      outliers via z-score and Tukey's IQR fence. Low latency, no training.
    * **ML layer**: an Isolation Forest (when scikit-learn is available)
      trained on multi-metric feature vectors, allowing detection of
      multivariate anomalies that single-metric statistics would miss.
    """

    def __init__(
        self,
        zscore_threshold: float = DEFAULT_ANOMALY_ZSCORE_THRESHOLD,
        iqr_multiplier: float = DEFAULT_ANOMALY_IQR_MULTIPLIER,
        min_samples: int = DEFAULT_BASELINE_TRAINING_SAMPLES,
        isolation_forest_contamination: float = 0.05,
        enabled_methods: Optional[Sequence[str]] = None,
    ) -> None:
        self.zscore_threshold = float(zscore_threshold)
        self.iqr_multiplier = float(iqr_multiplier)
        self.min_samples = int(min_samples)
        self.if_contamination = float(isolation_forest_contamination)
        self.enabled_methods = list(enabled_methods) if enabled_methods else ["zscore", "iqr", "isoforest"]
        self._baselines: Dict[str, MetricBaseline] = {}
        self._ml_model: Optional[Any] = None  # IsolationForest
        self._feature_window: Deque[List[float]] = deque(maxlen=50000)
        self._feature_names: List[str] = []
        self._lock = threading.RLock()
        self._model_dirty = False

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def train_baseline(self, metric: str, samples: Sequence[float]) -> MetricBaseline:
        """Train (or refresh) the baseline for a single metric."""
        with self._lock:
            bl = self._baselines.setdefault(metric, MetricBaseline(name=metric))
            for s in samples:
                bl.add_sample(s)
            logger.info(
                "Trained baseline for '%s' on %d samples (mean=%.3f, std=%.3f)",
                metric, bl.sample_count, bl.mean, bl.std,
            )
            return bl

    def add_observation(self, metric: str, value: float) -> None:
        """Feed a single observation into the rolling baseline."""
        with self._lock:
            bl = self._baselines.setdefault(metric, MetricBaseline(name=metric))
            bl.add_sample(value)

    def feed_multivariate(self, feature_vector: Dict[str, float]) -> None:
        """Feed a multi-metric observation for ML model training."""
        with self._lock:
            if not self._feature_names:
                self._feature_names = sorted(feature_vector.keys())
            row = [feature_vector.get(n, 0.0) for n in self._feature_names]
            self._feature_window.append(row)
            if len(self._feature_window) >= self.min_samples and (
                self._ml_model is None or self._model_dirty
            ):
                self._fit_isolation_forest()
            # Also feed each metric into its statistical baseline
            for k, v in feature_vector.items():
                self.add_observation(k, v)

    def get_baseline(self, metric: str) -> Optional[MetricBaseline]:
        """Return the baseline for a metric, if any."""
        with self._lock:
            return self._baselines.get(metric)

    def list_baselines(self) -> List[str]:
        with self._lock:
            return list(self._baselines.keys())

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_anomaly(self, metric: str, value: float) -> AnomalyResult:
        """Evaluate a single observation against its baseline."""
        with self._lock:
            bl = self._baselines.get(metric)
        if bl is None or bl.sample_count < max(2, self.min_samples // 10):
            return AnomalyResult(
                metric=metric,
                value=value,
                score=0.0,
                is_anomalous=False,
                method="none",
                threshold=0.0,
                details={"reason": "insufficient_baseline"},
            )

        results: List[AnomalyResult] = []
        if "zscore" in self.enabled_methods:
            results.append(self._zscore_check(bl, value))
        if "iqr" in self.enabled_methods:
            results.append(self._iqr_check(bl, value))
        if not results:
            results.append(AnomalyResult(
                metric=metric, value=value, score=0.0,
                is_anomalous=False, method="none", threshold=0.0,
            ))
        # Return the most anomalous result
        return max(results, key=lambda r: r.score)

    def detect_multivariate(self, feature_vector: Dict[str, float]) -> AnomalyResult:
        """Evaluate a multi-metric observation using the ML model."""
        if not _HAS_SKLEARN or self._ml_model is None or not self._feature_names:
            return AnomalyResult(
                metric="<multivariate>",
                value=0.0,
                score=0.0,
                is_anomalous=False,
                method="isoforest",
                threshold=0.0,
                details={"reason": "model_unavailable"},
            )
        row = np.array([[feature_vector.get(n, 0.0) for n in self._feature_names]])
        try:
            score = float(self._ml_model.decision_function(row)[0])
            pred = int(self._ml_model.predict(row)[0])  # -1 = anomaly, 1 = normal
        except Exception as exc:
            logger.error("IsolationForest inference failed: %s", exc)
            return AnomalyResult(
                metric="<multivariate>", value=0.0, score=0.0,
                is_anomalous=False, method="isoforest",
                threshold=0.0, details={"error": str(exc)},
            )
        return AnomalyResult(
            metric="<multivariate>",
            value=0.0,
            score=-score,  # higher = more anomalous
            is_anomalous=(pred == -1),
            method="isoforest",
            threshold=0.0,
            details={"feature_names": list(self._feature_names), "raw_score": score},
        )

    def get_anomaly_score(self, metric: str, value: float) -> float:
        """Return only the anomaly score for an observation."""
        return self.detect_anomaly(metric, value).score

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _zscore_check(self, bl: MetricBaseline, value: float) -> AnomalyResult:
        if bl.std == 0.0:
            # Fall back to absolute deviation
            score = abs(value - bl.mean)
            is_anom = score > 0 and value != bl.mean
            return AnomalyResult(
                metric=bl.name, value=value, score=score,
                is_anomalous=is_anom, method="zscore",
                threshold=self.zscore_threshold,
                details={"mean": bl.mean, "std": bl.std, "zero_variance": True},
            )
        z = abs((value - bl.mean) / bl.std)
        return AnomalyResult(
            metric=bl.name, value=value, score=z,
            is_anomalous=z >= self.zscore_threshold,
            method="zscore",
            threshold=self.zscore_threshold,
            details={"mean": bl.mean, "std": bl.std, "z": z},
        )

    def _iqr_check(self, bl: MetricBaseline, value: float) -> AnomalyResult:
        lower = bl.q1 - self.iqr_multiplier * bl.iqr
        upper = bl.q3 + self.iqr_multiplier * bl.iqr
        if lower <= value <= upper:
            score = 0.0
        else:
            # Distance beyond the fence, normalized by IQR
            if bl.iqr > 0:
                if value < lower:
                    score = (lower - value) / bl.iqr
                else:
                    score = (value - upper) / bl.iqr
            else:
                score = 1.0
        return AnomalyResult(
            metric=bl.name, value=value, score=score,
            is_anomalous=score > 0,
            method="iqr",
            threshold=self.iqr_multiplier,
            details={"lower_fence": lower, "upper_fence": upper, "q1": bl.q1, "q3": bl.q3},
        )

    def _fit_isolation_forest(self) -> None:
        if not _HAS_SKLEARN or not self._feature_window:
            return
        X = np.asarray(self._feature_window, dtype=float)
        try:
            self._ml_model = IsolationForest(
                contamination=self.if_contamination,
                random_state=42,
                n_estimators=100,
            )
            self._ml_model.fit(X)
            self._model_dirty = False
            logger.info("IsolationForest trained on %d samples, %d features", *X.shape)
        except Exception as exc:
            logger.error("Failed to fit IsolationForest: %s", exc)
            self._ml_model = None

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def update_model(self) -> None:
        """Force a refit of the ML model from the current feature window."""
        with self._lock:
            self._model_dirty = True
            self._fit_isolation_forest()

    def save(self, path: str) -> None:
        """Persist baselines + ML model to ``path`` via pickle."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
            payload = {
                "baselines": {k: v.to_dict() for k, v in self._baselines.items()},
                "samples": {k: list(v._samples) for k, v in self._baselines.items()},
                "feature_window": list(self._feature_window),
                "feature_names": self._feature_names,
                "zscore_threshold": self.zscore_threshold,
                "iqr_multiplier": self.iqr_multiplier,
                "min_samples": self.min_samples,
            }
            try:
                payload["ml_model"] = pickle.dumps(self._ml_model)
            except Exception:
                payload["ml_model"] = None
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        logger.info("Saved anomaly detector state to %s", path)

    def load(self, path: str) -> None:
        """Load a previously saved state from ``path``."""
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        with self._lock:
            self.zscore_threshold = payload.get("zscore_threshold", self.zscore_threshold)
            self.iqr_multiplier = payload.get("iqr_multiplier", self.iqr_multiplier)
            self.min_samples = payload.get("min_samples", self.min_samples)
            self._baselines.clear()
            for name, samples in payload.get("samples", {}).items():
                bl = MetricBaseline(name=name)
                for s in samples:
                    bl.add_sample(s)
                self._baselines[name] = bl
            self._feature_names = payload.get("feature_names", [])
            self._feature_window = deque(payload.get("feature_window", []), maxlen=50000)
            ml_blob = payload.get("ml_model")
            self._ml_model = pickle.loads(ml_blob) if ml_blob else None
        logger.info("Loaded anomaly detector state from %s", path)


__all__ = ["AnomalyDetector", "AnomalyResult", "MetricBaseline"]
