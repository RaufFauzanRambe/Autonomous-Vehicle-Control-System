"""Behavioral analysis: baseline normal vehicle / network / process behavior.

The :class:`BehaviorAnalyzer` builds rolling baselines of:

* CAN traffic per arbitration ID (rate, payload byte entropy).
* Network connection 5-tuple frequencies.
* Process-tree shape (parent -> child pairs and command lines).

It then computes an aggregate *behavior score* that quantifies how far the
current observation window deviates from the baseline. Scores above a
configurable threshold raise a behavior-deviation alert.
"""

from __future__ import annotations

import collections
import logging
import math
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from .constants import (
    AlertSeverity,
    DEFAULT_BASELINE_TRAINING_SAMPLES,
    ThreatType,
)
from .utils import timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BehaviorProfile:
    """A baseline profile for a category of behavior."""

    name: str
    sample_count: int = 0
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    histogram: Dict[str, int] = field(default_factory=dict)
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "means": dict(self.means),
            "stds": dict(self.stds),
            "histogram": dict(self.histogram),
            "last_updated": self.last_updated,
        }


@dataclass
class BehaviorScore:
    """Result of scoring a behavior sample against the baseline."""

    profile: str
    score: float  # 0.0 (normal) - 1.0 (very anomalous)
    is_anomalous: bool
    contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=timestamp_now)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class BehaviorAnalyzer:
    """Builds and evaluates behavioral baselines for vehicle subsystems."""

    def __init__(
        self,
        min_samples: int = DEFAULT_BASELINE_TRAINING_SAMPLES,
        deviation_threshold: float = 0.6,
    ) -> None:
        self.min_samples = int(min_samples)
        self.deviation_threshold = float(deviation_threshold)
        self._profiles: Dict[str, BehaviorProfile] = {}
        self._windows: Dict[str, Deque[Dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._lock = threading.RLock()
        self._scores: Deque[BehaviorScore] = deque(maxlen=20_000)
        self._stats = {"samples_observed": 0, "anomalies_detected": 0}

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    def build_baseline(
        self,
        profile: str,
        samples: List[Dict[str, float]],
    ) -> BehaviorProfile:
        """Train a baseline for ``profile`` from a list of feature dicts."""
        with self._lock:
            p = self._profiles.setdefault(profile, BehaviorProfile(name=profile))
            p.histogram.clear()
            per_feature: Dict[str, List[float]] = defaultdict(list)
            for s in samples:
                for k, v in s.items():
                    per_feature[k].append(float(v))
            p.means = {k: statistics.fmean(v) for k, v in per_feature.items()}
            p.stds = {
                k: (statistics.pstdev(v) if len(v) > 1 else 0.0)
                for k, v in per_feature.items()
            }
            p.sample_count = len(samples)
            p.last_updated = timestamp_now()
        logger.info("Built baseline '%s' on %d samples, %d features",
                    profile, p.sample_count, len(p.means))
        return p

    def update_baseline(self, profile: str, sample: Dict[str, float]) -> None:
        """Online update of a baseline with a single observation."""
        with self._lock:
            window = self._windows[profile]
            window.append(sample)
            self._stats["samples_observed"] += 1
            if len(window) >= self.min_samples and (
                profile not in self._profiles
                or timestamp_now() - self._profiles[profile].last_updated > 60.0
            ):
                self.build_baseline(profile, list(window))

    def get_profile(self, profile: str) -> Optional[BehaviorProfile]:
        with self._lock:
            return self._profiles.get(profile)

    def list_profiles(self) -> List[str]:
        with self._lock:
            return list(self._profiles.keys())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def analyze_behavior(
        self,
        profile: str,
        sample: Dict[str, float],
    ) -> BehaviorScore:
        """Score a single observation against the baseline for ``profile``."""
        with self._lock:
            p = self._profiles.get(profile)
        if p is None or p.sample_count < self.min_samples // 10:
            return BehaviorScore(
                profile=profile, score=0.0, is_anomalous=False,
                contributions={"reason": "no_baseline"},
            )
        contributions: Dict[str, float] = {}
        for k, value in sample.items():
            mean = p.means.get(k)
            std = p.stds.get(k)
            if mean is None:
                continue
            if std is None or std == 0.0:
                # Boolean / categorical deviation
                dev = 0.0 if value == mean else 1.0
            else:
                z = abs((value - mean) / std)
                # Sigmoid-like mapping: z=0 -> 0, z=3 -> ~0.95
                dev = 1.0 - math.exp(-z / 2.0)
            contributions[k] = dev
        if not contributions:
            return BehaviorScore(profile=profile, score=0.0, is_anomalous=False,
                                 contributions={"reason": "no_features"})
        # Take the maximum contribution but smooth with the mean for stability.
        max_dev = max(contributions.values())
        mean_dev = statistics.fmean(contributions.values())
        score = max(0.0, min(1.0, 0.7 * max_dev + 0.3 * mean_dev))
        is_anomalous = score >= self.deviation_threshold
        result = BehaviorScore(
            profile=profile, score=score, is_anomalous=is_anomalous,
            contributions=contributions,
        )
        with self._lock:
            self._scores.append(result)
            self._stats["samples_observed"] += 1
            if is_anomalous:
                self._stats["anomalies_detected"] += 1
        return result

    def get_behavior_score(self, profile: str) -> Optional[float]:
        """Return the most recent score for a profile, or None."""
        with self._lock:
            for s in reversed(self._scores):
                if s.profile == profile:
                    return s.score
        return None

    # ------------------------------------------------------------------
    # Convenience helpers for vehicle-specific behaviors
    # ------------------------------------------------------------------

    def record_can_traffic(self, can_id: int, rate_per_sec: float, entropy: float) -> BehaviorScore:
        """Score a CAN-traffic sample (per-ID rate + payload entropy)."""
        profile = f"can:0x{can_id:X}"
        return self.analyze_behavior(profile, {"rate": rate_per_sec, "entropy": entropy})

    def record_network_traffic(self, conns_per_min: float, bytes_per_sec: float) -> BehaviorScore:
        """Score aggregate network-traffic behavior."""
        return self.analyze_behavior(
            "network:aggregate",
            {"conns_per_min": conns_per_min, "bytes_per_sec": bytes_per_sec},
        )

    def record_process_tree(self, depth: int, breadth: int, unique_parents: int) -> BehaviorScore:
        """Score the shape of the process tree."""
        return self.analyze_behavior(
            "process:tree",
            {"depth": depth, "breadth": breadth, "unique_parents": unique_parents},
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "profiles": len(self._profiles),
                "scores_buffered": len(self._scores),
            }

    def get_scores(self, profile: Optional[str] = None, limit: int = 100) -> List[BehaviorScore]:
        with self._lock:
            scores = list(self._scores)
        if profile is not None:
            scores = [s for s in scores if s.profile == profile]
        return scores[-limit:]


__all__ = ["BehaviorAnalyzer", "BehaviorProfile", "BehaviorScore"]
