"""
Segmentation Metrics Module for Autonomous Vehicle Perception.

Comprehensive evaluation metrics for semantic segmentation including
mean IoU, pixel accuracy, dice coefficient, boundary F1, and
per-class metrics with statistical analysis.

Metrics Overview:
    ┌────────────────────────────────────────────┐
    │           Segmentation Metrics             │
    ├────────────────┬───────────────────────────┤
    │  Global        │  Per-Class                │
    │  - mIoU        │  - Class IoU              │
    │  - Pixel Acc   │  - Class Dice             │
    │  - Mean Dice   │  - Class Precision        │
    │  - FWIoU       │  - Class Recall           │
    │  - Boundary F1 │  - Class F1               │
    │  - Kappa       │  - Class Accuracy          │
    └────────────────┴───────────────────────────┘

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Data Classes
# ---------------------------------------------------------------------------

@dataclass
class PerClassMetrics:
    """Metrics for a single class.

    Attributes:
        class_id: Class index.
        class_name: Human-readable class name.
        iou: Intersection over Union.
        dice: Dice coefficient (F1 score).
        precision: Positive predictive value.
        recall: True positive rate (sensitivity).
        specificity: True negative rate.
        accuracy: Pixel accuracy for this class.
        f1: F1 score (same as dice for binary).
        pixel_count: Number of ground truth pixels.
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.
        tn: True negative count.
    """

    class_id: int = 0
    class_name: str = ""
    iou: float = 0.0
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    pixel_count: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "iou": round(self.iou, 6),
            "dice": round(self.dice, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "specificity": round(self.specificity, 6),
            "accuracy": round(self.accuracy, 6),
            "f1": round(self.f1, 6),
            "pixel_count": self.pixel_count,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


@dataclass
class SegmentationMetrics:
    """Complete segmentation evaluation metrics.

    Attributes:
        miou: Mean Intersection over Union (averaged over valid classes).
        pixel_accuracy: Global pixel accuracy.
        mean_pixel_accuracy: Mean per-class pixel accuracy.
        mean_dice: Mean Dice coefficient.
        frequency_weighted_iou: Frequency-weighted IoU.
        boundary_f1: Boundary F1 score.
        kappa: Cohen's Kappa coefficient.
        per_class: Per-class metrics.
        num_classes: Number of classes evaluated.
        ignore_label: Label value ignored in evaluation.
        total_pixels: Total number of evaluated pixels.
    """

    miou: float = 0.0
    pixel_accuracy: float = 0.0
    mean_pixel_accuracy: float = 0.0
    mean_dice: float = 0.0
    frequency_weighted_iou: float = 0.0
    boundary_f1: float = 0.0
    kappa: float = 0.0
    per_class: List[PerClassMetrics] = field(default_factory=list)
    num_classes: int = 0
    ignore_label: int = 255
    total_pixels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "miou": round(self.miou, 6),
            "pixel_accuracy": round(self.pixel_accuracy, 6),
            "mean_pixel_accuracy": round(self.mean_pixel_accuracy, 6),
            "mean_dice": round(self.mean_dice, 6),
            "frequency_weighted_iou": round(self.frequency_weighted_iou, 6),
            "boundary_f1": round(self.boundary_f1, 6),
            "kappa": round(self.kappa, 6),
            "num_classes": self.num_classes,
            "total_pixels": self.total_pixels,
            "per_class": [m.to_dict() for m in self.per_class],
        }

    def summary(self) -> str:
        """Generate human-readable summary string."""
        lines = [
            "=" * 60,
            "  Segmentation Evaluation Results",
            "=" * 60,
            f"  mIoU:                    {self.miou:.4f}",
            f"  Pixel Accuracy:          {self.pixel_accuracy:.4f}",
            f"  Mean Pixel Accuracy:     {self.mean_pixel_accuracy:.4f}",
            f"  Mean Dice:               {self.mean_dice:.4f}",
            f"  Frequency-Weighted IoU:  {self.frequency_weighted_iou:.4f}",
            f"  Boundary F1:             {self.boundary_f1:.4f}",
            f"  Cohen's Kappa:           {self.kappa:.4f}",
            "-" * 60,
            f"  {'Class':<20} {'IoU':>8} {'Dice':>8} {'Prec':>8} {'Recall':>8}",
            "-" * 60,
        ]

        for cls_metric in self.per_class:
            name = cls_metric.class_name or f"class_{cls_metric.class_id}"
            lines.append(
                f"  {name:<20} {cls_metric.iou:>8.4f} {cls_metric.dice:>8.4f} "
                f"{cls_metric.precision:>8.4f} {cls_metric.recall:>8.4f}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Confusion Matrix Accumulator
# ---------------------------------------------------------------------------

class ConfusionMatrixAccumulator:
    """Accumulates confusion matrix statistics across batches.

    Memory-efficient accumulation for large datasets where the
    full confusion matrix might not fit in memory.

    Attributes:
        num_classes: Number of classes.
        ignore_label: Label value to ignore.
        matrix: Accumulated confusion matrix.
    """

    def __init__(
        self,
        num_classes: int = 19,
        ignore_label: int = 255,
    ) -> None:
        """Initialize accumulator.

        Args:
            num_classes: Number of segmentation classes.
            ignore_label: Label value to ignore in computation.
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Update confusion matrix with new predictions.

        Args:
            prediction: Predicted segmentation mask (H, W) or (N, H, W).
            target: Ground truth mask (H, W) or (N, H, W).
        """
        pred_flat = prediction.flatten()
        target_flat = target.flatten()

        # Filter ignored labels
        valid = target_flat != self.ignore_label
        pred_valid = pred_flat[valid]
        target_valid = target_flat[valid]

        # Clip to valid class range
        pred_valid = np.clip(pred_valid, 0, self.num_classes - 1)
        target_valid = np.clip(target_valid, 0, self.num_classes - 1)

        # Accumulate
        for t, p in zip(target_valid, pred_valid):
            self.matrix[int(t), int(p)] += 1

    def update_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Update with a batch of predictions.

        Args:
            predictions: Batch of predictions (N, H, W).
            targets: Batch of targets (N, H, W).
        """
        for i in range(predictions.shape[0]):
            self.update(predictions[i], targets[i])

    def get_matrix(self) -> np.ndarray:
        """Get the accumulated confusion matrix.

        Returns:
            Confusion matrix of shape (num_classes, num_classes).
        """
        return self.matrix.copy()

    def reset(self) -> None:
        """Reset the accumulator."""
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

class MetricsComputer:
    """Computes comprehensive segmentation metrics from confusion matrix.

    Supports:
        - Mean IoU (mIoU)
        - Pixel Accuracy (PA)
        - Mean Pixel Accuracy (mPA)
        - Dice Coefficient
        - Frequency-Weighted IoU (FWIoU)
        - Boundary F1 Score
        - Cohen's Kappa
        - Per-class metrics

    Example:
        >>> computer = MetricsComputer(num_classes=19)
        >>> computer.update(pred_mask, gt_mask)
        >>> metrics = computer.compute()
        >>> print(metrics.summary())
    """

    # Cityscapes class names
    CITYSCAPES_NAMES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle",
    ]

    def __init__(
        self,
        num_classes: int = 19,
        ignore_label: int = 255,
        class_names: Optional[List[str]] = None,
        boundary_tolerance: int = 2,
    ) -> None:
        """Initialize metrics computer.

        Args:
            num_classes: Number of segmentation classes.
            ignore_label: Label to ignore in evaluation.
            class_names: Human-readable class names.
            boundary_tolerance: Tolerance for boundary F1 computation.
        """
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names or self.CITYSCAPES_NAMES[:num_classes]
        self.boundary_tolerance = boundary_tolerance

        self._accumulator = ConfusionMatrixAccumulator(num_classes, ignore_label)
        self._boundary_f1_scores: List[float] = []
        self._total_samples = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Update metrics with a new prediction-target pair.

        Args:
            prediction: Predicted mask (H, W).
            target: Ground truth mask (H, W).
        """
        self._accumulator.update(prediction, target)
        self._total_samples += 1

        # Compute boundary F1 for this sample
        bf1 = self._compute_boundary_f1_single(prediction, target)
        self._boundary_f1_scores.append(bf1)

    def update_batch(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """Update with a batch of predictions.

        Args:
            predictions: Batch predictions (N, H, W).
            targets: Batch targets (N, H, W).
        """
        self._accumulator.update_batch(predictions, targets)
        for i in range(predictions.shape[0]):
            bf1 = self._compute_boundary_f1_single(predictions[i], targets[i])
            self._boundary_f1_scores.append(bf1)
        self._total_samples += predictions.shape[0]

    def compute(self) -> SegmentationMetrics:
        """Compute all segmentation metrics.

        Returns:
            Complete metrics object.
        """
        matrix = self._accumulator.get_matrix()

        # Per-class metrics
        per_class_metrics = []
        valid_classes = []

        for c in range(self.num_classes):
            tp = matrix[c, c]
            fp = np.sum(matrix[:, c]) - tp  # Column sum - TP
            fn = np.sum(matrix[c, :]) - tp  # Row sum - TP
            tn = np.sum(matrix) - tp - fp - fn

            # IoU
            union = tp + fp + fn
            iou = tp / union if union > 0 else float("nan")

            # Dice
            dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")

            # Precision, Recall, Specificity
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

            # Accuracy
            class_total = tp + fn  # Total ground truth pixels for this class
            accuracy = tp / class_total if class_total > 0 else float("nan")

            # F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")

            class_name = self.class_names[c] if c < len(self.class_names) else f"class_{c}"
            pixel_count = int(np.sum(matrix[c, :]))

            metric = PerClassMetrics(
                class_id=c,
                class_name=class_name,
                iou=float(iou) if not math.isnan(iou) else 0.0,
                dice=float(dice) if not math.isnan(dice) else 0.0,
                precision=float(precision) if not math.isnan(precision) else 0.0,
                recall=float(recall) if not math.isnan(recall) else 0.0,
                specificity=float(specificity) if not math.isnan(specificity) else 0.0,
                accuracy=float(accuracy) if not math.isnan(accuracy) else 0.0,
                f1=float(f1) if not math.isnan(f1) else 0.0,
                pixel_count=pixel_count,
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
            )

            per_class_metrics.append(metric)

            # Track valid classes (those with ground truth pixels)
            if pixel_count > 0:
                valid_classes.append(c)

        # mIoU: mean over valid classes
        ious = [per_class_metrics[c].iou for c in valid_classes]
        miou = float(np.mean(ious)) if ious else 0.0

        # Pixel accuracy
        total_correct = np.sum(np.diag(matrix))
        total_pixels = np.sum(matrix)
        pixel_accuracy = float(total_correct / total_pixels) if total_pixels > 0 else 0.0

        # Mean pixel accuracy
        class_accuracies = [per_class_metrics[c].accuracy for c in valid_classes]
        mean_pixel_accuracy = float(np.mean(class_accuracies)) if class_accuracies else 0.0

        # Mean Dice
        dices = [per_class_metrics[c].dice for c in valid_classes]
        mean_dice = float(np.mean(dices)) if dices else 0.0

        # Frequency-weighted IoU
        freq_iou_sum = 0.0
        for c in valid_classes:
            freq = per_class_metrics[c].pixel_count / total_pixels if total_pixels > 0 else 0
            freq_iou_sum += freq * per_class_metrics[c].iou
        frequency_weighted_iou = freq_iou_sum

        # Boundary F1
        boundary_f1 = float(np.mean(self._boundary_f1_scores)) if self._boundary_f1_scores else 0.0

        # Cohen's Kappa
        kappa = self._compute_kappa(matrix)

        return SegmentationMetrics(
            miou=miou,
            pixel_accuracy=pixel_accuracy,
            mean_pixel_accuracy=mean_pixel_accuracy,
            mean_dice=mean_dice,
            frequency_weighted_iou=frequency_weighted_iou,
            boundary_f1=boundary_f1,
            kappa=kappa,
            per_class=per_class_metrics,
            num_classes=self.num_classes,
            ignore_label=self.ignore_label,
            total_pixels=int(total_pixels),
        )

    @staticmethod
    def _compute_kappa(matrix: np.ndarray) -> float:
        """Compute Cohen's Kappa coefficient.

        Args:
            matrix: Confusion matrix.

        Returns:
            Kappa value.
        """
        total = np.sum(matrix)
        if total == 0:
            return 0.0

        # Observed agreement
        po = np.sum(np.diag(matrix)) / total

        # Expected agreement
        row_sums = np.sum(matrix, axis=1) / total
        col_sums = np.sum(matrix, axis=0) / total
        pe = np.sum(row_sums * col_sums)

        if pe == 1.0:
            return 0.0

        return float((po - pe) / (1 - pe))

    def _compute_boundary_f1_single(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute Boundary F1 for a single prediction-target pair.

        Args:
            pred: Predicted mask (H, W).
            target: Ground truth mask (H, W).

        Returns:
            Boundary F1 score.
        """
        from .segmentation_utils import compute_boundary_f1

        # Filter ignored pixels
        valid = target != self.ignore_label
        pred_filtered = pred.copy()
        target_filtered = target.copy()
        pred_filtered[~valid] = 0
        target_filtered[~valid] = 0

        return compute_boundary_f1(pred_filtered, target_filtered, self.boundary_tolerance)

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._accumulator.reset()
        self._boundary_f1_scores = []
        self._total_samples = 0


# ---------------------------------------------------------------------------
# Statistical Analysis
# ---------------------------------------------------------------------------

class MetricsStatistics:
    """Statistical analysis of segmentation metrics across runs.

    Computes mean, standard deviation, and confidence intervals
    for metrics across multiple evaluation runs or cross-validation folds.

    Example:
        >>> stats = MetricsStatistics()
        >>> stats.add_run(metrics_run_1)
        >>> stats.add_run(metrics_run_2)
        >>> stats.add_run(metrics_run_3)
        >>> report = stats.compute_statistics()
    """

    def __init__(self, confidence_level: float = 0.95) -> None:
        """Initialize metrics statistics.

        Args:
            confidence_level: Confidence level for intervals (default: 95%).
        """
        self.confidence_level = confidence_level
        self._runs: List[SegmentationMetrics] = []

    def add_run(self, metrics: SegmentationMetrics) -> None:
        """Add a metrics run.

        Args:
            metrics: Segmentation metrics from one evaluation run.
        """
        self._runs.append(metrics)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics across all runs.

        Returns:
            Dictionary with mean, std, and confidence intervals.
        """
        if not self._runs:
            return {}

        # Extract scalar metrics from all runs
        scalar_metrics = ["miou", "pixel_accuracy", "mean_pixel_accuracy",
                         "mean_dice", "frequency_weighted_iou", "boundary_f1", "kappa"]

        results: Dict[str, Any] = {}

        for metric_name in scalar_metrics:
            values = [getattr(run, metric_name) for run in self._runs]
            mean = float(np.mean(values))
            std = float(np.std(values))

            # Confidence interval (t-distribution for small samples)
            n = len(values)
            if n > 1:
                # t-value approximation for 95% CI
                t_value = 1.96  # Approximate for large n
                if n < 30:
                    # Use a simple approximation of t-value
                    t_value = 2.0 + 2.0 / max(n - 1, 1)
                ci_half = t_value * std / math.sqrt(n)
            else:
                ci_half = 0.0

            results[metric_name] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(min(values), 6),
                "max": round(max(values), 6),
                "ci_lower": round(mean - ci_half, 6),
                "ci_upper": round(mean + ci_half, 6),
            }

        # Per-class IoU statistics
        class_ious: Dict[str, List[float]] = {}
        for run in self._runs:
            for cls_metric in run.per_class:
                name = cls_metric.class_name
                if name not in class_ious:
                    class_ious[name] = []
                class_ious[name].append(cls_metric.iou)

        results["per_class_iou"] = {}
        for name, ious in class_ious.items():
            results["per_class_iou"][name] = {
                "mean": round(float(np.mean(ious)), 6),
                "std": round(float(np.std(ious)), 6),
                "min": round(min(ious), 6),
                "max": round(max(ious), 6),
            }

        results["num_runs"] = len(self._runs)
        results["confidence_level"] = self.confidence_level

        return results

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a formatted statistical report.

        Args:
            output_path: Optional file path to save the report.

        Returns:
            Formatted report string.
        """
        stats = self.compute_statistics()

        lines = [
            "=" * 70,
            f"  Segmentation Metrics Statistical Report",
            f"  Runs: {stats.get('num_runs', 0)}, Confidence: {stats.get('confidence_level', 0.95):.0%}",
            "=" * 70,
            "",
            f"  {'Metric':<25} {'Mean':>8} {'Std':>8} {'CI Lower':>10} {'CI Upper':>10}",
            "-" * 70,
        ]

        for metric_name in ["miou", "pixel_accuracy", "mean_pixel_accuracy",
                           "mean_dice", "frequency_weighted_iou", "boundary_f1", "kappa"]:
            if metric_name in stats:
                s = stats[metric_name]
                lines.append(
                    f"  {metric_name:<25} {s['mean']:>8.4f} {s['std']:>8.4f} "
                    f"{s['ci_lower']:>10.4f} {s['ci_upper']:>10.4f}"
                )

        lines.extend(["", "-" * 70, "  Per-Class IoU Statistics:", "-" * 70])

        per_class = stats.get("per_class_iou", {})
        for name, s in sorted(per_class.items()):
            lines.append(
                f"  {name:<25} {s['mean']:>8.4f} {s['std']:>8.4f} "
                f"{s['min']:>8.4f} {s['max']:>8.4f}"
            )

        lines.append("=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report
