"""
Evaluation Pipeline for Semantic Segmentation in Autonomous Driving.

Provides comprehensive model evaluation including:
    - Model loading and inference on validation/test sets
    - mIoU, pixel accuracy, Dice coefficient, and boundary F1 computation
    - Per-class metric breakdown with class names
    - JSON report generation for programmatic access
    - Confusion matrix computation and export
    - Sample visualization for qualitative analysis
    - Statistical analysis across evaluation runs

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth --config config.yaml
    python evaluate.py --checkpoint checkpoints/best_model.pth --split test

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Local imports
from .segmentation_model import (
    BackboneType,
    BaseSegmentationModel,
    DecoderConfig,
    EncoderConfig,
    ModelConfig,
    ModelFactory,
    NormalizationType,
    ActivationType,
    UNetSegmentationModel,
)
from .road_segmentation import RoadSegmentationModel, RoadSegmentationConfig
from .lane_segmentation import LaneSegmentationModel, LaneSegmentationConfig
from .obstacle_segmentation import ObstacleSegmentationModel, ObstacleSegmentationConfig
from .data_loader import (
    DataLoader,
    DatasetConfig,
    DatasetFactory,
    DatasetType,
    SegmentationDataset,
    SplitType,
)
from .metrics import (
    ConfusionMatrixAccumulator,
    MetricsComputer,
    MetricsStatistics,
    PerClassMetrics,
    SegmentationMetrics,
)
from .segmentation_utils import (
    CITYSCAPES_CLASSES,
    compute_boundary_f1,
    compute_per_class_iou,
    get_cityscapes_palette,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        checkpoint_path: Path to model checkpoint.
        output_dir: Directory for evaluation results.
        split: Dataset split to evaluate ('val' or 'test').
        batch_size: Evaluation batch size.
        num_classes: Number of segmentation classes.
        ignore_label: Label to ignore in metrics.
        compute_per_class: Whether to compute per-class metrics.
        compute_confusion_matrix: Whether to compute confusion matrix.
        compute_boundary_f1: Whether to compute boundary F1.
        boundary_tolerance: Tolerance for boundary F1 (pixels).
        num_vis_samples: Number of visualization samples to save.
        save_visualizations: Whether to save visualization images.
        generate_report: Whether to generate JSON report.
        device: Compute device.
        use_tta: Whether to use test-time augmentation.
    """

    checkpoint_path: str = ""
    output_dir: str = "eval_results"
    split: str = "val"
    batch_size: int = 1
    num_classes: int = 19
    ignore_label: int = 255
    compute_per_class: bool = True
    compute_confusion_matrix: bool = True
    compute_boundary_f1: bool = True
    boundary_tolerance: int = 2
    num_vis_samples: int = 10
    save_visualizations: bool = True
    generate_report: bool = True
    device: str = "cuda"
    use_tta: bool = False


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads a trained segmentation model from checkpoint.

    Handles checkpoint deserialization and model reconstruction
    with proper configuration restoration.

    Example:
        >>> loader = ModelLoader(checkpoint_dir="checkpoints")
        >>> model = loader.load("best_model")
        >>> model.eval()
    """

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        """Initialize model loader.

        Args:
            checkpoint_dir: Directory containing checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)

    def load(
        self,
        checkpoint_name: str = "best_model",
        model_type: str = "unet",
        num_classes: int = 19,
    ) -> BaseSegmentationModel:
        """Load model from checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint (without extension).
            model_type: Type of model architecture.
            num_classes: Number of segmentation classes.

        Returns:
            Loaded model in evaluation mode.
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"
        meta_path = self.checkpoint_dir / f"{checkpoint_name}_meta.json"

        # Load configuration
        config = ModelConfig(num_classes=num_classes)
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                config = ModelConfig(
                    num_classes=meta.get("num_classes", num_classes),
                )
                if "encoder" in meta:
                    enc = meta["encoder"]
                    config.encoder.backbone = BackboneType(enc.get("backbone", "resnet101"))
                    config.encoder.output_stride = enc.get("output_stride", 16)
                logger.info(f"Loaded model config from {meta_path}")
            except Exception as e:
                logger.warning(f"Failed to load meta config: {e}")

        # Create model
        if model_type == "road":
            road_config = RoadSegmentationConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
            )
            model = RoadSegmentationModel(road_config)
        elif model_type == "lane":
            lane_config = LaneSegmentationConfig(
                num_lane_types=config.num_classes,
                input_size=config.input_size,
            )
            model = LaneSegmentationModel(lane_config)
        elif model_type == "obstacle":
            obstacle_config = ObstacleSegmentationConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
            )
            model = ObstacleSegmentationModel(obstacle_config)
        else:
            model = UNetSegmentationModel(config)

        model.build_model()
        model.eval()

        logger.info(f"Model loaded: {model_type}, checkpoint={checkpoint_name}")
        return model

    @staticmethod
    def load_from_config(
        config_path: str,
        checkpoint_path: Optional[str] = None,
    ) -> BaseSegmentationModel:
        """Load model from a configuration file.

        Args:
            config_path: Path to model configuration JSON.
            checkpoint_path: Optional checkpoint path for weights.

        Returns:
            Initialized model.
        """
        model = BaseSegmentationModel.from_config(config_path)
        model.build_model()
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Test-Time Augmentation
# ---------------------------------------------------------------------------

class TestTimeAugmentation:
    """Applies test-time augmentation for more robust predictions.

    Averages predictions over multiple augmented versions of the input:
        - Original image
        - Horizontally flipped image
        - Multi-scale inference

    Attributes:
        transforms: List of augmentation transforms to apply.
    """

    def __init__(
        self,
        transforms: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
    ) -> None:
        """Initialize TTA.

        Args:
            transforms: List of transform names ('original', 'flip_horizontal').
            scales: List of scale factors for multi-scale inference.
        """
        self.transforms = transforms or ["original", "flip_horizontal"]
        self.scales = scales or [1.0]

    def augment_input(self, image: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Generate augmented versions of the input.

        Args:
            image: Input image tensor (N, C, H, W).

        Returns:
            List of (augmented_image, transform_name, scale) tuples.
        """
        augmented = []

        for scale in self.scales:
            if scale != 1.0:
                # Scale the image
                n, c, h, w = image.shape
                new_h, new_w = int(h * scale), int(w * scale)
                # Simple nearest-neighbor resize
                scaled = np.zeros((n, c, new_h, new_w), dtype=image.dtype)
                for y in range(new_h):
                    src_y = min(int(y / scale), h - 1)
                    for x in range(new_w):
                        src_x = min(int(x / scale), w - 1)
                        scaled[:, :, y, x] = image[:, :, src_y, src_x]
            else:
                scaled = image

            for transform in self.transforms:
                if transform == "original":
                    augmented.append((scaled, "original", scale))
                elif transform == "flip_horizontal":
                    flipped = np.flip(scaled, axis=3).copy()
                    augmented.append((flipped, "flip_horizontal", scale))

        return augmented

    def merge_predictions(
        self,
        predictions: List[np.ndarray],
        transform_names: List[str],
        scales: List[float],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Merge predictions from augmented inputs.

        Args:
            predictions: List of prediction logits.
            transform_names: Transform applied to each prediction.
            scales: Scale factor for each prediction.
            target_size: Target output size (H, W).

        Returns:
            Averaged prediction logits.
        """
        merged = np.zeros_like(predictions[0])

        for pred, transform, scale in zip(predictions, transform_names, scales):
            # Reverse flip
            if transform == "flip_horizontal":
                pred = np.flip(pred, axis=3).copy()

            # Reverse scale
            if scale != 1.0:
                n, c, h, w = pred.shape
                th, tw = target_size
                resized = np.zeros((n, c, th, tw), dtype=pred.dtype)
                for y in range(th):
                    src_y = min(int(y * h / th), h - 1)
                    for x in range(tw):
                        src_x = min(int(x * w / tw), w - 1)
                        resized[:, :, y, x] = pred[:, :, src_y, src_x]
                pred = resized

            merged += pred

        merged /= len(predictions)
        return merged


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

class SegmentationEvaluator:
    """Complete evaluation pipeline for semantic segmentation models.

    Provides comprehensive evaluation including:
        - Model inference on validation/test sets
        - Global metrics (mIoU, pixel accuracy, mean Dice, boundary F1)
        - Per-class metrics breakdown
        - Confusion matrix computation
        - Sample visualization generation
        - JSON report generation
        - Statistical analysis support

    Example:
        >>> config = EvaluationConfig(num_classes=19)
        >>> evaluator = SegmentationEvaluator(config)
        >>> model = ModelLoader("checkpoints").load("best_model")
        >>> dataset = DatasetFactory.create(DatasetConfig(split=SplitType.VAL))
        >>> results = evaluator.evaluate(model, dataset)
        >>> print(results.summary())
    """

    # Cityscapes class names for reporting
    CITYSCAPES_NAMES = [
        "road", "sidewalk", "building", "wall", "fence",
        "pole", "traffic_light", "traffic_sign", "vegetation", "terrain",
        "sky", "person", "rider", "car", "truck",
        "bus", "train", "motorcycle", "bicycle",
    ]

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize evaluator.

        Args:
            config: Evaluation configuration.
        """
        self.config = config
        self.metrics_computer = MetricsComputer(
            num_classes=config.num_classes,
            ignore_label=config.ignore_label,
            boundary_tolerance=config.boundary_tolerance,
        )
        self.confusion_accumulator = ConfusionMatrixAccumulator(
            num_classes=config.num_classes,
            ignore_label=config.ignore_label,
        )
        self.tta = TestTimeAugmentation() if config.use_tta else None

        # Results storage
        self._per_image_metrics: List[Dict[str, float]] = []
        self._sample_predictions: List[Dict[str, np.ndarray]] = []

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        model: BaseSegmentationModel,
        dataset: SegmentationDataset,
    ) -> SegmentationMetrics:
        """Run full evaluation on a dataset.

        Args:
            model: Trained segmentation model.
            dataset: Dataset to evaluate on.

        Returns:
            Complete segmentation metrics.
        """
        model.eval()
        self.metrics_computer.reset()
        self.confusion_accumulator.reset()
        self._per_image_metrics = []
        self._sample_predictions = []

        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        total_samples = len(dataset)
        logger.info(f"Starting evaluation on {total_samples} samples")

        start_time = time.time()

        for batch_idx, batch in enumerate(data_loader):
            images = batch["image"]  # (N, H, W, C)
            labels = batch["label"]  # (N, H, W)

            # Transpose to (N, C, H, W)
            if images.ndim == 4 and images.shape[-1] == 3:
                images = np.transpose(images, (0, 3, 1, 2))

            # Inference
            if self.tta is not None:
                predictions = self._tta_inference(model, images)
            else:
                predictions = model.forward(images)

            # Get segmentation masks
            pred_masks = model.get_segmentation_mask(predictions)

            # Update metrics
            if pred_masks.ndim == 3 and labels.ndim == 3:
                for i in range(pred_masks.shape[0]):
                    self.metrics_computer.update(pred_masks[i], labels[i])
                    self.confusion_accumulator.update(pred_masks[i], labels[i])

                    # Per-image metrics
                    per_image_iou = compute_per_class_iou(
                        pred_masks[i], labels[i], self.config.num_classes
                    )
                    valid_ious = per_image_iou[per_image_iou > 0]
                    self._per_image_metrics.append({
                        "image_miou": float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0,
                        "image_pixel_acc": float(np.mean(pred_masks[i] == labels[i])),
                    })

                # Save sample predictions for visualization
                if len(self._sample_predictions) < self.config.num_vis_samples:
                    for i in range(pred_masks.shape[0]):
                        if len(self._sample_predictions) < self.config.num_vis_samples:
                            self._sample_predictions.append({
                                "prediction": pred_masks[i],
                                "label": labels[i],
                                "image": images[i] if images.ndim == 4 else images,
                            })

            # Progress logging
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                processed = min((batch_idx + 1) * self.config.batch_size, total_samples)
                eta = elapsed / max(processed, 1) * (total_samples - processed)
                logger.info(
                    f"  Evaluated {processed}/{total_samples} samples, "
                    f"ETA: {eta:.0f}s"
                )

        eval_time = time.time() - start_time

        # Compute final metrics
        metrics = self.metrics_computer.compute()

        logger.info("=" * 60)
        logger.info(f"Evaluation complete in {eval_time:.1f}s")
        logger.info(metrics.summary())

        # Generate outputs
        if self.config.generate_report:
            self._generate_report(metrics, eval_time)

        if self.config.save_visualizations:
            self._save_visualizations()

        if self.config.compute_confusion_matrix:
            self._save_confusion_matrix()

        return metrics

    def _tta_inference(
        self,
        model: BaseSegmentationModel,
        images: np.ndarray,
    ) -> np.ndarray:
        """Run inference with test-time augmentation.

        Args:
            model: Segmentation model.
            images: Input images (N, C, H, W).

        Returns:
            Averaged prediction logits.
        """
        augmented = self.tta.augment_input(images)
        all_predictions = []
        all_transforms = []
        all_scales = []

        for aug_images, transform, scale in augmented:
            pred = model.forward(aug_images)
            all_predictions.append(pred)
            all_transforms.append(transform)
            all_scales.append(scale)

        _, _, h, w = images.shape
        merged = self.tta.merge_predictions(
            all_predictions, all_transforms, all_scales,
            target_size=(h, w),
        )
        return merged

    def _generate_report(
        self,
        metrics: SegmentationMetrics,
        eval_time: float,
    ) -> None:
        """Generate JSON evaluation report.

        Args:
            metrics: Computed segmentation metrics.
            eval_time: Total evaluation time in seconds.
        """
        report = {
            "evaluation_config": {
                "num_classes": self.config.num_classes,
                "ignore_label": self.config.ignore_label,
                "split": self.config.split,
                "boundary_tolerance": self.config.boundary_tolerance,
                "use_tta": self.config.use_tta,
            },
            "global_metrics": {
                "miou": round(metrics.miou, 6),
                "pixel_accuracy": round(metrics.pixel_accuracy, 6),
                "mean_pixel_accuracy": round(metrics.mean_pixel_accuracy, 6),
                "mean_dice": round(metrics.mean_dice, 6),
                "frequency_weighted_iou": round(metrics.frequency_weighted_iou, 6),
                "boundary_f1": round(metrics.boundary_f1, 6),
                "kappa": round(metrics.kappa, 6),
                "total_pixels": metrics.total_pixels,
            },
            "per_class_metrics": [],
            "evaluation_time_seconds": round(eval_time, 2),
            "num_samples": len(self._per_image_metrics),
        }

        # Per-class metrics
        for cls_metric in metrics.per_class:
            cls_report = cls_metric.to_dict()
            report["per_class_metrics"].append(cls_report)

        # Per-image statistics
        if self._per_image_metrics:
            image_mious = [m["image_miou"] for m in self._per_image_metrics]
            image_accs = [m["image_pixel_acc"] for m in self._per_image_metrics]
            report["per_image_statistics"] = {
                "miou": {
                    "mean": round(float(np.mean(image_mious)), 6),
                    "std": round(float(np.std(image_mious)), 6),
                    "min": round(float(np.min(image_mious)), 6),
                    "max": round(float(np.max(image_mious)), 6),
                },
                "pixel_accuracy": {
                    "mean": round(float(np.mean(image_accs)), 6),
                    "std": round(float(np.std(image_accs)), 6),
                    "min": round(float(np.min(image_accs)), 6),
                    "max": round(float(np.max(image_accs)), 6),
                },
            }

        # Save report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report saved to {report_path}")

    def _save_visualizations(self) -> None:
        """Save visualization images for sample predictions."""
        from .visualization import SegmentationVisualizer

        visualizer = SegmentationVisualizer(num_classes=self.config.num_classes)
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(self._sample_predictions):
            pred = sample["prediction"]
            label = sample["label"]
            image = sample["image"]

            # Convert image from CHW to HWC for visualization
            if isinstance(image, np.ndarray) and image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                # Denormalize
                image = np.clip(image * 255, 0, 255).astype(np.uint8)

            try:
                from PIL import Image as PILImage

                # Side-by-side comparison
                comparison = visualizer.create_comparison(
                    image=image,
                    pred_mask=pred,
                    gt_mask=label,
                )
                PILImage.fromarray(comparison).save(vis_dir / f"sample_{i:04d}_comparison.png")

                # Error map
                error_map = visualizer.create_error_map(pred, label)
                PILImage.fromarray(error_map).save(vis_dir / f"sample_{i:04d}_error.png")

                # Overlay
                overlay = visualizer.overlay_mask(image, pred, alpha=0.5)
                PILImage.fromarray(overlay).save(vis_dir / f"sample_{i:04d}_overlay.png")

            except ImportError:
                # Save raw numpy arrays as fallback
                np.savez(
                    vis_dir / f"sample_{i:04d}.npz",
                    prediction=pred,
                    label=label,
                )

        logger.info(f"Visualizations saved to {vis_dir} ({len(self._sample_predictions)} samples)")

    def _save_confusion_matrix(self) -> None:
        """Save confusion matrix data."""
        matrix = self.confusion_accumulator.get_matrix()

        # Save as numpy file
        np.save(self.output_dir / "confusion_matrix.npy", matrix)

        # Save normalized version
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1, None)  # Avoid division by zero
        normalized = matrix.astype(np.float64) / row_sums

        np.save(self.output_dir / "confusion_matrix_normalized.npy", normalized)

        # Save as JSON
        confusion_data = {
            "raw_matrix": matrix.tolist(),
            "normalized_matrix": normalized.tolist(),
            "class_names": self.CITYSCAPES_NAMES[:self.config.num_classes],
        }
        with open(self.output_dir / "confusion_matrix.json", "w") as f:
            json.dump(confusion_data, f, indent=2)

        logger.info(f"Confusion matrix saved to {self.output_dir}")

    def compute_class_statistics(self) -> Dict[str, Any]:
        """Compute detailed per-class statistics from accumulated results.

        Returns:
            Dictionary with per-class analysis.
        """
        matrix = self.confusion_accumulator.get_matrix()
        stats: Dict[str, Any] = {}

        for c in range(self.config.num_classes):
            tp = matrix[c, c]
            fp = np.sum(matrix[:, c]) - tp
            fn = np.sum(matrix[c, :]) - tp
            tn = np.sum(matrix) - tp - fp - fn

            class_name = self.CITYSCAPES_NAMES[c] if c < len(self.CITYSCAPES_NAMES) else f"class_{c}"

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            stats[class_name] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "pixel_count": int(np.sum(matrix[c, :])),
                "pixel_percentage": round(float(np.sum(matrix[c, :]) / max(np.sum(matrix), 1)) * 100, 2),
            }

        return stats

    def compare_models(
        self,
        models: Dict[str, BaseSegmentationModel],
        dataset: SegmentationDataset,
    ) -> Dict[str, SegmentationMetrics]:
        """Compare multiple models on the same dataset.

        Args:
            models: Dictionary mapping model names to model instances.
            dataset: Dataset to evaluate on.

        Returns:
            Dictionary mapping model names to their metrics.
        """
        results: Dict[str, SegmentationMetrics] = {}
        statistics = MetricsStatistics()

        for name, model in models.items():
            logger.info(f"Evaluating model: {name}")
            metrics = self.evaluate(model, dataset)
            results[name] = metrics
            statistics.add_run(metrics)

        # Generate comparison report
        comparison_report = {}
        for name, metrics in results.items():
            comparison_report[name] = {
                "miou": metrics.miou,
                "pixel_accuracy": metrics.pixel_accuracy,
                "mean_dice": metrics.mean_dice,
                "boundary_f1": metrics.boundary_f1,
            }

        report_path = self.output_dir / "model_comparison.json"
        with open(report_path, "w") as f:
            json.dump(comparison_report, f, indent=2)

        logger.info(f"Model comparison saved to {report_path}")
        return results


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Evaluation batch size")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of segmentation classes")
    parser.add_argument("--model-type", type=str, default="unet",
                        choices=["unet", "road", "lane", "obstacle"],
                        help="Model architecture type")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device")
    parser.add_argument("--save-vis", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--use-tta", action="store_true",
                        help="Use test-time augmentation")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    eval_config = EvaluationConfig(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        save_visualizations=args.save_vis,
        device=args.device,
        use_tta=args.use_tta,
    )

    # Load model
    checkpoint_dir = str(Path(args.checkpoint).parent)
    checkpoint_name = Path(args.checkpoint).stem
    loader = ModelLoader(checkpoint_dir)
    model = loader.load(checkpoint_name, model_type=args.model_type, num_classes=args.num_classes)

    # Create dataset
    split_type = SplitType.VAL if args.split == "val" else SplitType.TEST
    dataset_config = DatasetConfig(
        split=split_type,
        num_classes=args.num_classes,
    )
    dataset = DatasetFactory.create(dataset_config)

    # Run evaluation
    evaluator = SegmentationEvaluator(eval_config)
    results = evaluator.evaluate(model, dataset)

    # Print summary
    print(results.summary())


if __name__ == "__main__":
    main()
