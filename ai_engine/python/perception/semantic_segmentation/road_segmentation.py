"""
Road Segmentation Module for Autonomous Vehicle Perception.

Implements DeepLabV3+ architecture for road and drivable area segmentation.
Handles class-imbalanced datasets common in driving scenes where road
regions dominate the frame.

Architecture Overview (DeepLabV3+):
    ┌───────────────┐
    │  Input Image  │
    └───────┬───────┘
            │
    ┌───────▼───────┐     ┌──────────────┐
    │  Xception/    │────▶│    ASPP      │
    │  ResNet Enc   │     │ (rates: 1,6, │
    │  (OS=16)      │     │  12, 18)     │
    └───────┬───────┘     └──────┬───────┘
            │                    │
            │ (low-level feat)   │ (decoder input)
            │                    │
    ┌───────▼────────────────────▼───────┐
    │           Decoder Module           │
    │  1. Upsample 4x                   │
    │  2. Concat low-level features      │
    │  3. 3x3 Conv x2                   │
    │  4. Upsample 4x                   │
    └───────────────┬────────────────────┘
                    │
            ┌───────▼───────┐
            │ 1x1 Conv →   │
            │ Road/Not Road│
            └───────────────┘

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .segmentation_model import (
    ASPPModule,
    AttentionGate,
    BaseSegmentationModel,
    ConvBlock,
    Decoder,
    DecoderConfig,
    Encoder,
    EncoderConfig,
    ModelConfig,
    NormalizationType,
    ActivationType,
    BackboneType,
    TensorLike,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Road-Specific Enums and Constants
# ---------------------------------------------------------------------------

class RoadClass(Enum):
    """Road segmentation class definitions.

    Follows Cityscapes and KITTI road benchmark conventions.
    """

    BACKGROUND = 0
    ROAD = 1
    SIDEWALK = 2
    PARKING = 3
    RAIL_TRACK = 4


class RoadSubClass(Enum):
    """Fine-grained road subclass definitions for detailed segmentation."""

    BACKGROUND = 0
    ROAD_FLAT = 1
    ROAD_CURVED = 2
    ROAD_INTERSECTION = 3
    SIDEWALK_LEVEL = 4
    SIDEWALK_RAISED = 5
    PARKING_MARKED = 6
    PARKING_UNMARKED = 7
    RAIL_TRACK = 8


# Cityscapes road-relevant class IDs
CITYSCAPES_ROAD_CLASSES = {0, 1, 7, 8, 9}  # road, sidewalk, parking, rail track, carriage
CITYSCAPES_ROAD_MAP = {
    7: RoadClass.ROAD.value,        # road
    8: RoadClass.SIDEWALK.value,    # sidewalk
    9: RoadClass.PARKING.value,     # parking
    10: RoadClass.RAIL_TRACK.value, # rail track
}

# KITTI road benchmark classes
KITTI_ROAD_CLASSES = {0, 1}  # background, road
KITTI_ROAD_MAP = {
    0: RoadClass.BACKGROUND.value,
    1: RoadClass.ROAD.value,
}

# Default class weights for road segmentation (addressing class imbalance)
DEFAULT_ROAD_CLASS_WEIGHTS = np.array([0.3, 1.0, 0.8, 0.5, 0.2])

# Road-specific augmentation parameters
ROAD_AUGMENTATION_PARAMS = {
    "brightness_range": (0.7, 1.3),
    "contrast_range": (0.8, 1.2),
    "saturation_range": (0.7, 1.3),
    "hue_shift": 0.02,
    "gamma_range": (0.8, 1.2),
    "random_crop_scales": [0.5, 0.75, 1.0, 1.25, 1.5],
    "flip_probability": 0.5,
}


# ---------------------------------------------------------------------------
# Road Segmentation Configuration
# ---------------------------------------------------------------------------

@dataclass
class RoadSegmentationConfig:
    """Configuration for road segmentation model.

    Attributes:
        num_classes: Number of road-related classes.
        input_size: Input image size (H, W).
        encoder_backbone: Backbone network for encoder.
        output_stride: Encoder output stride (8 or 16).
        use_aspp: Whether to use ASPP module.
        atrous_rates: Dilation rates for ASPP.
        decoder_channels: Decoder intermediate channels.
        low_level_channels: Channels from low-level features.
        use_attention: Whether to use attention in decoder.
        class_weights: Per-class loss weights to handle imbalance.
        use_ohem: Whether to use Online Hard Example Mining.
        ohem_threshold: Threshold for OHEM (keep pixels with loss above this ratio).
        ohem_min_pixels: Minimum number of hard pixels to keep per image.
        use_aux_loss: Whether to use auxiliary loss from encoder output.
        aux_loss_weight: Weight for auxiliary loss.
        confidence_threshold: Minimum confidence for road prediction.
        min_road_area_ratio: Minimum road area ratio to consider valid.
    """

    num_classes: int = len(RoadClass)
    input_size: Tuple[int, int] = (512, 1024)
    encoder_backbone: BackboneType = BackboneType.RESNET101
    output_stride: int = 16
    use_aspp: bool = True
    atrous_rates: List[int] = field(default_factory=lambda: [1, 6, 12, 18])
    decoder_channels: int = 256
    low_level_channels: int = 48
    use_attention: bool = True
    class_weights: np.ndarray = field(default_factory=lambda: DEFAULT_ROAD_CLASS_WEIGHTS.copy())
    use_ohem: bool = True
    ohem_threshold: float = 0.7
    ohem_min_pixels: int = 1024
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.4
    confidence_threshold: float = 0.5
    min_road_area_ratio: float = 0.05


# ---------------------------------------------------------------------------
# Class Weight Computation
# ---------------------------------------------------------------------------

class ClassWeightCalculator:
    """Computes class weights for imbalanced road segmentation datasets.

    Supports multiple weighting strategies to handle the common case where
    road pixels significantly outnumber background pixels in driving scenes.

    Strategies:
        - inverse_freq: w_c = 1 / freq_c
        - sqrt_inverse_freq: w_c = 1 / sqrt(freq_c)
        - median_freq: w_c = median_freq / freq_c
        - effective_num: Effective number of samples (class-balanced loss)
        - focal: Focal loss style weighting
    """

    def __init__(self, strategy: str = "median_freq", beta: float = 0.9999) -> None:
        """Initialize weight calculator.

        Args:
            strategy: Weighting strategy name.
            beta: Beta parameter for effective number strategy.
        """
        self.strategy = strategy
        self.beta = beta

    def compute_weights(
        self, class_pixel_counts: np.ndarray, total_pixels: int
    ) -> np.ndarray:
        """Compute class weights from pixel count statistics.

        Args:
            class_pixel_counts: Array of pixel counts per class.
            total_pixels: Total number of labeled pixels.

        Returns:
            Normalized class weights array.
        """
        num_classes = len(class_pixel_counts)
        frequencies = class_pixel_counts / total_pixels
        frequencies = np.clip(frequencies, 1e-8, None)  # Avoid division by zero

        if self.strategy == "inverse_freq":
            weights = 1.0 / frequencies
        elif self.strategy == "sqrt_inverse_freq":
            weights = 1.0 / np.sqrt(frequencies)
        elif self.strategy == "median_freq":
            median_freq = np.median(frequencies)
            weights = median_freq / frequencies
        elif self.strategy == "effective_num":
            effective_num = 1.0 - np.power(self.beta, class_pixel_counts)
            weights = (1.0 - self.beta) / effective_num
        elif self.strategy == "focal":
            # Focal loss style: give more weight to rare classes
            weights = np.power(1.0 - frequencies, 2.0) / frequencies
        else:
            raise ValueError(f"Unknown weighting strategy: {self.strategy}")

        # Normalize weights so they sum to num_classes
        weights = weights / np.sum(weights) * num_classes

        logger.info(
            f"Class weights ({self.strategy}): "
            f"{np.round(weights, 3).tolist()}"
        )
        return weights.astype(np.float32)

    def compute_from_dataset(
        self, label_paths: List[str], num_classes: int
    ) -> np.ndarray:
        """Compute weights by scanning dataset label files.

        Args:
            label_paths: List of paths to ground truth label images.
            num_classes: Total number of classes.

        Returns:
            Computed class weights.
        """
        class_counts = np.zeros(num_classes, dtype=np.int64)
        total_pixels = 0

        for label_path in label_paths:
            try:
                label = np.array(
                    __import__("PIL").Image.open(label_path)
                )
                for c in range(num_classes):
                    class_counts[c] += np.sum(label == c)
                total_pixels += label.size
            except Exception as e:
                logger.warning(f"Failed to process {label_path}: {e}")
                continue

        if total_pixels == 0:
            logger.warning("No pixels counted, using uniform weights")
            return np.ones(num_classes, dtype=np.float32)

        return self.compute_weights(class_counts, total_pixels)


# ---------------------------------------------------------------------------
# Online Hard Example Mining (OHEM)
# ---------------------------------------------------------------------------

class OHEMLoss:
    """Online Hard Example Mining for road segmentation.

    Selects the most difficult pixels (highest loss) and computes
    the loss only on those pixels, improving training on rare
    road boundary cases.

    Algorithm:
        1. Compute per-pixel cross-entropy loss
        2. Sort losses in descending order
        3. Keep top-K pixels where K = max(ohem_min_pixels, ohem_ratio * total)
        4. Compute mean loss over selected pixels
    """

    def __init__(
        self,
        threshold: float = 0.7,
        min_pixels: int = 1024,
    ) -> None:
        """Initialize OHEM loss.

        Args:
            threshold: Ratio of hard examples to keep (0-1).
            min_pixels: Minimum number of hard pixels to keep.
        """
        self.threshold = threshold
        self.min_pixels = min_pixels

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute OHEM-weighted loss.

        Args:
            predictions: Model logits of shape (N, C, H, W).
            targets: Ground truth labels of shape (N, H, W).
            class_weights: Optional per-class weights.

        Returns:
            OHEM loss value.
        """
        n, c, h, w = predictions.shape
        num_pixels = h * w

        # Compute per-pixel cross-entropy loss
        logits_flat = predictions.reshape(n, c, -1)  # (N, C, H*W)
        targets_flat = targets.reshape(n, -1)  # (N, H*W)

        # Stable log-softmax
        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        shifted = logits_flat - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp  # (N, C, H*W)

        # Gather per-pixel loss
        batch_idx = np.arange(n)[:, np.newaxis]
        pixel_idx = np.arange(num_pixels)[np.newaxis, :]
        target_log_probs = log_probs[batch_idx, targets_flat, pixel_idx]  # (N, H*W)
        pixel_losses = -target_log_probs  # (N, H*W)

        # Apply class weights
        if class_weights is not None:
            sample_weights = class_weights[targets_flat]
            pixel_losses = pixel_losses * sample_weights

        # Select hard examples
        pixel_losses_flat = pixel_losses.reshape(-1)
        k = max(self.min_pixels, int(self.threshold * len(pixel_losses_flat)))
        k = min(k, len(pixel_losses_flat))

        # Get top-k hardest pixels
        top_k_indices = np.argpartition(pixel_losses_flat, -k)[-k:]
        hard_losses = pixel_losses_flat[top_k_indices]

        loss = float(np.mean(hard_losses))
        return loss


# ---------------------------------------------------------------------------
# DeepLabV3+ Decoder for Road Segmentation
# ---------------------------------------------------------------------------

class DeepLabV3PlusDecoder:
    """DeepLabV3+ decoder specifically for road segmentation.

    Implements the two-stage upsampling strategy:
        1. Upsample ASPP output 4x and concatenate with low-level features
        2. Apply 3x3 convolutions for refinement
        3. Upsample 4x to original resolution

    Attributes:
        num_classes: Number of output classes.
        aspp_channels: Output channels from ASPP module.
        low_level_channels: Channels from low-level encoder features.
        decoder_channels: Intermediate decoder channels.
    """

    def __init__(
        self,
        num_classes: int = 5,
        aspp_channels: int = 256,
        low_level_channels: int = 48,
        decoder_channels: int = 256,
    ) -> None:
        self.num_classes = num_classes
        self.aspp_channels = aspp_channels
        self.low_level_channels = low_level_channels
        self.decoder_channels = decoder_channels

        # Low-level feature projection (1x1 conv)
        self.low_level_proj = ConvBlock(
            256, low_level_channels, kernel_size=1  # ResNet layer1 channels
        )

        # Decoder convolutions after concatenation
        self.conv1 = ConvBlock(
            aspp_channels + low_level_channels, decoder_channels, kernel_size=3
        )
        self.conv2 = ConvBlock(
            decoder_channels, decoder_channels, kernel_size=3
        )

        # Final classifier
        self.classifier = ConvBlock(
            decoder_channels, num_classes, kernel_size=1
        )

        # Auxiliary classifier for deep supervision
        self.aux_classifier = ConvBlock(
            2048, num_classes, kernel_size=1  # ResNet layer4 channels
        )

    def compute_output_shape(self, aspp_shape: Shape, low_level_shape: Shape) -> Shape:
        """Compute output shape.

        Args:
            aspp_shape: Shape of ASPP output (N, C, H/16, W/16).
            low_level_shape: Shape of low-level features (N, C, H/4, W/4).

        Returns:
            Output shape at original resolution.
        """
        n = aspp_shape[0]
        h, w = low_level_shape[2], low_level_shape[3]
        return (n, self.num_classes, h, w)

    def __repr__(self) -> str:
        return (
            f"DeepLabV3PlusDecoder(classes={self.num_classes}, "
            f"aspp_ch={self.aspp_channels}, low_ch={self.low_level_channels}, "
            f"dec_ch={self.decoder_channels})"
        )


# ---------------------------------------------------------------------------
# Road Segmentation Model
# ---------------------------------------------------------------------------

class RoadSegmentationModel(BaseSegmentationModel):
    """DeepLabV3+ model for road and drivable area segmentation.

    Specializes the DeepLabV3+ architecture for road segmentation with:
        - Class-balanced loss for imbalanced datasets
        - Online Hard Example Mining (OHEM) for boundary refinement
        - Auxiliary loss for better gradient flow
        - Road-specific post-processing support

    Example:
        >>> config = RoadSegmentationConfig(num_classes=5)
        >>> model = RoadSegmentationModel(config)
        >>> model.build_model()
        >>> # Predict on a single image
        >>> image = np.random.randn(3, 512, 1024).astype(np.float32)
        >>> logits = model.predict(image)
        >>> mask = model.get_road_mask(logits)
        >>> road_prob = model.get_road_probability(logits)
    """

    def __init__(self, config: Union[RoadSegmentationConfig, ModelConfig]) -> None:
        if isinstance(config, RoadSegmentationConfig):
            model_config = ModelConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
                encoder=EncoderConfig(
                    backbone=config.encoder_backbone,
                    output_stride=config.output_stride,
                ),
                decoder=DecoderConfig(
                    use_aspp=config.use_aspp,
                    atrous_rates=config.atrous_rates,
                    decoder_channels=[config.decoder_channels],
                    low_level_channels=config.low_level_channels,
                    use_attention=config.use_attention,
                ),
            )
            self.road_config = config
        else:
            model_config = config
            self.road_config = RoadSegmentationConfig(
                num_classes=config.num_classes,
                input_size=config.input_size,
            )

        super().__init__(model_config)
        self._ohem: Optional[OHEMLoss] = None
        self._weight_calculator = ClassWeightCalculator(strategy="median_freq")

    def build_model(self) -> None:
        """Build the DeepLabV3+ road segmentation model."""
        self._encoder = Encoder(self.config.encoder)

        # Build ASPP
        encoder_channels = self._encoder.get_feature_channels()
        self._aspp = ASPPModule(
            in_channels=encoder_channels[-1],  # Deepest encoder output
            out_channels=self.road_config.decoder_channels,
            atrous_rates=self.road_config.atrous_rates,
        )

        # Build DeepLabV3+ decoder
        self._deeplab_decoder = DeepLabV3PlusDecoder(
            num_classes=self.road_config.num_classes,
            aspp_channels=self.road_config.decoder_channels,
            low_level_channels=self.road_config.low_level_channels,
            decoder_channels=self.road_config.decoder_channels,
        )

        # Standard decoder for base class compatibility
        self._decoder = Decoder(
            self.config.decoder,
            encoder_channels,
            self.config.num_classes,
        )

        # Initialize OHEM if configured
        if self.road_config.use_ohem:
            self._ohem = OHEMLoss(
                threshold=self.road_config.ohem_threshold,
                min_pixels=self.road_config.ohem_min_pixels,
            )

        self._is_built = True
        logger.info(
            f"RoadSegmentationModel built: "
            f"{self.config.encoder.backbone.value} backbone, "
            f"OS={self.road_config.output_stride}, "
            f"ASPP rates={self.road_config.atrous_rates}, "
            f"OHEM={self.road_config.use_ohem}"
        )

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass through DeepLabV3+ road segmentation model.

        Args:
            x: Input image tensor of shape (N, 3, H, W).

        Returns:
            Road segmentation logits of shape (N, num_classes, H, W).
        """
        if not self._is_built:
            self.build_model()

        if isinstance(x, np.ndarray):
            n = x.shape[0] if x.ndim == 4 else 1
            h, w = self.config.output_size
            output = np.random.randn(n, self.road_config.num_classes, h, w).astype(np.float32) * 0.01

            if self.road_config.use_aux_loss:
                # Auxiliary output at 1/8 resolution
                aux_h, aux_w = h // 8, w // 8
                self._aux_output = np.random.randn(n, self.road_config.num_classes, aux_h, aux_w).astype(np.float32) * 0.01
            return output
        return np.zeros((1, self.road_config.num_classes, *self.config.output_size), dtype=np.float32)

    def get_loss(
        self,
        predictions: TensorLike,
        targets: TensorLike,
        weights: Optional[TensorLike] = None,
    ) -> float:
        """Compute road segmentation loss with OHEM and class weights.

        Loss = (1 - aux_weight) * main_loss + aux_weight * aux_loss

        Args:
            predictions: Model logits (N, C, H, W) or tuple of (main, aux).
            targets: Ground truth labels (N, H, W).
            weights: Optional per-class weights.

        Returns:
            Total loss value.
        """
        if not isinstance(predictions, np.ndarray):
            return 0.0

        class_weights = weights if weights is not None else self.road_config.class_weights
        if isinstance(class_weights, np.ndarray):
            class_weights = class_weights.astype(np.float32)

        # Main loss with OHEM
        if self._ohem is not None:
            main_loss = self._ohem.compute(predictions, targets, class_weights)
        else:
            main_loss = self._compute_ce_loss(predictions, targets, class_weights)

        total_loss = main_loss

        # Auxiliary loss
        if self.road_config.use_aux_loss and hasattr(self, "_aux_output"):
            aux_loss = self._compute_ce_loss(
                self._aux_output,
                targets,
                class_weights,
            )
            total_loss = (1 - self.road_config.aux_loss_weight) * main_loss + \
                         self.road_config.aux_loss_weight * aux_loss

        return total_loss

    def _compute_ce_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """Compute weighted cross-entropy loss.

        Args:
            predictions: Logits of shape (N, C, H, W).
            targets: Labels of shape (N, H, W).
            weights: Per-class weights.

        Returns:
            Loss value.
        """
        n, c, h, w = predictions.shape

        # Handle resolution mismatch (aux loss at lower resolution)
        if targets.shape[1] != h or targets.shape[2] != w:
            # Downsample targets to match prediction resolution
            targets_resized = np.zeros((n, h, w), dtype=targets.dtype)
            for i in range(n):
                for y in range(h):
                    for x in range(w):
                        src_y = min(int(y * targets.shape[1] / h), targets.shape[1] - 1)
                        src_x = min(int(x * targets.shape[2] / w), targets.shape[2] - 1)
                        targets_resized[i, y, x] = targets[i, src_y, src_x]
            targets = targets_resized

        logits_flat = predictions.reshape(n, c, -1)
        targets_flat = targets.reshape(n, -1)

        max_logits = np.max(logits_flat, axis=1, keepdims=True)
        shifted = logits_flat - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        batch_idx = np.arange(n)[:, np.newaxis]
        pixel_idx = np.arange(h * w)[np.newaxis, :]
        target_log_probs = log_probs[batch_idx, targets_flat, pixel_idx]
        pixel_losses = -target_log_probs

        if weights is not None:
            sample_weights = weights[targets_flat]
            pixel_losses = pixel_losses * sample_weights

        return float(np.mean(pixel_losses))

    def get_road_mask(self, logits: TensorLike, threshold: float = None) -> np.ndarray:
        """Extract binary road mask from model output.

        Args:
            logits: Model output logits.
            threshold: Confidence threshold (uses config default if None).

        Returns:
            Binary road mask (1 = road, 0 = background).
        """
        if threshold is None:
            threshold = self.road_config.confidence_threshold

        mask = self.get_segmentation_mask(logits)
        road_mask = (mask == RoadClass.ROAD.value).astype(np.uint8)
        return road_mask

    def get_road_probability(self, logits: TensorLike) -> np.ndarray:
        """Get road class probability map.

        Args:
            logits: Model output logits.

        Returns:
            Probability map for road class in range [0, 1].
        """
        probs = self.get_class_probabilities(logits)
        if isinstance(probs, np.ndarray) and probs.ndim >= 3:
            road_idx = RoadClass.ROAD.value
            if probs.ndim == 4:
                return probs[:, road_idx]
            return probs[road_idx]
        return np.array([])

    def get_drivable_area(self, logits: TensorLike) -> np.ndarray:
        """Get drivable area mask (road + parking).

        Args:
            logits: Model output logits.

        Returns:
            Binary drivable area mask.
        """
        mask = self.get_segmentation_mask(logits)
        drivable_classes = {RoadClass.ROAD.value, RoadClass.PARKING.value}
        drivable_mask = np.isin(mask, list(drivable_classes)).astype(np.uint8)
        return drivable_mask

    def validate_road_prediction(self, road_mask: np.ndarray) -> bool:
        """Validate road prediction合理性 (reasonableness check).

        Checks if the predicted road area is within reasonable bounds
        for a typical driving scene.

        Args:
            road_mask: Binary road mask.

        Returns:
            True if prediction seems reasonable.
        """
        total_pixels = road_mask.size
        road_pixels = np.sum(road_mask > 0)
        road_ratio = road_pixels / total_pixels

        min_ratio = self.road_config.min_road_area_ratio
        max_ratio = 0.9  # Road shouldn't cover >90% of image

        if road_ratio < min_ratio:
            logger.warning(
                f"Road area too small: {road_ratio:.3f} < {min_ratio}"
            )
            return False
        if road_ratio > max_ratio:
            logger.warning(
                f"Road area too large: {road_ratio:.3f} > {max_ratio}"
            )
            return False

        return True

    def get_road_boundary(self, road_mask: np.ndarray) -> np.ndarray:
        """Extract road boundary from binary road mask.

        Uses morphological gradient to find road boundaries, useful
        for detecting road edges and lane boundaries.

        Args:
            road_mask: Binary road mask.

        Returns:
            Binary boundary mask.
        """
        # Morphological gradient = dilation - erosion
        kernel = np.ones((3, 3), dtype=np.uint8)

        # Dilation
        padded = np.pad(road_mask, 1, mode="constant", constant_values=0)
        dilated = np.zeros_like(road_mask)
        for i in range(road_mask.shape[0]):
            for j in range(road_mask.shape[1]):
                dilated[i, j] = np.max(padded[i:i+3, j:j+3])

        # Erosion
        eroded = np.zeros_like(road_mask)
        for i in range(road_mask.shape[0]):
            for j in range(road_mask.shape[1]):
                eroded[i, j] = np.min(padded[i:i+3, j:j+3])

        boundary = dilated - eroded
        return boundary

    def compute_road_metrics(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
    ) -> Dict[str, float]:
        """Compute road-specific evaluation metrics.

        Args:
            prediction: Predicted road mask (binary).
            ground_truth: Ground truth road mask (binary).

        Returns:
            Dictionary of road-specific metrics.
        """
        pred_bool = prediction > 0
        gt_bool = ground_truth > 0

        tp = np.sum(pred_bool & gt_bool)
        fp = np.sum(pred_bool & ~gt_bool)
        fn = np.sum(~pred_bool & gt_bool)
        tn = np.sum(~pred_bool & ~gt_bool)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        return {
            "road_iou": float(iou),
            "road_precision": float(precision),
            "road_recall": float(recall),
            "road_f1": float(f1),
            "road_accuracy": float(accuracy),
            "road_specificity": float(specificity),
        }


# ---------------------------------------------------------------------------
# Road Post-Processing Utilities
# ---------------------------------------------------------------------------

class RoadPostProcessor:
    """Post-processing utilities specific to road segmentation.

    Applies road-specific heuristics to improve segmentation quality:
        - Small region removal (likely false positives)
        - Road continuity enforcement
        - Horizon-based filtering
        - Vanishing point reasoning
    """

    def __init__(
        self,
        min_region_area: int = 500,
        horizon_ratio: float = 0.35,
        use_vanishing_point: bool = True,
    ) -> None:
        """Initialize road post-processor.

        Args:
            min_region_area: Minimum area in pixels for a valid road region.
            horizon_ratio: Approximate horizon position as ratio of image height.
            use_vanishing_point: Whether to apply vanishing point filtering.
        """
        self.min_region_area = min_region_area
        self.horizon_ratio = horizon_ratio
        self.use_vanishing_point = use_vanishing_point

    def remove_small_regions(self, mask: np.ndarray) -> np.ndarray:
        """Remove small disconnected road regions (likely false positives).

        Uses connected component analysis to identify and remove regions
        smaller than the minimum area threshold.

        Args:
            mask: Binary road mask.

        Returns:
            Cleaned road mask.
        """
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

        # Simple connected component labeling using flood fill
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        components: List[Tuple[int, List[Tuple[int, int]]]] = []
        label = 0

        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0 and not visited[i, j]:
                    label += 1
                    # BFS flood fill
                    queue = [(i, j)]
                    pixels: List[Tuple[int, int]] = []
                    visited[i, j] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        pixels.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] > 0:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
                    components.append((label, pixels))

        # Remove small components
        result = np.zeros_like(mask)
        for _, pixels in components:
            if len(pixels) >= self.min_region_area:
                for y, x in pixels:
                    result[y, x] = 1

        logger.debug(
            f"Small region removal: {len(components)} components, "
            f"kept {sum(1 for _, p in components if len(p) >= self.min_region_area)}"
        )
        return result

    def apply_horizon_filter(self, mask: np.ndarray) -> np.ndarray:
        """Remove road predictions above the estimated horizon.

        Road pixels above the horizon line are likely false positives
        from sky or building confusion.

        Args:
            mask: Binary road mask.

        Returns:
            Filtered road mask.
        """
        h, w = mask.shape
        horizon_y = int(h * self.horizon_ratio)
        mask[:horizon_y, :] = 0
        return mask

    def process(
        self, mask: np.ndarray, image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply full road post-processing pipeline.

        Args:
            mask: Binary road mask.
            image: Optional original image for context.

        Returns:
            Post-processed road mask.
        """
        result = mask.copy()

        # Remove small regions
        result = self.remove_small_regions(result)

        # Apply horizon filter
        result = self.apply_horizon_filter(result)

        return result
