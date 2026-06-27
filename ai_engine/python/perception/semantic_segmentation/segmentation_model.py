"""
Base Segmentation Model Module for Autonomous Vehicle Perception.

Provides abstract base classes and concrete implementations for semantic segmentation
models used in autonomous driving scenarios. Includes encoder-decoder architectures
with skip connections, supporting various backbone networks.

Architecture Overview:
    ┌──────────┐     ┌──────────┐     ┌──────────┐
    │ Encoder  │────▶│  Bridge  │────▶│ Decoder  │
    │ (ResNet) │     │ (ASPP)   │     │ (Upsample)│
    └────┬─────┘     └──────────┘     └────┬─────┘
         │                                  │
         └────── Skip Connections ──────────┘

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
TensorLike = Union[np.ndarray, Any]  # Supports numpy and torch tensors
Shape = Tuple[int, ...]


class BackboneType(Enum):
    """Supported encoder backbone architectures."""

    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNEXT101 = "resnext101_32x8d"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B7 = "efficientnet_b7"
    SWIN_BASE = "swin_base_patch4_window7_224"
    MOBILENET_V3 = "mobilenet_v3_large"
    CONVNEXT_BASE = "convnext_base"


class NormalizationType(Enum):
    """Normalization layer types."""

    BATCH = "batch"
    GROUP = "group"
    LAYER = "layer"
    INSTANCE = "instance"
    SYNC_BATCH = "sync_batch"


class ActivationType(Enum):
    """Activation function types."""

    RELU = "relu"
    RELU6 = "relu6"
    GELU = "gelu"
    SWISH = "swish"
    MISH = "mish"
    LEAKY_RELU = "leaky_relu"
    HARD_SWISH = "hard_swish"


@dataclass
class EncoderConfig:
    """Configuration for the encoder backbone.

    Attributes:
        backbone: Backbone architecture type.
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_bn: Whether to freeze batch normalization layers.
        freeze_stages: Number of early stages to freeze (0 = none, 4 = all).
        output_stride: Output stride for the encoder (8 or 16).
        normalization: Type of normalization layer.
        activation: Type of activation function.
        in_channels: Number of input channels (3 for RGB).
        drop_path_rate: Stochastic depth rate for training.
    """

    backbone: BackboneType = BackboneType.RESNET101
    pretrained: bool = True
    freeze_bn: bool = False
    freeze_stages: int = 1
    output_stride: int = 16
    normalization: NormalizationType = NormalizationType.BATCH
    activation: ActivationType = ActivationType.RELU
    in_channels: int = 3
    drop_path_rate: float = 0.2


@dataclass
class DecoderConfig:
    """Configuration for the decoder module.

    Attributes:
        decoder_channels: List of channel sizes for each decoder stage.
        atrous_rates: Dilation rates for ASPP module.
        use_aspp: Whether to use Atrous Spatial Pyramid Pooling.
        use_attention: Whether to use attention gates in skip connections.
        attention_type: Type of attention ('cbam', 'se', 'eca', 'self').
        upsample_mode: Upsampling method ('bilinear', 'nearest', 'pixel_shuffle').
        use_skip: Whether to use skip connections from encoder.
        low_level_channels: Number of channels from low-level encoder features.
    """

    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    atrous_rates: List[int] = field(default_factory=lambda: [1, 6, 12, 18])
    use_aspp: bool = True
    use_attention: bool = True
    attention_type: str = "cbam"
    upsample_mode: str = "bilinear"
    use_skip: bool = True
    low_level_channels: int = 48


@dataclass
class SkipConnectionConfig:
    """Configuration for skip connections.

    Attributes:
        enabled: Whether skip connections are enabled.
        connection_type: Type of skip connection ('add', 'concat', 'attention').
        num_skips: Number of skip connections (1-4).
        projection_channels: Channels for 1x1 projection before concatenation.
        apply_bn: Whether to apply batch normalization after projection.
        apply_relu: Whether to apply ReLU after projection.
    """

    enabled: bool = True
    connection_type: str = "concat"
    num_skips: int = 4
    projection_channels: int = 48
    apply_bn: bool = True
    apply_relu: bool = True


@dataclass
class ModelConfig:
    """Complete model configuration.

    Attributes:
        num_classes: Number of segmentation classes.
        encoder: Encoder configuration.
        decoder: Decoder configuration.
        skip: Skip connection configuration.
        input_size: Expected input size (H, W).
        output_size: Output size (H, W), defaults to input_size.
        use_deep_supervision: Whether to use deep supervision during training.
        deep_supervision_weights: Weights for auxiliary losses.
        dropout_rate: Dropout rate before final classification.
    """

    num_classes: int = 19
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    skip: SkipConnectionConfig = field(default_factory=SkipConnectionConfig)
    input_size: Tuple[int, int] = (512, 1024)
    output_size: Tuple[int, int] = (512, 1024)
    use_deep_supervision: bool = False
    deep_supervision_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    dropout_rate: float = 0.1


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ConvBlock:
    """Convolutional block with optional normalization and activation.

    Implements: Conv2d -> [Norm] -> [Activation]

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of convolution kernel.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        groups: Number of convolution groups.
        bias: Whether to use bias.
        normalization: Normalization type.
        activation: Activation type.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        normalization: NormalizationType = NormalizationType.BATCH,
        activation: ActivationType = ActivationType.RELU,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.normalization = normalization
        self.activation = activation

        if padding is None:
            self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        else:
            self.padding = padding

        # Parameter initialization
        self._weights: Optional[np.ndarray] = None
        self._bias_value: Optional[np.ndarray] = None
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Kaiming normal initialization."""
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        gain = math.sqrt(2.0)  # ReLU gain
        std = gain / math.sqrt(fan_in)
        self._weights = np.random.normal(0, std, (self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size))
        if self.bias:
            self._bias_value = np.zeros(self.out_channels)

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        """Compute output shape given input shape.

        Args:
            input_shape: Input tensor shape (N, C, H, W).

        Returns:
            Output tensor shape.
        """
        n, _, h, w = input_shape
        h_out = (h + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        w_out = (w + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        return (n, self.out_channels, h_out, w_out)

    def __repr__(self) -> str:
        return (
            f"ConvBlock({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"norm={self.normalization.value}, act={self.activation.value})"
        )


class BottleneckBlock:
    """ResNet-style bottleneck block with 1x1-3x3-1x1 convolutions.

    Implements:
        1x1 conv (compress) -> 3x3 conv -> 1x1 conv (expand)
        with optional shortcut connection.

    Attributes:
        in_channels: Input channels.
        bottleneck_channels: Mid-layer channels (compressed).
        out_channels: Output channels.
        stride: Stride for 3x3 convolution.
        dilation: Dilation for 3x3 convolution.
        downsample: Whether a shortcut downsample is needed.
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: bool = False,
    ) -> None:
        self.in_channels = in_channels
        self.bottleneck_channels = bottleneck_channels
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = ConvBlock(
            in_channels, bottleneck_channels, kernel_size=1, stride=1
        )
        self.conv2 = ConvBlock(
            bottleneck_channels, bottleneck_channels, kernel_size=3,
            stride=stride, dilation=dilation,
        )
        self.conv3 = ConvBlock(
            bottleneck_channels, out_channels, kernel_size=1, stride=1
        )

        if downsample:
            self.shortcut = ConvBlock(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.shortcut = None

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        """Compute output shape given input shape."""
        shape = self.conv1.compute_output_shape(input_shape)
        shape = self.conv2.compute_output_shape(shape)
        shape = self.conv3.compute_output_shape(shape)
        return shape


class ASPPModule:
    """Atrous Spatial Pyramid Pooling module.

    Captures multi-scale context by applying parallel atrous convolutions
    at different rates, then concatenating the results.

    Architecture:
        Input ─┬── 1x1 Conv ──────────────┐
               ├── 3x3 Conv (rate=6) ──────┤
               ├── 3x3 Conv (rate=12) ─────┤── Concat ── 1x1 Conv ── Output
               ├── 3x3 Conv (rate=18) ─────┤
               └── Global AvgPool + Upsample┘

    Attributes:
        in_channels: Input channels.
        out_channels: Output channels per branch.
        atrous_rates: List of dilation rates.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        atrous_rates: List[int] = None,
    ) -> None:
        if atrous_rates is None:
            atrous_rates = [1, 6, 12, 18]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates

        # 1x1 convolution branch
        self.branches: List[ConvBlock] = []

        # 1x1 conv branch
        self.branches.append(
            ConvBlock(in_channels, out_channels, kernel_size=1)
        )

        # Atrous conv branches
        for rate in atrous_rates:
            if rate == 1:
                continue  # Already handled above
            self.branches.append(
                ConvBlock(
                    in_channels, out_channels, kernel_size=3,
                    padding=rate, dilation=rate,
                )
            )

        # Global average pooling branch (simulated)
        self.global_pool_channels = out_channels
        self.global_pool_conv = ConvBlock(in_channels, out_channels, kernel_size=1)

        # Project after concatenation
        total_channels = (len(self.branches) + 1) * out_channels
        self.project = ConvBlock(total_channels, out_channels, kernel_size=1)

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        """Compute output shape."""
        n, _, h, w = input_shape
        return (n, self.out_channels, h, w)

    def __repr__(self) -> str:
        return (
            f"ASPPModule(in={self.in_channels}, out={self.out_channels}, "
            f"rates={self.atrous_rates})"
        )


class AttentionGate:
    """Attention gate for skip connections.

    Computes attention weights to focus on relevant features from the
    skip connection using gating signal from the decoder.

    Architecture:
        Skip ──── W_x ──┐
                         ├── Add -> ReLU -> W_psi -> Sigmoid -> Weights
        Gate ──── W_g ──┘

    Attributes:
        skip_channels: Number of channels in skip connection features.
        gate_channels: Number of channels in gating signal.
        intermediate_channels: Number of intermediate channels.
    """

    def __init__(
        self,
        skip_channels: int,
        gate_channels: int,
        intermediate_channels: Optional[int] = None,
    ) -> None:
        if intermediate_channels is None:
            intermediate_channels = max(skip_channels // 2, 32)

        self.skip_channels = skip_channels
        self.gate_channels = gate_channels
        self.intermediate_channels = intermediate_channels

        self.W_x = ConvBlock(skip_channels, intermediate_channels, kernel_size=1, bias=True)
        self.W_g = ConvBlock(gate_channels, intermediate_channels, kernel_size=1, bias=True)
        self.W_psi = ConvBlock(intermediate_channels, 1, kernel_size=1, bias=True)

    def compute_attention_weights(
        self, skip_features: TensorLike, gate_features: TensorLike
    ) -> TensorLike:
        """Compute attention weights for skip features.

        Args:
            skip_features: Features from encoder skip connection.
            gate_features: Gating signal from decoder.

        Returns:
            Attention weights in range [0, 1].
        """
        # In a real implementation, this would use torch operations
        # Here we simulate the attention computation
        logger.debug(
            f"Computing attention: skip={skip_features.shape}, "
            f"gate={gate_features.shape}"
        )
        # Placeholder: actual computation happens in torch/numpy
        if isinstance(skip_features, np.ndarray):
            # Simple simulation: channel-wise attention
            weights = np.mean(skip_features, axis=1, keepdims=True)
            weights = 1.0 / (1.0 + np.exp(-weights))  # sigmoid
            return weights
        return np.ones_like(skip_features) * 0.5

    def __repr__(self) -> str:
        return (
            f"AttentionGate(skip={self.skip_channels}, gate={self.gate_channels}, "
            f"inter={self.intermediate_channels})"
        )


class SEBlock:
    """Squeeze-and-Excitation block for channel attention.

    Recalibrates channel-wise feature responses through
    global average pooling and fully connected layers.

    Architecture:
        Input ── Global AvgPool ── FC (reduce) ── ReLU ── FC (expand) ── Sigmoid ── Scale ── Output

    Attributes:
        channels: Number of input/output channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 8)

        self.fc_reduce = ConvBlock(channels, self.reduced_channels, kernel_size=1)
        self.fc_expand = ConvBlock(self.reduced_channels, channels, kernel_size=1)

    def compute_output(self, x: TensorLike) -> TensorLike:
        """Apply SE attention to input features.

        Args:
            x: Input features of shape (N, C, H, W).

        Returns:
            Channel-attention-weighted features.
        """
        if isinstance(x, np.ndarray):
            # Squeeze: global average pooling
            squeeze = np.mean(x, axis=(2, 3), keepdims=True)  # (N, C, 1, 1)
            # Excitation: FC -> ReLU -> FC -> Sigmoid
            scale = 1.0 / (1.0 + np.exp(-squeeze))  # Sigmoid approximation
            return x * scale
        return x


# ---------------------------------------------------------------------------
# Encoder Module
# ---------------------------------------------------------------------------

class Encoder:
    """Feature extraction encoder based on configurable backbone.

    Supports ResNet, EfficientNet, and Swin Transformer backbones with
    configurable output stride and feature extraction at multiple scales.

    Attributes:
        config: Encoder configuration.
        feature_channels: List of output channel sizes at each stage.
    """

    # Default feature channels for each backbone
    BACKbone_CHANNELS: Dict[BackboneType, List[int]] = {
        BackboneType.RESNET50: [64, 256, 512, 1024, 2048],
        BackboneType.RESNET101: [64, 256, 512, 1024, 2048],
        BackboneType.RESNEXT101: [64, 256, 512, 1024, 2048],
        BackboneType.EFFICIENTNET_B4: [32, 56, 160, 448, 1792],
        BackboneType.EFFICIENTNET_B7: [32, 48, 224, 672, 2560],
        BackboneType.SWIN_BASE: [96, 192, 384, 768, 1024],
        BackboneType.MOBILENET_V3: [16, 24, 48, 96, 576],
        BackboneType.CONVNEXT_BASE: [128, 128, 256, 512, 1024],
    }

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.feature_channels = self.BACKbone_CHANNELS.get(
            config.backbone, [64, 256, 512, 1024, 2048]
        )
        self._stages: List[Any] = []
        self._build_encoder()

    def _build_encoder(self) -> None:
        """Build encoder stages based on configuration."""
        logger.info(
            f"Building encoder with {self.config.backbone.value} backbone, "
            f"output_stride={self.config.output_stride}"
        )

        backbone = self.config.backbone
        channels = self.feature_channels

        # Build stem (initial conv + maxpool for ResNet-like)
        if backbone.value.startswith("res") or backbone.value.startswith("convnext"):
            self._stages.append(
                ConvBlock(self.config.in_channels, channels[0], kernel_size=7, stride=2, padding=3)
            )

        # Build residual stages
        for i in range(1, len(channels)):
            in_ch = channels[i - 1] if i == 1 else channels[i]
            out_ch = channels[i]
            stride = 2 if i < len(channels) - 1 else 1
            if self.config.output_stride == 8 and i >= 3:
                stride = 1
            elif self.config.output_stride == 16 and i >= 4:
                stride = 1

            self._stages.append(
                BottleneckBlock(in_ch, out_ch // 4, out_ch, stride=stride)
            )

        logger.info(
            f"Encoder built: {len(self._stages)} stages, "
            f"channels={self.feature_channels}"
        )

    def get_feature_channels(self) -> List[int]:
        """Return the output channel sizes at each encoder stage.

        Returns:
            List of channel sizes for skip connections.
        """
        return self.feature_channels[1:]  # Exclude stem

    def compute_shapes(self, input_shape: Shape) -> List[Shape]:
        """Compute output shapes at each encoder stage.

        Args:
            input_shape: Input tensor shape (N, C, H, W).

        Returns:
            List of shapes at each stage.
        """
        shapes = [input_shape]
        current_shape = input_shape
        for stage in self._stages:
            if isinstance(stage, ConvBlock):
                current_shape = stage.compute_output_shape(current_shape)
            elif isinstance(stage, BottleneckBlock):
                current_shape = stage.compute_output_shape(current_shape)
            shapes.append(current_shape)
        return shapes

    def freeze_layers(self) -> None:
        """Freeze encoder layers according to configuration."""
        freeze_count = min(self.config.freeze_stages, len(self._stages))
        for i in range(freeze_count):
            logger.debug(f"Freezing encoder stage {i}")
        if self.config.freeze_bn:
            logger.info("Freezing batch normalization in encoder")
        logger.info(f"Frozen {freeze_count}/{len(self._stages)} encoder stages")


# ---------------------------------------------------------------------------
# Decoder Module
# ---------------------------------------------------------------------------

class Decoder:
    """Feature decoder with skip connections and progressive upsampling.

    Supports both standard decoder and DeepLabV3+ style decoder with
    optional ASPP module and attention gates.

    Attributes:
        config: Decoder configuration.
        encoder_channels: Channel sizes from encoder skip connections.
    """

    def __init__(
        self,
        config: DecoderConfig,
        encoder_channels: List[int],
        num_classes: int,
    ) -> None:
        self.config = config
        self.encoder_channels = encoder_channels
        self.num_classes = num_classes
        self._blocks: List[Any] = []
        self._skip_projections: List[Any] = []
        self._attention_gates: List[Any] = []
        self._build_decoder()

    def _build_decoder(self) -> None:
        """Build decoder stages with skip connections."""
        enc_channels = list(reversed(self.encoder_channels))
        dec_channels = self.config.decoder_channels

        logger.info(
            f"Building decoder: enc_channels={self.encoder_channels}, "
            f"dec_channels={dec_channels}"
        )

        # ASPP module at the bottleneck
        if self.config.use_aspp:
            self.aspp = ASPPModule(
                enc_channels[0],
                out_channels=dec_channels[0],
                atrous_rates=self.config.atrous_rates,
            )
            current_channels = dec_channels[0]
        else:
            self.aspp = None
            current_channels = enc_channels[0]

        # Decoder blocks with skip connections
        for i in range(min(len(dec_channels), len(enc_channels) - 1)):
            skip_ch = enc_channels[i + 1]

            # Low-level feature projection (DeepLabV3+ style)
            projection = ConvBlock(
                skip_ch, self.config.low_level_channels, kernel_size=1
            )
            self._skip_projections.append(projection)

            # Attention gate
            if self.config.use_attention:
                gate = AttentionGate(
                    skip_channels=self.config.low_level_channels,
                    gate_channels=current_channels,
                )
                self._attention_gates.append(gate)
            else:
                self._attention_gates.append(None)

            # After concatenation: decoder channels + projected skip channels
            combined_channels = current_channels + self.config.low_level_channels

            # Decoder conv block
            block = ConvBlock(
                combined_channels, dec_channels[i], kernel_size=3
            )
            self._blocks.append(block)
            current_channels = dec_channels[i]

        # Final classifier
        self.classifier = ConvBlock(
            current_channels, self.num_classes, kernel_size=1,
            normalization=NormalizationType.BATCH,
            activation=ActivationType.RELU,
        )

        logger.info(
            f"Decoder built: {len(self._blocks)} blocks, "
            f"ASPP={self.config.use_aspp}, attention={self.config.use_attention}"
        )

    def compute_output_shape(self, input_shapes: List[Shape]) -> Shape:
        """Compute decoder output shape.

        Args:
            input_shapes: List of encoder feature shapes from deep to shallow.

        Returns:
            Final output shape.
        """
        # Start with deepest features
        current_shape = input_shapes[0]
        n = current_shape[0]

        # ASPP preserves spatial dimensions
        if self.aspp is not None:
            current_shape = (n, self.config.decoder_channels[0], current_shape[2], current_shape[3])

        # Progressive upsampling
        h, w = current_shape[2], current_shape[3]
        for i in range(len(self._blocks)):
            h *= 2
            w *= 2
            current_shape = (n, self.config.decoder_channels[i], h, w)

        return (n, self.num_classes, h, w)


# ---------------------------------------------------------------------------
# Abstract Base Segmentation Model
# ---------------------------------------------------------------------------

class BaseSegmentationModel(abc.ABC):
    """Abstract base class for all segmentation models.

    Defines the interface that all concrete segmentation models must implement.
    Provides common functionality for model lifecycle management, inference,
    and feature extraction.

    Subclasses must implement:
        - build_model(): Construct the model architecture
        - forward(): Forward pass computation
        - get_loss(): Loss computation

    Example:
        >>> config = ModelConfig(num_classes=19)
        >>> model = RoadSegmentationModel(config)
        >>> output = model.predict(image)
        >>> mask = model.get_segmentation_mask(output)
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._encoder: Optional[Encoder] = None
        self._decoder: Optional[Decoder] = None
        self._is_built = False
        self._training = True
        self._device = "cpu"
        self._parameter_count = 0
        self._flops_estimate = 0

    @abc.abstractmethod
    def build_model(self) -> None:
        """Construct the complete model architecture.

        Must set self._encoder and self._decoder.
        """
        ...

    @abc.abstractmethod
    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass through the model.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Output logits of shape (N, num_classes, H', W').
        """
        ...

    @abc.abstractmethod
    def get_loss(
        self,
        predictions: TensorLike,
        targets: TensorLike,
        weights: Optional[TensorLike] = None,
    ) -> float:
        """Compute loss between predictions and targets.

        Args:
            predictions: Model output logits.
            targets: Ground truth labels.
            weights: Optional per-class weights.

        Returns:
            Computed loss value.
        """
        ...

    def predict(self, x: TensorLike) -> TensorLike:
        """Run inference on input.

        Automatically handles model building, eval mode, and gradient context.

        Args:
            x: Input tensor of shape (N, C, H, W) or (C, H, W).

        Returns:
            Prediction logits.
        """
        if not self._is_built:
            self.build_model()
            self._is_built = True

        was_training = self._training
        self._training = False

        # Handle single image input
        single_image = False
        if isinstance(x, np.ndarray):
            if x.ndim == 3:
                x = x[np.newaxis, ...]
                single_image = True

        output = self.forward(x)

        if single_image and isinstance(output, np.ndarray):
            output = output[0]

        self._training = was_training
        return output

    def get_segmentation_mask(
        self, logits: TensorLike, threshold: float = 0.5
    ) -> np.ndarray:
        """Convert model logits to segmentation mask.

        Args:
            logits: Raw model output logits.
            threshold: Confidence threshold for binary segmentation.

        Returns:
            Segmentation mask as numpy array with class indices.
        """
        if isinstance(logits, np.ndarray):
            if logits.ndim == 3:
                # Multi-class: take argmax along class dimension
                if logits.shape[0] > 1:
                    return np.argmax(logits, axis=0).astype(np.uint8)
                else:
                    return (logits[0] > threshold).astype(np.uint8)
            elif logits.ndim == 4:
                if logits.shape[1] > 1:
                    return np.argmax(logits, axis=1).astype(np.uint8)
                else:
                    return (logits[:, 0] > threshold).astype(np.uint8)
        return np.zeros((1,), dtype=np.uint8)

    def get_class_probabilities(self, logits: TensorLike) -> np.ndarray:
        """Convert logits to class probabilities using softmax.

        Args:
            logits: Raw model output.

        Returns:
            Class probability map.
        """
        if isinstance(logits, np.ndarray):
            # Numerically stable softmax
            shifted = logits - np.max(logits, axis=0 if logits.ndim == 3 else 1, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=0 if logits.ndim == 3 else 1, keepdims=True)
        return np.array([])

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        if self._parameter_count > 0:
            return self._parameter_count

        total = 0
        if self._encoder is not None:
            for stage in self._encoder._stages:
                if isinstance(stage, ConvBlock):
                    total += np.prod(stage._weights.shape) if stage._weights is not None else 0
                elif isinstance(stage, BottleneckBlock):
                    for conv in [stage.conv1, stage.conv2, stage.conv3]:
                        total += np.prod(conv._weights.shape) if conv._weights is not None else 0
                    if stage.shortcut is not None:
                        total += np.prod(stage.shortcut._weights.shape) if stage.shortcut._weights is not None else 0

        self._parameter_count = total
        return total

    def estimate_flops(self, input_shape: Shape = (1, 3, 512, 1024)) -> int:
        """Estimate FLOPs for a single forward pass.

        Args:
            input_shape: Input tensor shape.

        Returns:
            Estimated FLOPs count.
        """
        if self._flops_estimate > 0:
            return self._flops_estimate

        h, w = input_shape[2], input_shape[3]
        total_flops = 0

        if self._encoder is not None:
            for stage in self._encoder._stages:
                if isinstance(stage, ConvBlock):
                    # FLOPs ≈ 2 * K^2 * C_in * C_out * H * W
                    k = stage.kernel_size
                    total_flops += 2 * k * k * stage.in_channels * stage.out_channels * h * w
                    if stage.stride == 2:
                        h //= 2
                        w //= 2

        # Decoder has roughly 1/4 of encoder FLOPs
        total_flops += total_flops // 4

        self._flops_estimate = total_flops
        return total_flops

    def summary(self) -> str:
        """Generate model summary string.

        Returns:
            Formatted summary of model architecture and parameters.
        """
        params = self.count_parameters()
        flops = self.estimate_flops()

        lines = [
            "=" * 60,
            f"  {self.__class__.__name__} Summary",
            "=" * 60,
            f"  Backbone:        {self.config.encoder.backbone.value}",
            f"  Num Classes:     {self.config.num_classes}",
            f"  Input Size:      {self.config.input_size}",
            f"  Output Stride:   {self.config.encoder.output_stride}",
            f"  ASPP:            {self.config.decoder.use_aspp}",
            f"  Attention:       {self.config.decoder.use_attention}",
            f"  Skip Connect:    {self.config.skip.enabled}",
            "-" * 60,
            f"  Parameters:      {params:,}",
            f"  FLOPs:           {flops:,}",
            f"  Model Size:      {params * 4 / 1024 / 1024:.1f} MB (float32)",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to(self, device: str) -> "BaseSegmentationModel":
        """Move model to specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.).

        Returns:
            Self for method chaining.
        """
        self._device = device
        logger.info(f"Moving model to {device}")
        return self

    def train(self) -> "BaseSegmentationModel":
        """Set model to training mode."""
        self._training = True
        return self

    def eval(self) -> "BaseSegmentationModel":
        """Set model to evaluation mode."""
        self._training = False
        return self

    def save_config(self, path: str) -> None:
        """Save model configuration to file.

        Args:
            path: Output file path.
        """
        import json
        from dataclasses import asdict

        config_dict = {
            "num_classes": self.config.num_classes,
            "encoder": {
                "backbone": self.config.encoder.backbone.value,
                "pretrained": self.config.encoder.pretrained,
                "output_stride": self.config.encoder.output_stride,
            },
            "decoder": {
                "use_aspp": self.config.decoder.use_aspp,
                "use_attention": self.config.decoder.use_attention,
                "atrous_rates": self.config.decoder.atrous_rates,
            },
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Config saved to {path}")

    @classmethod
    def from_config(cls, config_path: str) -> "BaseSegmentationModel":
        """Create model from configuration file.

        Args:
            config_path: Path to configuration JSON file.

        Returns:
            Model instance with loaded configuration.
        """
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = ModelConfig(
            num_classes=config_dict.get("num_classes", 19),
        )
        if "encoder" in config_dict:
            enc = config_dict["encoder"]
            config.encoder.backbone = BackboneType(enc.get("backbone", "resnet101"))
            config.encoder.output_stride = enc.get("output_stride", 16)

        instance = cls(config)
        return instance


# ---------------------------------------------------------------------------
# Concrete Implementation: UNet-style Segmentation Model
# ---------------------------------------------------------------------------

class UNetSegmentationModel(BaseSegmentationModel):
    """UNet-style segmentation model with skip connections.

    Implements the classic UNet architecture with configurable backbone
    encoder and progressive decoder with skip connections.

    Example:
        >>> config = ModelConfig(num_classes=19, encoder=EncoderConfig(backbone=BackboneType.RESNET50))
        >>> model = UNetSegmentationModel(config)
        >>> model.build_model()
        >>> print(model.summary())
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._skip_features: List[TensorLike] = []

    def build_model(self) -> None:
        """Build the UNet model with encoder and decoder."""
        self._encoder = Encoder(self.config.encoder)
        self._decoder = Decoder(
            self.config.decoder,
            self._encoder.get_feature_channels(),
            self.config.num_classes,
        )
        self._is_built = True
        logger.info(
            f"UNet model built: {self.config.encoder.backbone.value} encoder, "
            f"{self.config.num_classes} classes"
        )

    def forward(self, x: TensorLike) -> TensorLike:
        """Forward pass through UNet.

        Args:
            x: Input tensor of shape (N, C, H, W).

        Returns:
            Segmentation logits of shape (N, num_classes, H, W).
        """
        if not self._is_built:
            self.build_model()

        if isinstance(x, np.ndarray):
            input_shape = x.shape
        else:
            input_shape = (1, 3, 512, 1024)

        # Simulate forward pass
        n = input_shape[0]
        h, w = self.config.output_size

        # In production, this would run through actual torch modules
        # Here we return the expected output shape with random logits
        output = np.random.randn(n, self.config.num_classes, h, w).astype(np.float32) * 0.01

        return output

    def get_loss(
        self,
        predictions: TensorLike,
        targets: TensorLike,
        weights: Optional[TensorLike] = None,
    ) -> float:
        """Compute cross-entropy loss with optional class weights.

        Args:
            predictions: Model logits of shape (N, C, H, W).
            targets: Ground truth of shape (N, H, W) with class indices.
            weights: Optional per-class weights of shape (C,).

        Returns:
            Cross-entropy loss value.
        """
        if isinstance(predictions, np.ndarray) and isinstance(targets, np.ndarray):
            # Numerically stable softmax cross-entropy
            n, c, h, w = predictions.shape
            logits = predictions.reshape(n, c, -1)  # (N, C, H*W)
            targets_flat = targets.reshape(n, -1)  # (N, H*W)

            # Log-softmax
            max_logits = np.max(logits, axis=1, keepdims=True)
            shifted = logits - max_logits
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
            log_probs = shifted - log_sum_exp  # (N, C, H*W)

            # Gather target probabilities
            batch_indices = np.arange(n)[:, np.newaxis]
            pixel_indices = np.arange(h * w)[np.newaxis, :]
            target_probs = np.zeros_like(logits)
            target_probs[batch_indices, targets_flat, pixel_indices] = 1.0

            # Weighted loss
            loss = -np.sum(target_probs * log_probs, axis=1)  # (N, H*W)

            if weights is not None:
                sample_weights = weights[targets_flat]  # (N, H*W)
                loss = loss * sample_weights

            return float(np.mean(loss))

        return 0.0


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Factory for creating segmentation models from configuration.

    Provides a unified interface for instantiating different segmentation
    model architectures based on configuration.

    Example:
        >>> factory = ModelFactory()
        >>> model = factory.create("unet", ModelConfig(num_classes=19))
        >>> model = factory.create_from_yaml("config.yaml")
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class with a name.

        Args:
            name: Model name for lookup.
            model_class: Model class to register.
        """
        cls._registry[name.lower()] = model_class
        logger.info(f"Registered model: {name} -> {model_class.__name__}")

    @classmethod
    def create(cls, model_type: str, config: ModelConfig) -> BaseSegmentationModel:
        """Create a model instance by type name.

        Args:
            model_type: Name of registered model type.
            config: Model configuration.

        Returns:
            Instantiated model.

        Raises:
            ValueError: If model type is not registered.
        """
        model_type = model_type.lower()
        if model_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available: {available}"
            )
        return cls._registry[model_type](config)

    @classmethod
    def create_from_yaml(cls, yaml_path: str) -> BaseSegmentationModel:
        """Create a model from a YAML configuration file.

        Args:
            yaml_path: Path to YAML config file.

        Returns:
            Instantiated model.
        """
        import yaml

        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_type = cfg.get("model", {}).get("type", "unet")
        config = ModelConfig(
            num_classes=cfg.get("model", {}).get("num_classes", 19),
        )
        return cls.create(model_type, config)


# Register built-in models
ModelFactory.register("unet", UNetSegmentationModel)
ModelFactory.register("base", UNetSegmentationModel)
