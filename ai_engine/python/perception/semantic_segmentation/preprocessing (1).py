"""
Preprocessing Module for Semantic Segmentation.

Provides comprehensive image and label preprocessing pipeline for
segmentation tasks in autonomous driving scenarios.

Pipeline:
    Raw Image ──▶ Resize ──▶ Pad ──▶ Normalize ──▶ Color Jitter ──▶ To Tensor

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Configuration for preprocessing pipeline.

    Attributes:
        target_size: Target image size (H, W).
        resize_mode: Resize interpolation ('bilinear', 'nearest', 'area').
        pad_mode: Padding mode ('constant', 'reflect', 'edge', 'symmetric').
        pad_value: Padding value for constant mode.
        pad_to_multiple: Pad dimensions to be multiple of this value.
        normalize: Whether to apply normalization.
        mean: Normalization mean per channel.
        std: Normalization std per channel.
        use_color_jitter: Whether to apply color jitter.
        brightness: Brightness jitter range.
        contrast: Contrast jitter range.
        saturation: Saturation jitter range.
        hue: Hue jitter range.
        use_random_grayscale: Whether to randomly convert to grayscale.
        grayscale_prob: Probability of grayscale conversion.
        use_gaussian_blur: Whether to apply Gaussian blur.
        blur_kernel_range: Range of blur kernel sizes.
        blur_prob: Probability of applying blur.
        use_random_crop: Whether to apply random crop.
        crop_size: Random crop size (H, W).
        crop_scale: Scale range for random resized crop.
        crop_ratio: Aspect ratio range for random resized crop.
        use_center_crop: Whether to apply center crop.
        use_random_rotation: Whether to apply random rotation.
        rotation_range: Maximum rotation angle in degrees.
        use_cutout: Whether to apply cutout regularization.
        cutout_size: Size of cutout region.
        cutout_prob: Probability of applying cutout.
    """

    target_size: Tuple[int, int] = (512, 1024)
    resize_mode: str = "bilinear"
    pad_mode: str = "constant"
    pad_value: int = 0
    pad_to_multiple: int = 32
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    use_color_jitter: bool = False
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    hue: float = 0.02
    use_random_grayscale: bool = False
    grayscale_prob: float = 0.1
    use_gaussian_blur: bool = False
    blur_kernel_range: Tuple[int, int] = (3, 7)
    blur_prob: float = 0.1
    use_random_crop: bool = True
    crop_size: Tuple[int, int] = (512, 1024)
    crop_scale: Tuple[float, float] = (0.5, 2.0)
    crop_ratio: Tuple[float, float] = (0.75, 1.333)
    use_center_crop: bool = False
    use_random_rotation: bool = False
    rotation_range: float = 10.0
    use_cutout: bool = False
    cutout_size: int = 32
    cutout_prob: float = 0.1


# ---------------------------------------------------------------------------
# Individual Preprocessing Functions
# ---------------------------------------------------------------------------

def resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """Resize image to target size.

    Args:
        image: Input image (H, W, C) or (H, W).
        target_size: Target size (H', W').
        mode: Interpolation mode ('bilinear', 'nearest', 'area').

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    th, tw = target_size

    if (h, w) == (th, tw):
        return image.copy()

    ndim = image.ndim

    if mode == "nearest":
        if ndim == 3:
            result = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
            for y in range(th):
                src_y = min(int(y * h / th), h - 1)
                for x in range(tw):
                    src_x = min(int(x * w / tw), w - 1)
                    result[y, x] = image[src_y, src_x]
        else:
            result = np.zeros(target_size, dtype=image.dtype)
            for y in range(th):
                src_y = min(int(y * h / th), h - 1)
                for x in range(tw):
                    src_x = min(int(x * w / tw), w - 1)
                    result[y, x] = image[src_y, src_x]

    elif mode == "bilinear":
        if ndim == 3:
            result = np.zeros((th, tw, image.shape[2]), dtype=np.float32)
            for y in range(th):
                src_y = y * (h - 1) / max(th - 1, 1)
                y0 = int(np.floor(src_y))
                y1 = min(y0 + 1, h - 1)
                wy = src_y - y0
                for x in range(tw):
                    src_x = x * (w - 1) / max(tw - 1, 1)
                    x0 = int(np.floor(src_x))
                    x1 = min(x0 + 1, w - 1)
                    wx = src_x - x0
                    result[y, x] = (
                        image[y0, x0] * (1 - wy) * (1 - wx) +
                        image[y0, x1] * (1 - wy) * wx +
                        image[y1, x0] * wy * (1 - wx) +
                        image[y1, x1] * wy * wx
                    )
            result = result.astype(image.dtype)
        else:
            result = np.zeros(target_size, dtype=np.float32)
            for y in range(th):
                src_y = y * (h - 1) / max(th - 1, 1)
                y0 = int(np.floor(src_y))
                y1 = min(y0 + 1, h - 1)
                wy = src_y - y0
                for x in range(tw):
                    src_x = x * (w - 1) / max(tw - 1, 1)
                    x0 = int(np.floor(src_x))
                    x1 = min(x0 + 1, w - 1)
                    wx = src_x - x0
                    result[y, x] = (
                        image[y0, x0] * (1 - wy) * (1 - wx) +
                        image[y0, x1] * (1 - wy) * wx +
                        image[y1, x0] * wy * (1 - wx) +
                        image[y1, x1] * wy * wx
                    )
            result = result.astype(image.dtype)

    elif mode == "area":
        # Area-based downsampling
        if ndim == 3:
            result = np.zeros((th, tw, image.shape[2]), dtype=np.float32)
            scale_y = h / th
            scale_x = w / tw
            for y in range(th):
                y_start = int(y * scale_y)
                y_end = int((y + 1) * scale_y)
                for x in range(tw):
                    x_start = int(x * scale_x)
                    x_end = int((x + 1) * scale_x)
                    region = image[y_start:y_end, x_start:x_end]
                    result[y, x] = np.mean(region, axis=(0, 1))
            result = result.astype(image.dtype)
        else:
            result = np.zeros(target_size, dtype=np.float32)
            scale_y = h / th
            scale_x = w / tw
            for y in range(th):
                y_start = int(y * scale_y)
                y_end = int((y + 1) * scale_y)
                for x in range(tw):
                    x_start = int(x * scale_x)
                    x_end = int((x + 1) * scale_x)
                    result[y, x] = np.mean(image[y_start:y_end, x_start:x_end])
            result = result.astype(image.dtype)
    else:
        raise ValueError(f"Unknown resize mode: {mode}")

    return result


def resize_label(
    label: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Resize segmentation label using nearest neighbor interpolation.

    Args:
        label: Label image (H, W).
        target_size: Target size (H', W').

    Returns:
        Resized label.
    """
    return resize(label, target_size, mode="nearest")


def pad(
    image: np.ndarray,
    target_size: Tuple[int, int],
    mode: str = "constant",
    value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad image to target size.

    Args:
        image: Input image (H, W, C) or (H, W).
        target_size: Target size (H', W').
        mode: Padding mode.
        value: Padding value for constant mode.

    Returns:
        Tuple of (padded image, padding (top, left, bottom, right)).
    """
    h, w = image.shape[:2]
    th, tw = target_size

    pad_top = max(0, (th - h) // 2)
    pad_bottom = max(0, th - h - pad_top)
    pad_left = max(0, (tw - w) // 2)
    pad_right = max(0, tw - w - pad_left)

    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return image.copy(), (0, 0, 0, 0)

    if image.ndim == 3:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    if mode == "constant":
        padded = np.pad(image, pad_width, mode="constant", constant_values=value)
    elif mode == "reflect":
        padded = np.pad(image, pad_width, mode="reflect")
    elif mode == "edge":
        padded = np.pad(image, pad_width, mode="edge")
    elif mode == "symmetric":
        padded = np.pad(image, pad_width, mode="symmetric")
    else:
        padded = np.pad(image, pad_width, mode="constant", constant_values=value)

    return padded, (pad_top, pad_left, pad_bottom, pad_right)


def pad_to_multiple(
    image: np.ndarray,
    multiple: int = 32,
    value: int = 0,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Pad image so dimensions are multiples of a given value.

    Args:
        image: Input image (H, W, C) or (H, W).
        multiple: Target multiple.
        value: Padding value.

    Returns:
        Tuple of (padded image, padding).
    """
    h, w = image.shape[:2]

    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    return pad(image, (new_h, new_w), value=value)


def normalize(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize image with per-channel mean and standard deviation.

    Args:
        image: Input image (H, W, C) in range [0, 255] or [0, 1].
        mean: Per-channel mean.
        std: Per-channel std.

    Returns:
        Normalized image.
    """
    result = image.astype(np.float32)

    # Auto-detect range
    if result.max() > 1.5:
        result /= 255.0

    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)

    result = (result - mean_arr) / std_arr
    return result


def denormalize(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize image back to [0, 255] range.

    Args:
        image: Normalized image (H, W, C).
        mean: Per-channel mean used during normalization.
        std: Per-channel std used during normalization.

    Returns:
        Denormalized image in [0, 255] uint8.
    """
    mean_arr = np.array(mean).reshape(1, 1, 3)
    std_arr = np.array(std).reshape(1, 1, 3)

    result = image * std_arr + mean_arr
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def color_jitter(
    image: np.ndarray,
    brightness: float = 0.3,
    contrast: float = 0.3,
    saturation: float = 0.3,
    hue: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Apply random color jittering to image.

    Args:
        image: Input image (H, W, C) float32 or uint8.
        brightness: Brightness jitter factor.
        contrast: Contrast jitter factor.
        saturation: Saturation jitter factor.
        hue: Hue jitter factor.
        rng: Random state for reproducibility.

    Returns:
        Color-jittered image.
    """
    if rng is None:
        rng = np.random.RandomState()

    result = image.copy().astype(np.float32)

    # Brightness
    if brightness > 0:
        factor = rng.uniform(max(0, 1 - brightness), 1 + brightness)
        result *= factor

    # Contrast
    if contrast > 0:
        factor = rng.uniform(max(0, 1 - contrast), 1 + contrast)
        mean = np.mean(result)
        result = (result - mean) * factor + mean

    # Saturation
    if saturation > 0 and result.shape[2] == 3:
        factor = rng.uniform(max(0, 1 - saturation), 1 + saturation)
        gray = np.mean(result, axis=2, keepdims=True)
        result = gray + factor * (result - gray)

    # Hue
    if hue > 0 and result.shape[2] == 3:
        factor = rng.uniform(-hue, hue)
        # Convert to HSV, shift hue, convert back
        result_rgb = np.clip(result / 255.0 if result.max() > 1 else result, 0, 1)
        # Simplified hue shift by rotating channels
        r, g, b = result_rgb[:, :, 0], result_rgb[:, :, 1], result_rgb[:, :, 2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Compute hue
        hue_map = np.zeros_like(r)
        mask = delta > 0
        # Simplified: rotate RGB channels slightly
        shift = int(factor * 10)  # Small channel rotation
        if shift != 0:
            channels = [result[:, :, i] for i in range(3)]
            shifted = channels[shift % 3:] + channels[:shift % 3]
            for i in range(3):
                result[:, :, i] = shifted[i]

    if image.dtype == np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    else:
        result = np.clip(result, 0.0, 1.0).astype(np.float32)

    return result


def random_crop(
    image: np.ndarray,
    label: np.ndarray,
    crop_size: Tuple[int, int],
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random crop to image and label.

    Args:
        image: Input image (H, W, C).
        label: Input label (H, W).
        crop_size: Crop size (H', W').
        rng: Random state.

    Returns:
        Tuple of (cropped image, cropped label).
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w = image.shape[:2]
    ch, cw = crop_size

    if h < ch or w < cw:
        # Pad if image is smaller than crop
        image, _ = pad(image, (max(h, ch), max(w, cw)))
        label, _ = pad(label, (max(h, ch), max(w, cw)), value=255)
        h, w = image.shape[:2]

    y_start = rng.randint(0, h - ch + 1)
    x_start = rng.randint(0, w - cw + 1)

    image_crop = image[y_start:y_start + ch, x_start:x_start + cw]
    label_crop = label[y_start:y_start + ch, x_start:x_start + cw]

    return image_crop, label_crop


def center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int],
) -> np.ndarray:
    """Apply center crop to image.

    Args:
        image: Input image (H, W, C) or (H, W).
        crop_size: Crop size (H', W').

    Returns:
        Center-cropped image.
    """
    h, w = image.shape[:2]
    ch, cw = crop_size

    y_start = (h - ch) // 2
    x_start = (w - cw) // 2

    return image[y_start:y_start + ch, x_start:x_start + cw]


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """Apply Gaussian blur to image.

    Args:
        image: Input image (H, W, C).
        kernel_size: Kernel size (must be odd).
        sigma: Gaussian sigma.

    Returns:
        Blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Generate 1D Gaussian kernel
    x = np.arange(kernel_size) - kernel_size // 2
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    result = image.copy().astype(np.float32)
    h, w = result.shape[:2]
    pad_size = kernel_size // 2

    # Horizontal pass
    padded = np.pad(result, ((0, 0), (pad_size, pad_size), (0, 0)), mode="edge")
    temp = np.zeros_like(result)
    for i in range(kernel_size):
        temp += padded[:, i:i + w, :] * kernel_1d[i]

    # Vertical pass
    padded = np.pad(temp, ((pad_size, pad_size), (0, 0), (0, 0)), mode="edge")
    result = np.zeros_like(result)
    for i in range(kernel_size):
        result += padded[i:i + h, :, :] * kernel_1d[i]

    return result.astype(image.dtype)


def cutout(
    image: np.ndarray,
    size: int = 32,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Apply cutout regularization to image.

    Randomly masks a square region of the image.

    Args:
        image: Input image (H, W, C).
        size: Size of the cutout square.
        rng: Random state.

    Returns:
        Image with cutout applied.
    """
    if rng is None:
        rng = np.random.RandomState()

    h, w = image.shape[:2]
    result = image.copy()

    y = rng.randint(0, h)
    x = rng.randint(0, w)

    y1 = max(0, y - size // 2)
    y2 = min(h, y + size // 2)
    x1 = max(0, x - size // 2)
    x2 = min(w, x + size // 2)

    if result.ndim == 3:
        result[y1:y2, x1:x2, :] = 0
    else:
        result[y1:y2, x1:x2] = 0

    return result


# ---------------------------------------------------------------------------
# Complete Preprocessing Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """Complete preprocessing pipeline for segmentation.

    Applies a configurable sequence of preprocessing operations
    to both images and labels, ensuring consistent transformations.

    Example:
        >>> config = PreprocessConfig(target_size=(512, 1024))
        >>> pipeline = PreprocessingPipeline(config)
        >>> processed = pipeline(image, label, is_training=True)
        >>> image_out = processed["image"]
        >>> label_out = processed["label"]
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """Initialize preprocessing pipeline.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config
        self.rng = np.random.RandomState(42)

    def __call__(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None,
        is_training: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Apply preprocessing pipeline.

        Args:
            image: Input image (H, W, C).
            label: Optional label (H, W).
            is_training: Whether in training mode (enables augmentation).

        Returns:
            Dictionary with 'image' and optionally 'label'.
        """
        result_image = image.copy()
        result_label = label.copy() if label is not None else None
        padding_info = (0, 0, 0, 0)

        # 1. Resize
        result_image = resize(result_image, self.config.target_size, self.config.resize_mode)
        if result_label is not None:
            result_label = resize_label(result_label, self.config.target_size)

        # 2. Pad to multiple
        if self.config.pad_to_multiple > 1:
            result_image, padding_info = pad_to_multiple(
                result_image, self.config.pad_to_multiple, self.config.pad_value
            )
            if result_label is not None:
                result_label, _ = pad_to_multiple(
                    result_label, self.config.pad_to_multiple, value=255
                )

        # Training-only augmentations
        if is_training:
            # 3. Random crop
            if self.config.use_random_crop and result_label is not None:
                result_image, result_label = random_crop(
                    result_image, result_label, self.config.crop_size, self.rng
                )

            # 4. Color jitter
            if self.config.use_color_jitter:
                result_image = color_jitter(
                    result_image,
                    brightness=self.config.brightness,
                    contrast=self.config.contrast,
                    saturation=self.config.saturation,
                    hue=self.config.hue,
                    rng=self.rng,
                )

            # 5. Gaussian blur
            if self.config.use_gaussian_blur and self.rng.random() < self.config.blur_prob:
                kernel_size = self.rng.randint(
                    self.config.blur_kernel_range[0],
                    self.config.blur_kernel_range[1] + 1,
                )
                if kernel_size % 2 == 0:
                    kernel_size += 1
                result_image = gaussian_blur(result_image, kernel_size)

            # 6. Cutout
            if self.config.use_cutout and self.rng.random() < self.config.cutout_prob:
                result_image = cutout(
                    result_image, self.config.cutout_size, self.rng
                )
        else:
            # Center crop for validation
            if self.config.use_center_crop:
                result_image = center_crop(result_image, self.config.target_size)
                if result_label is not None:
                    result_label = center_crop(result_label, self.config.target_size)

        # 7. Normalize
        if self.config.normalize:
            result_image = normalize(result_image, self.config.mean, self.config.std)

        output = {
            "image": result_image,
            "padding": padding_info,
        }

        if result_label is not None:
            output["label"] = result_label

        return output

    def preprocess_batch(
        self,
        images: np.ndarray,
        labels: Optional[np.ndarray] = None,
        is_training: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Preprocess a batch of images.

        Args:
            images: Batch of images (N, H, W, C).
            labels: Optional batch of labels (N, H, W).
            is_training: Whether in training mode.

        Returns:
            Dictionary with batched preprocessed data.
        """
        n = images.shape[0]
        processed_images = []
        processed_labels = []

        for i in range(n):
            result = self(
                images[i],
                labels[i] if labels is not None else None,
                is_training,
            )
            processed_images.append(result["image"])
            if labels is not None:
                processed_labels.append(result["label"])

        output = {"image": np.stack(processed_images)}
        if processed_labels:
            output["label"] = np.stack(processed_labels)

        return output
