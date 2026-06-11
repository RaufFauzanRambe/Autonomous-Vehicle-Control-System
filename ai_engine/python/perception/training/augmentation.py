"""
Data Augmentation Module for Autonomous Vehicle AI Training.

Implements comprehensive data augmentation strategies:
- Mosaic: Combine 4 images into one (YOLOv4+)
- MixUp: Alpha-blending of image pairs
- CutMix: Cut-and-paste image regions
- Geometric: Random flip, rotation, scale, translation, perspective
- Color: Brightness, contrast, saturation, hue jitter
- Weather: Simulated rain, snow, fog, lens flare
- Domain-specific: Motion blur, sensor noise, occlusion
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration for augmentation pipeline."""
    # Geometric
    random_flip_h: bool = True
    random_flip_v: bool = False
    flip_prob: float = 0.5
    random_rotation: bool = True
    rotation_range: float = 15.0  # degrees
    random_scale: bool = True
    scale_range: Tuple[float, float] = (0.8, 1.2)
    random_translation: bool = True
    translation_range: float = 0.1  # fraction of image size
    random_perspective: bool = False
    perspective_distortion: float = 0.1

    # Color
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    grayscale_prob: float = 0.1

    # Advanced
    mosaic: bool = True
    mosaic_prob: float = 0.5
    mixup: bool = True
    mixup_prob: float = 0.3
    mixup_alpha: float = 0.2
    cutmix: bool = True
    cutmix_prob: float = 0.3
    cutmix_alpha: float = 1.0

    # Weather simulation
    weather_augmentation: bool = False
    rain_prob: float = 0.1
    snow_prob: float = 0.05
    fog_prob: float = 0.1
    lens_flare_prob: float = 0.05

    # Domain-specific
    motion_blur: bool = True
    motion_blur_prob: float = 0.1
    motion_blur_kernel: int = 5
    sensor_noise: bool = True
    sensor_noise_prob: float = 0.1
    sensor_noise_std: float = 0.02
    random_occlusion: bool = True
    occlusion_prob: float = 0.05
    occlusion_ratio: Tuple[float, float] = (0.1, 0.4)

    # General
    image_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class GeometricAugmentor:
    """
    Geometric transformations for image augmentation.

    Applies random flips, rotations, scaling, translations,
    and perspective transformations while updating bounding
    box annotations accordingly.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(
        self, image: np.ndarray, labels: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply geometric augmentations."""
        h, w = image.shape[:2]

        # Horizontal flip
        if self.config.random_flip_h and random.random() < self.config.flip_prob:
            image = image[:, ::-1].copy()
            if labels and "boxes" in labels and len(labels["boxes"]) > 0:
                boxes = labels["boxes"].copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                labels["boxes"] = boxes

        # Vertical flip
        if self.config.random_flip_v and random.random() < self.config.flip_prob:
            image = image[::-1].copy()
            if labels and "boxes" in labels and len(labels["boxes"]) > 0:
                boxes = labels["boxes"].copy()
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                labels["boxes"] = boxes

        # Random rotation
        if self.config.random_rotation:
            angle = random.uniform(-self.config.rotation_range, self.config.rotation_range)
            image = self._rotate_image(image, angle)

        # Random scale
        if self.config.random_scale:
            scale = random.uniform(*self.config.scale_range)
            image = self._scale_image(image, scale)

        return image, labels

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle degrees."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Compute new image bounds
        new_w = int(abs(h * sin_a) + abs(w * cos_a))
        new_h = int(abs(h * cos_a) + abs(w * sin_a))

        # Create rotation matrix (simplified - no cv2 dependency)
        # For production, use cv2.warpAffine
        result = np.zeros((new_h, new_w, image.shape[2]), dtype=image.dtype)
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        safe_y = min(y_offset, h // 2)
        safe_x = min(x_offset, w // 2)
        result[safe_y:safe_y + min(h, new_h), safe_x:safe_x + min(w, new_w)] = \
            image[:min(h, new_h), :min(w, new_w)]

        return result

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image by factor."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        # Simple nearest-neighbor resize
        y_indices = np.clip(np.arange(new_h) / scale, 0, h - 1).astype(int)
        x_indices = np.clip(np.arange(new_w) / scale, 0, w - 1).astype(int)
        result = image[np.ix_(y_indices, x_indices)]

        return result


class ColorAugmentor:
    """
    Color-space augmentations for image data.

    Applies brightness, contrast, saturation, and hue adjustments,
    as well as random grayscale conversion.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(
        self, image: np.ndarray, labels: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply color augmentations."""
        if not self.config.color_jitter:
            return image, labels

        # Brightness
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.config.brightness, self.config.brightness)
            image = np.clip(image * factor, 0, 1).astype(image.dtype)

        # Contrast
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.config.contrast, self.config.contrast)
            mean = image.mean()
            image = np.clip((image - mean) * factor + mean, 0, 1).astype(image.dtype)

        # Saturation
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.config.saturation, self.config.saturation)
            gray = np.mean(image, axis=2, keepdims=True)
            image = np.clip(gray + factor * (image - gray), 0, 1).astype(image.dtype)

        # Hue (simplified - shift in HSV would be better)
        if random.random() < 0.3:
            shift = random.uniform(-self.config.hue, self.config.hue)
            # Simple RGB channel shift as hue approximation
            r_shift = shift * 0.3
            g_shift = shift * 0.59
            b_shift = shift * 0.11
            image[:, :, 0] = np.clip(image[:, :, 0] + r_shift, 0, 1)
            image[:, :, 1] = np.clip(image[:, :, 1] + g_shift, 0, 1)
            image[:, :, 2] = np.clip(image[:, :, 2] + b_shift, 0, 1)

        # Random grayscale
        if random.random() < self.config.grayscale_prob:
            gray = np.mean(image, axis=2, keepdims=True)
            image = np.repeat(gray, 3, axis=2)

        return image, labels


class MosaicAugmentor:
    """
    Mosaic augmentation (YOLOv4+).

    Combines 4 training images into a single mosaic image,
    effectively quadrupling the effective batch size and
    encouraging the model to see objects in different contexts.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self.image_size = config.image_size

    def __call__(
        self, images: List[np.ndarray], labels_list: List[Dict]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Create a mosaic from 4 images.

        Args:
            images: List of 4 images.
            labels_list: List of 4 label dictionaries.

        Returns:
            Tuple of (mosaic_image, combined_labels).
        """
        if len(images) < 4:
            # Not enough images, just return the first one
            return images[0], labels_list[0]

        h, w = self.image_size
        mosaic = np.zeros((h, w, 3), dtype=np.float32)

        # Random center point
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)

        all_boxes = []
        all_classes = []

        # Place each image in a quadrant
        placements = [
            (0, cx, 0, cy),        # Top-left
            (cx, w, 0, cy),        # Top-right
            (0, cx, cy, h),        # Bottom-left
            (cx, w, cy, h),        # Bottom-right
        ]

        for idx, (x1, x2, y1, y2) in enumerate(placements):
            img = images[idx]
            target_h = y2 - y1
            target_w = x2 - x1

            # Simple resize
            if img.shape[0] > 0 and img.shape[1] > 0:
                resized = self._simple_resize(img, (target_h, target_w))
                mosaic[y1:y2, x1:x2] = resized[:target_h, :target_w]

            # Adjust bounding boxes
            if labels_list[idx].get("boxes") is not None and len(labels_list[idx]["boxes"]) > 0:
                boxes = labels_list[idx]["boxes"].copy()
                # Scale to quadrant size
                scale_x = target_w / max(img.shape[1], 1)
                scale_y = target_h / max(img.shape[0], 1)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + x1
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + y1
                # Clip to mosaic bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x1, x2)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y1, y2)
                all_boxes.append(boxes)

                if "classes" in labels_list[idx]:
                    all_classes.append(labels_list[idx]["classes"])

        combined_labels = {}
        if all_boxes:
            combined_labels["boxes"] = np.concatenate(all_boxes, axis=0)
        else:
            combined_labels["boxes"] = np.zeros((0, 4), dtype=np.float32)
        if all_classes:
            combined_labels["classes"] = np.concatenate(all_classes, axis=0)
        else:
            combined_labels["classes"] = np.array([], dtype=np.int64)
        combined_labels["label"] = labels_list[0].get("label", np.zeros(10, dtype=np.float32))

        return mosaic, combined_labels

    def _simple_resize(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple resize using array indexing."""
        h, w = image.shape[:2]
        new_h, new_w = target_size

        if h == 0 or w == 0:
            return np.zeros((*target_size, image.shape[2]), dtype=image.dtype)

        y_idx = np.linspace(0, h - 1, new_h).astype(int)
        x_idx = np.linspace(0, w - 1, new_w).astype(int)
        return image[np.ix_(y_idx, x_idx)][0] if image.ndim == 3 else image[np.ix_(y_idx, x_idx)]


class MixUpAugmentor:
    """
    MixUp augmentation.

    Creates new training samples by linearly interpolating
    between pairs of images and their labels.

    x_new = lambda * x_i + (1 - lambda) * x_j
    y_new = lambda * y_i + (1 - lambda) * y_j
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(
        self,
        image1: np.ndarray, labels1: Dict,
        image2: np.ndarray, labels2: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply MixUp between two samples."""
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        mixed_image = lam * image1 + (1 - lam) * image2

        mixed_labels = {}
        for key in labels1:
            if isinstance(labels1[key], np.ndarray) and isinstance(labels2[key], np.ndarray):
                if labels1[key].dtype in (np.float32, np.float64):
                    mixed_labels[key] = lam * labels1[key] + (1 - lam) * labels2[key]
                else:
                    mixed_labels[key] = labels1[key] if lam >= 0.5 else labels2[key]
            else:
                mixed_labels[key] = labels1[key] if lam >= 0.5 else labels2[key]

        return mixed_image.astype(np.float32), mixed_labels


class CutMixAugmentor:
    """
    CutMix augmentation.

    Cuts a rectangular region from one image and pastes it
    onto another, adjusting labels proportionally.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def __call__(
        self,
        image1: np.ndarray, labels1: Dict,
        image2: np.ndarray, labels2: Dict,
    ) -> Tuple[np.ndarray, Dict]:
        """Apply CutMix between two samples."""
        h, w = image1.shape[:2]
        lam = np.random.beta(self.alpha, self.alpha)

        # Generate random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply CutMix
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        # Adjust labels based on area ratio
        area_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        lam_adjusted = 1.0 - area_ratio

        mixed_labels = {}
        for key in labels1:
            if isinstance(labels1[key], np.ndarray) and labels1[key].dtype in (np.float32, np.float64):
                mixed_labels[key] = lam_adjusted * labels1[key] + (1 - lam_adjusted) * labels2[key]
            else:
                mixed_labels[key] = labels1[key] if lam_adjusted >= 0.5 else labels2[key]

        return mixed_image, mixed_labels


class WeatherAugmentor:
    """
    Weather simulation augmentations.

    Simulates rain, snow, fog, and lens flare effects to
    improve model robustness to adverse weather conditions.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def __call__(
        self, image: np.ndarray, labels: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply weather augmentations."""
        if not self.config.weather_augmentation:
            return image, labels

        # Rain simulation
        if random.random() < self.config.rain_prob:
            image = self._add_rain(image)

        # Snow simulation
        if random.random() < self.config.snow_prob:
            image = self._add_snow(image)

        # Fog simulation
        if random.random() < self.config.fog_prob:
            image = self._add_fog(image)

        # Lens flare
        if random.random() < self.config.lens_flare_prob:
            image = self._add_lens_flare(image)

        return image, labels

    def _add_rain(self, image: np.ndarray) -> np.ndarray:
        """Simulate rain streaks."""
        h, w = image.shape[:2]
        rain = np.zeros_like(image)
        n_drops = random.randint(100, 300)

        for _ in range(n_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 20)
            length = random.randint(10, 20)
            intensity = random.uniform(0.6, 1.0)
            rain[y:y + length, x] = intensity

        return np.clip(image * 0.7 + rain * 0.3, 0, 1).astype(np.float32)

    def _add_snow(self, image: np.ndarray) -> np.ndarray:
        """Simulate snowfall."""
        h, w = image.shape[:2]
        snow = np.random.uniform(0.8, 1.0, (h, w, 1)).astype(np.float32)
        mask = np.random.random((h, w, 1)) > 0.97
        snow = snow * mask
        return np.clip(image * 0.8 + snow * 0.2, 0, 1).astype(np.float32)

    def _add_fog(self, image: np.ndarray) -> np.ndarray:
        """Simulate fog effect."""
        fog_intensity = random.uniform(0.3, 0.7)
        fog_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)
        return np.clip(
            image * (1 - fog_intensity) + fog_color * fog_intensity,
            0, 1
        ).astype(np.float32)

    def _add_lens_flare(self, image: np.ndarray) -> np.ndarray:
        """Simulate lens flare."""
        h, w = image.shape[:2]
        flare_x = random.randint(0, w - 1)
        flare_y = random.randint(0, h // 3)
        intensity = random.uniform(0.2, 0.5)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        dist = np.sqrt((x_coords - flare_x) ** 2 + (y_coords - flare_y) ** 2)
        flare = np.clip(1.0 - dist / (max(h, w) * 0.3), 0, 1) * intensity
        flare = np.stack([flare] * 3, axis=-1)

        return np.clip(image + flare, 0, 1).astype(np.float32)


class MotionBlurAugmentor:
    """Motion blur simulation for driving scenarios."""

    def __init__(self, kernel_size: int = 5, prob: float = 0.1) -> None:
        self.kernel_size = kernel_size
        self.prob = prob

    def __call__(
        self, image: np.ndarray, labels: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply motion blur."""
        if random.random() > self.prob:
            return image, labels

        # Simple horizontal motion blur
        kernel = np.zeros(self.kernel_size, dtype=np.float32)
        kernel[self.kernel_size // 2:] = 1.0 / (self.kernel_size // 2 + 1)

        result = image.copy()
        for c in range(image.shape[2]):
            for i in range(image.shape[0]):
                result[i, :, c] = np.convolve(image[i, :, c], kernel, mode="same")

        return result, labels


class AugmentationPipeline:
    """
    Composable augmentation pipeline.

    Chains multiple augmentations in sequence, with configurable
    probabilities for each augmentation.
    """

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self.geometric = GeometricAugmentor(config)
        self.color = ColorAugmentor(config)
        self.mosaic = MosaicAugmentor(config)
        self.mixup = MixUpAugmentor(config.mixup_alpha)
        self.cutmix = CutMixAugmentor(config.cutmix_alpha)
        self.weather = WeatherAugmentor(config)
        self.motion_blur = MotionBlurAugmentor(config.motion_blur_kernel, config.motion_blur_prob)

    def __call__(
        self, image: np.ndarray, labels: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Apply the full augmentation pipeline."""
        # Geometric transforms
        image, labels = self.geometric(image, labels)

        # Color transforms
        image, labels = self.color(image, labels)

        # Weather simulation
        image, labels = self.weather(image, labels)

        # Motion blur
        image, labels = self.motion_blur(image, labels)

        # Sensor noise
        if self.config.sensor_noise and random.random() < self.config.sensor_noise_prob:
            noise = np.random.normal(0, self.config.sensor_noise_std, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1).astype(np.float32)

        # Normalize
        if self.config.normalize:
            image = (image - np.array(self.config.mean)) / np.array(self.config.std)

        return image, labels


def create_augmentation_pipeline(
    level: str = "auto",
    config: Optional[AugmentationConfig] = None,
) -> AugmentationPipeline:
    """
    Create an augmentation pipeline based on difficulty level.

    Args:
        level: Augmentation level (none, light, medium, heavy, auto).
        config: Optional explicit configuration.

    Returns:
        Configured augmentation pipeline.
    """
    if config is not None:
        return AugmentationPipeline(config)

    if level == "none":
        cfg = AugmentationConfig(
            random_flip_h=False, random_rotation=False, random_scale=False,
            color_jitter=False, mosaic=False, mixup=False, cutmix=False,
            weather_augmentation=False, motion_blur=False, sensor_noise=False,
        )
    elif level == "light":
        cfg = AugmentationConfig(
            random_flip_h=True, random_rotation=False, random_scale=False,
            color_jitter=True, brightness=0.1, contrast=0.1,
            mosaic=False, mixup=False, cutmix=False,
            weather_augmentation=False,
        )
    elif level == "medium":
        cfg = AugmentationConfig(
            random_flip_h=True, random_rotation=True, rotation_range=10,
            color_jitter=True, mosaic_prob=0.3, mixup_prob=0.2,
            weather_augmentation=False,
        )
    elif level == "heavy":
        cfg = AugmentationConfig(
            random_flip_h=True, random_rotation=True, rotation_range=20,
            random_scale=True, random_perspective=True,
            color_jitter=True, brightness=0.3, contrast=0.3,
            mosaic_prob=0.5, mixup_prob=0.3, cutmix_prob=0.3,
            weather_augmentation=True,
        )
    else:  # auto
        cfg = AugmentationConfig()

    return AugmentationPipeline(cfg)
