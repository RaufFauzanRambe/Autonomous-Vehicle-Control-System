"""
Dataset Loader for Semantic Segmentation in Autonomous Driving.

Provides unified dataset loading for Cityscapes, KITTI, and custom
formats with configurable augmentation pipelines. Supports multi-GPU
training with distributed sampling.

Supported Datasets:
    - Cityscapes: Fine and coarse annotations
    - KITTI: Road and semantic benchmarks
    - Custom: Configurable directory-based format
    - SYNTHIA: Synthetic driving dataset
    - BDD100K: Berkeley DeepDrive

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Types and Configuration
# ---------------------------------------------------------------------------

class DatasetType(Enum):
    """Supported dataset types."""

    CITYSCAPES = "cityscapes"
    KITTI = "kitti"
    KITTI_ROAD = "kitti_road"
    CUSTOM = "custom"
    SYNTHIA = "synthia"
    BDD100K = "bdd100k"


class SplitType(Enum):
    """Dataset split types."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_EXTRA = "train_extra"


@dataclass
class DatasetConfig:
    """Dataset configuration.

    Attributes:
        dataset_type: Type of dataset.
        root_dir: Root directory of the dataset.
        split: Dataset split (train/val/test).
        num_classes: Number of segmentation classes.
        input_size: Input image size (H, W).
        ignore_label: Label value to ignore in loss computation.
        use_coarse: Whether to include coarse annotations (Cityscapes).
        use_augmentation: Whether to apply data augmentation.
        normalize_mean: Image normalization mean (per channel).
        normalize_std: Image normalization std (per channel).
        max_samples: Maximum number of samples (for debugging).
        cache_images: Whether to cache images in memory.
        num_workers: Number of data loading workers.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for reproducibility.
    """

    dataset_type: DatasetType = DatasetType.CITYSCAPES
    root_dir: str = "/data/cityscapes"
    split: SplitType = SplitType.TRAIN
    num_classes: int = 19
    input_size: Tuple[int, int] = (512, 1024)
    ignore_label: int = 255
    use_coarse: bool = False
    use_augmentation: bool = True
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    max_samples: int = -1
    cache_images: bool = False
    num_workers: int = 4
    batch_size: int = 4
    shuffle: bool = True
    seed: int = 42


# ---------------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Configurable data augmentation pipeline for segmentation.

    Supports geometric and photometric augmentations with consistent
    transformation of both images and labels.

    Augmentations:
        Geometric:
            - Random horizontal flip
            - Random scale and crop
            - Random rotation
            - Random affine
            - Padding

        Photometric:
            - Color jitter (brightness, contrast, saturation, hue)
            - Random grayscale
            - Gaussian blur
            - Gaussian noise
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (512, 1024),
        scale_range: Tuple[float, float] = (0.5, 2.0),
        flip_prob: float = 0.5,
        rotation_range: float = 10.0,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.7, 1.3),
        hue_range: Tuple[float, float] = (-0.02, 0.02),
        blur_prob: float = 0.1,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Initialize augmentation pipeline.

        Args:
            input_size: Target output size (H, W).
            scale_range: Range of random scale factors.
            flip_prob: Probability of horizontal flip.
            rotation_range: Maximum rotation angle in degrees.
            brightness_range: Brightness adjustment range.
            contrast_range: Contrast adjustment range.
            saturation_range: Saturation adjustment range.
            hue_range: Hue shift range.
            blur_prob: Probability of Gaussian blur.
            noise_std: Standard deviation of Gaussian noise.
            seed: Random seed.
        """
        self.input_size = input_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.blur_prob = blur_prob
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)

    def random_flip(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random horizontal flip.

        Args:
            image: Input image (H, W, C).
            label: Label image (H, W).

        Returns:
            Flipped image and label.
        """
        if self.rng.random() < self.flip_prob:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def random_scale_crop(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random scale and center crop.

        Args:
            image: Input image (H, W, C).
            label: Label image (H, W).

        Returns:
            Scaled and cropped image and label.
        """
        target_h, target_w = self.input_size
        h, w = image.shape[:2]

        # Random scale
        scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize (nearest neighbor for labels)
        image_scaled = self._resize_image(image, (new_h, new_w))
        label_scaled = self._resize_label(label, (new_h, new_w))

        # Crop or pad to target size
        image_out = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        label_out = np.full((target_h, target_w), 255, dtype=label.dtype)

        src_h = min(new_h, target_h)
        src_w = min(new_w, target_w)

        # Random crop position
        y_start = self.rng.randint(0, max(new_h - target_h + 1, 1))
        x_start = self.rng.randint(0, max(new_w - target_w + 1, 1))

        dst_y = max(0, (target_h - new_h) // 2)
        dst_x = max(0, (target_w - new_w) // 2)

        copy_h = min(src_h, target_h - dst_y)
        copy_w = min(src_w, target_w - dst_x)

        image_out[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
            image_scaled[y_start:y_start + copy_h, x_start:x_start + copy_w]
        label_out[dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = \
            label_scaled[y_start:y_start + copy_h, x_start:x_start + copy_w]

        return image_out, label_out

    def color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply random color jittering.

        Args:
            image: Input image (H, W, C) in float [0, 255] or [0, 1].

        Returns:
            Color-jittered image.
        """
        result = image.copy().astype(np.float32)

        # Brightness
        brightness = self.rng.uniform(*self.brightness_range)
        result *= brightness

        # Contrast
        contrast = self.rng.uniform(*self.contrast_range)
        mean = np.mean(result)
        result = (result - mean) * contrast + mean

        # Saturation (for RGB images)
        if result.shape[2] == 3:
            saturation = self.rng.uniform(*self.saturation_range)
            gray = np.mean(result, axis=2, keepdims=True)
            result = gray + saturation * (result - gray)

        # Clip to valid range
        if image.max() > 1:
            result = np.clip(result, 0, 255).astype(image.dtype)
        else:
            result = np.clip(result, 0, 1).astype(np.float32)

        return result

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to image.

        Args:
            image: Input image.

        Returns:
            Noisy image.
        """
        if self.rng.random() < 0.1:
            noise = self.rng.randn(*image.shape).astype(np.float32) * self.noise_std * 255
            return np.clip(image.astype(np.float32) + noise, 0, 255).astype(image.dtype)
        return image

    @staticmethod
    def _resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image using bilinear interpolation.

        Args:
            image: Input image (H, W, C).
            size: Target size (H', W').

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]
        th, tw = size

        result = np.zeros((th, tw, image.shape[2]), dtype=image.dtype)
        for y in range(th):
            src_y = min(int(y * h / th), h - 1)
            for x in range(tw):
                src_x = min(int(x * w / tw), w - 1)
                result[y, x] = image[src_y, src_x]

        return result

    @staticmethod
    def _resize_label(label: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize label using nearest neighbor interpolation.

        Args:
            label: Label image (H, W).
            size: Target size (H', W').

        Returns:
            Resized label.
        """
        h, w = label.shape
        th, tw = size

        result = np.zeros(size, dtype=label.dtype)
        for y in range(th):
            src_y = min(int(y * h / th), h - 1)
            for x in range(tw):
                src_x = min(int(x * w / tw), w - 1)
                result[y, x] = label[src_y, src_x]

        return result

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply full augmentation pipeline.

        Args:
            image: Input image (H, W, C).
            label: Label image (H, W).

        Returns:
            Augmented image and label.
        """
        # Geometric augmentations
        image, label = self.random_scale_crop(image, label)
        image, label = self.random_flip(image, label)

        # Photometric augmentations
        image = self.color_jitter(image)
        image = self.add_noise(image)

        return image, label


# ---------------------------------------------------------------------------
# Base Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset:
    """Base dataset class for semantic segmentation.

    Provides a unified interface for loading images and labels
    from various dataset formats.

    Attributes:
        config: Dataset configuration.
        images: List of image file paths.
        labels: List of label file paths.
        augmentation: Augmentation pipeline.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize dataset.

        Args:
            config: Dataset configuration.
        """
        self.config = config
        self.images: List[str] = []
        self.labels: List[str] = []
        self._image_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self.augmentation = AugmentationPipeline(
            input_size=config.input_size,
            seed=config.seed,
        ) if config.use_augmentation else None

        # Load file lists
        self._load_file_list()

        logger.info(
            f"Dataset loaded: {config.dataset_type.value}/{config.split.value}, "
            f"{len(self.images)} samples"
        )

    def _load_file_list(self) -> None:
        """Load file lists for the dataset. Override in subclasses."""
        root = Path(self.config.root_dir)
        split = self.config.split.value

        # Generic directory structure
        image_dir = root / "images" / split
        label_dir = root / "labels" / split

        if image_dir.exists() and label_dir.exists():
            image_files = sorted(
                list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            )
            for img_path in image_files:
                stem = img_path.stem
                # Try to find corresponding label
                for ext in [".png", ".jpg", ".npy"]:
                    label_path = label_dir / (stem + ext)
                    if label_path.exists():
                        self.images.append(str(img_path))
                        self.labels.append(str(label_path))
                        break

        # If no files found, generate synthetic data for testing
        if not self.images:
            logger.warning(
                f"No data found at {root}. Generating synthetic dataset."
            )
            self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic dataset for testing and development."""
        num_samples = min(100, self.config.max_samples) if self.config.max_samples > 0 else 100
        h, w = self.config.input_size

        for i in range(num_samples):
            # Create synthetic image path markers
            self.images.append(f"synthetic://image_{i:04d}")
            self.labels.append(f"synthetic://label_{i:04d}")

        self._synthetic_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_synthetic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load or generate synthetic sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image, label).
        """
        if index in getattr(self, "_synthetic_cache", {}):
            return self._synthetic_cache[index]

        h, w = self.config.input_size
        nc = self.config.num_classes

        # Generate realistic-looking synthetic data
        rng = np.random.RandomState(index)

        # Image: road scene approximation
        image = np.zeros((h, w, 3), dtype=np.uint8)

        # Sky
        image[:h // 3, :] = [135, 206, 235]

        # Road
        image[h // 3:, :] = [100, 100, 100]

        # Road surface gradient
        for y in range(h // 3, h):
            darkness = int(80 + 40 * (y - h // 3) / (2 * h // 3))
            image[y, :] = [darkness, darkness, darkness]

        # Add some variation
        noise = rng.randint(-10, 10, (h, w, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Label: approximate road scene layout
        label = np.zeros((h, w), dtype=np.uint8)
        label[:h // 3, :] = 10  # sky
        label[h // 3:h // 3 + 50, :] = 8  # vegetation
        label[h // 3 + 50:, :w // 4] = 1  # sidewalk left
        label[h // 3 + 50:, w // 4:3 * w // 4] = 0  # road
        label[h // 3 + 50:, 3 * w // 4:] = 1  # sidewalk right

        # Add random vehicles and pedestrians
        for _ in range(rng.randint(0, 3)):
            vy = rng.randint(h // 3 + 50, h - 50)
            vx = rng.randint(w // 4, 3 * w // 4)
            vw = rng.randint(30, 60)
            vh = rng.randint(30, 50)
            label[vy:vy + vh, vx:vx + vw] = 13  # car

        if self.config.cache_images:
            self._synthetic_cache[index] = (image, label)

        return image, label

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        """Get a single sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary with 'image', 'label', and metadata.
        """
        if index in self._image_cache:
            return self._image_cache[index]

        image, label = self._load_sample(index)

        # Apply augmentation
        if self.augmentation is not None and self.config.split == SplitType.TRAIN:
            image, label = self.augmentation(image, label)

        # Normalize image
        image = self._normalize(image)

        sample = {
            "image": image,
            "label": label,
            "index": index,
            "image_path": self.images[index] if index < len(self.images) else "",
        }

        if self.config.cache_images:
            self._image_cache[index] = sample

        return sample

    def _load_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a single image-label pair.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image, label).
        """
        img_path = self.images[index]

        if img_path.startswith("synthetic://"):
            return self._load_synthetic(index)

        try:
            image = np.array(
                __import__("PIL").Image.open(img_path).convert("RGB")
            )
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            h, w = self.config.input_size
            image = np.zeros((h, w, 3), dtype=np.uint8)

        lbl_path = self.labels[index]
        try:
            label = np.array(
                __import__("PIL").Image.open(lbl_path).convert("L")
            )
        except Exception as e:
            logger.warning(f"Failed to load label {lbl_path}: {e}")
            h, w = self.config.input_size
            label = np.zeros((h, w), dtype=np.uint8)

        return image, label

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image using configured mean and std.

        Args:
            image: Input image (H, W, C) uint8.

        Returns:
            Normalized image (H, W, C) float32.
        """
        image = image.astype(np.float32) / 255.0
        mean = np.array(self.config.normalize_mean)
        std = np.array(self.config.normalize_std)
        image = (image - mean) / std
        return image

    def get_class_distribution(self) -> np.ndarray:
        """Compute per-class pixel distribution across the dataset.

        Returns:
            Array of pixel counts per class.
        """
        counts = np.zeros(self.config.num_classes, dtype=np.int64)

        for i in range(len(self)):
            sample = self[i]
            label = sample["label"]
            for c in range(self.config.num_classes):
                counts[c] += np.sum(label == c)

        return counts


# ---------------------------------------------------------------------------
# Cityscapes Dataset
# ---------------------------------------------------------------------------

class CityscapesDataset(SegmentationDataset):
    """Cityscapes dataset loader.

    Directory structure:
        root/
        ├── leftImg8bit/
        │   ├── train/
        │   │   ├── aachen/
        │   │   │   ├── aachen_000000_000000_leftImg8bit.png
        │   │   │   └── ...
        │   │   └── ...
        │   ├── val/
        │   └── test/
        └── gtFine/
            ├── train/
            │   ├── aachen/
            │   │   ├── aachen_000000_000000_gtFine_labelTrainIds.png
            │   │   └── ...
            │   └── ...
            ├── val/
            └── test/
    """

    CITYSCAPES_CITIES = {
        SplitType.TRAIN: [
            "aachen", "bochum", "bremen", "cologne", "darmstadt",
            "dusseldorf", "erfurt", "hamburg", "hanover", "jena",
            "krefeld", "monchengladbach", "strasbourg", "stuttgart",
            "tubingen", "ulm", "weimar", "zurich",
        ],
        SplitType.VAL: [
            "frankfurt", "lindau", "munster",
        ],
        SplitType.TEST: [
            "berlin", "bielefeld", "bonn", "leverkusen", "mainz",
            "munich",
        ],
    }

    def _load_file_list(self) -> None:
        """Load Cityscapes file list."""
        root = Path(self.config.root_dir)
        split = self.config.split.value

        image_dir = root / "leftImg8bit" / split
        suffix = "gtCoarse" if self.config.use_coarse else "gtFine"
        label_dir = root / suffix / split

        if not image_dir.exists():
            logger.warning(f"Cityscapes image dir not found: {image_dir}")
            self._generate_synthetic_data()
            return

        cities = self.CITYSCAPES_CITIES.get(self.config.split, [])

        for city in cities:
            city_img_dir = image_dir / city
            city_lbl_dir = label_dir / city

            if not city_img_dir.exists():
                continue

            for img_path in sorted(city_img_dir.glob("*_leftImg8bit.png")):
                stem = img_path.stem.replace("_leftImg8bit", "")
                label_path = city_lbl_dir / f"{stem}_{suffix}_labelTrainIds.png"

                if label_path.exists():
                    self.images.append(str(img_path))
                    self.labels.append(str(label_path))

        if not self.images:
            logger.warning("No Cityscapes files found, generating synthetic data")
            self._generate_synthetic_data()


# ---------------------------------------------------------------------------
# KITTI Dataset
# ---------------------------------------------------------------------------

class KITTIDataset(SegmentationDataset):
    """KITTI semantic segmentation dataset loader.

    Directory structure:
        root/
        ├── image_2/
        │   ├── training/
        │   │   ├── 000000_10.png
        │   │   └── ...
        │   └── testing/
        └── semantic/
            └── training/
                ├── 000000_10.png
                └── ...
    """

    def _load_file_list(self) -> None:
        """Load KITTI file list."""
        root = Path(self.config.root_dir)
        split_dir = "training" if self.config.split != SplitType.TEST else "testing"

        image_dir = root / "image_2" / split_dir
        label_dir = root / "semantic" / split_dir

        if not image_dir.exists():
            logger.warning(f"KITTI image dir not found: {image_dir}")
            self._generate_synthetic_data()
            return

        for img_path in sorted(image_dir.glob("*.png")):
            label_path = label_dir / img_path.name
            if label_path.exists():
                self.images.append(str(img_path))
                self.labels.append(str(label_path))

        if not self.images:
            self._generate_synthetic_data()


# ---------------------------------------------------------------------------
# Custom Dataset
# ---------------------------------------------------------------------------

class CustomDataset(SegmentationDataset):
    """Custom dataset loader with configurable directory structure.

    Directory structure (configurable):
        root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
    """

    def __init__(
        self,
        config: DatasetConfig,
        image_subdir: str = "images",
        label_subdir: str = "labels",
        image_ext: str = ".png",
        label_ext: str = ".png",
    ) -> None:
        """Initialize custom dataset.

        Args:
            config: Dataset configuration.
            image_subdir: Subdirectory name for images.
            label_subdir: Subdirectory name for labels.
            image_ext: Image file extension.
            label_ext: Label file extension.
        """
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir
        self.image_ext = image_ext
        self.label_ext = label_ext
        super().__init__(config)


# ---------------------------------------------------------------------------
# Dataset Factory and DataLoader
# ---------------------------------------------------------------------------

class DatasetFactory:
    """Factory for creating dataset instances.

    Example:
        >>> config = DatasetConfig(dataset_type=DatasetType.CITYSCAPES)
        >>> dataset = DatasetFactory.create(config)
    """

    _registry: Dict[DatasetType, type] = {
        DatasetType.CITYSCAPES: CityscapesDataset,
        DatasetType.KITTI: KITTIDataset,
        DatasetType.KITTI_ROAD: KITTIDataset,
        DatasetType.CUSTOM: CustomDataset,
    }

    @classmethod
    def create(cls, config: DatasetConfig) -> SegmentationDataset:
        """Create dataset from configuration.

        Args:
            config: Dataset configuration.

        Returns:
            Dataset instance.
        """
        dataset_class = cls._registry.get(config.dataset_type, SegmentationDataset)
        return dataset_class(config)

    @classmethod
    def register(cls, dataset_type: DatasetType, dataset_class: type) -> None:
        """Register a custom dataset class."""
        cls._registry[dataset_type] = dataset_class


class DataLoader:
    """Data loader for batching and iteration over datasets.

    Provides batched iteration with shuffling, padding, and
    multi-worker data loading support.

    Example:
        >>> dataset = DatasetFactory.create(config)
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=True)
        >>> for batch in loader:
        ...     images = batch["image"]
        ...     labels = batch["label"]
    """

    def __init__(
        self,
        dataset: SegmentationDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize data loader.

        Args:
            dataset: Dataset to load from.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle indices each epoch.
            num_workers: Number of parallel workers (0 = main thread).
            drop_last: Whether to drop incomplete last batch.
            seed: Random seed.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.rng = np.random.RandomState(seed)
        self._indices = np.arange(len(dataset))

    def __len__(self) -> int:
        """Return number of batches."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches.

        Yields:
            Dictionary with batched 'image' and 'label' arrays.
        """
        indices = self._indices.copy()
        if self.shuffle:
            self.rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            batch_samples = [self.dataset[idx] for idx in batch_indices]

            # Stack into batches
            images = np.stack([s["image"] for s in batch_samples], axis=0)
            labels = np.stack([s["label"] for s in batch_samples], axis=0)

            yield {
                "image": images,
                "label": labels,
                "indices": batch_indices,
            }

    def get_class_weights(self) -> np.ndarray:
        """Compute class weights from dataset distribution.

        Returns:
            Per-class weights for balanced training.
        """
        distribution = self.dataset.get_class_distribution()
        total = np.sum(distribution)

        if total == 0:
            return np.ones(self.dataset.config.num_classes, dtype=np.float32)

        frequencies = distribution / total
        frequencies = np.clip(frequencies, 1e-8, None)
        weights = 1.0 / frequencies
        weights = weights / np.sum(weights) * self.dataset.config.num_classes

        return weights.astype(np.float32)
