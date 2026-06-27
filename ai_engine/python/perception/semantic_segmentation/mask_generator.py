"""
Mask Generation Module for Autonomous Vehicle Semantic Segmentation.

Generates refined segmentation masks from raw model outputs using
CRF (Conditional Random Field) refinement, morphological operations,
and confidence-based filtering.

Processing Pipeline:
    Raw Logits ──▶ Softmax Probabilities ──▶ Argmax Mask
                                              │
                                    ┌─────────▼─────────┐
                                    │  Confidence Filter │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  CRF Refinement    │
                                    │  (Pairwise +       │
                                    │   Unary Potentials)│
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  Morphological Ops │
                                    │  (Close → Open →   │
                                    │   Fill Holes)      │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────▼─────────┐
                                    │  Small Region      │
                                    │  Removal           │
                                    └─────────┬─────────┘
                                              │
                                        Final Mask

Author: Autonomous Vehicle Control System Team
License: MIT
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MaskGeneratorConfig:
    """Configuration for mask generation pipeline.

    Attributes:
        confidence_threshold: Minimum confidence for pixel assignment.
        use_crf: Whether to apply CRF refinement.
        crf_iterations: Number of CRF inference iterations.
        crf bilateral_weight: Weight for bilateral kernel in CRF.
        crf_spatial_sigma: Spatial sigma for bilateral kernel.
        crf_color_sigma: Color sigma for bilateral kernel.
        crf_compatibility_weight: Compatibility weight for CRF.
        use_morphology: Whether to apply morphological operations.
        morph_kernel_size: Kernel size for morphological operations.
        morph_close_iterations: Number of closing iterations.
        morph_open_iterations: Number of opening iterations.
        fill_holes: Whether to fill holes in masks.
        remove_small_regions: Whether to remove small regions.
        min_region_area: Minimum area in pixels for a region to keep.
        smooth_boundaries: Whether to apply boundary smoothing.
        boundary_smooth_sigma: Sigma for Gaussian boundary smoothing.
    """

    confidence_threshold: float = 0.5
    use_crf: bool = True
    crf_iterations: int = 5
    crf_bilateral_weight: float = 5.0
    crf_spatial_sigma: float = 3.0
    crf_color_sigma: float = 10.0
    crf_compatibility_weight: float = 3.0
    use_morphology: bool = True
    morph_kernel_size: int = 5
    morph_close_iterations: int = 1
    morph_open_iterations: int = 1
    fill_holes: bool = True
    remove_small_regions: bool = True
    min_region_area: int = 256
    smooth_boundaries: bool = False
    boundary_smooth_sigma: float = 1.0


# ---------------------------------------------------------------------------
# CRF Refinement
# ---------------------------------------------------------------------------

class CRFRefiner:
    """Dense Conditional Random Field for mask refinement.

    Implements a simplified dense CRF with bilateral and spatial
    kernels for refining segmentation masks using image appearance.

    Energy function:
        E(x) = Σ_i ψ_u(x_i) + Σ_{i<j} ψ_p(x_i, x_j)

    Where:
        ψ_u(x_i) = -log P(x_i)           (unary from model)
        ψ_p(x_i, x_j) = w * k(f_i, f_j)  (pairwise from image)

    Bilateral kernel:
        k_bilateral = exp(-|p_i - p_j|²/(2σ_α²) - |I_i - I_j|²/(2σ_β²))

    Spatial kernel:
        k_spatial = exp(-|p_i - p_j|²/(2σ_γ²))

    Attributes:
        config: CRF configuration parameters.
    """

    def __init__(self, config: Optional[MaskGeneratorConfig] = None) -> None:
        """Initialize CRF refiner.

        Args:
            config: Configuration for CRF parameters.
        """
        self.config = config or MaskGeneratorConfig()

    def _compute_bilateral_kernel(
        self,
        image: np.ndarray,
        positions: np.ndarray,
        spatial_sigma: float,
        color_sigma: float,
    ) -> np.ndarray:
        """Compute bilateral affinity between sampled pixel pairs.

        Args:
            image: Input image of shape (H, W, 3).
            positions: Pixel positions (N, 2).
            spatial_sigma: Spatial distance sigma.
            color_sigma: Color distance sigma.

        Returns:
            Affinity matrix of shape (N, N).
        """
        n = len(positions)
        if n > 500:  # Limit for memory
            # Subsample for efficiency
            indices = np.random.choice(n, 500, replace=False)
            positions = positions[indices]
            n = 500

        # Get colors at positions
        colors = image[positions[:, 0], positions[:, 1]].astype(np.float64)  # (N, 3)

        # Compute spatial distances
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
        spatial_dist = np.sum(pos_diff ** 2, axis=2)  # (N, N)

        # Compute color distances
        color_diff = colors[:, np.newaxis, :] - colors[np.newaxis, :, :]  # (N, N, 3)
        color_dist = np.sum(color_diff ** 2, axis=2)  # (N, N)

        # Bilateral kernel
        spatial_exp = np.exp(-spatial_dist / (2 * spatial_sigma ** 2))
        color_exp = np.exp(-color_dist / (2 * color_sigma ** 2))
        bilateral = spatial_exp * color_exp

        return bilateral

    def _compute_spatial_kernel(
        self,
        positions: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Compute spatial Gaussian kernel.

        Args:
            positions: Pixel positions (N, 2).
            sigma: Spatial sigma.

        Returns:
            Spatial affinity matrix (N, N).
        """
        n = len(positions)
        pos_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        spatial_dist = np.sum(pos_diff ** 2, axis=2)
        return np.exp(-spatial_dist / (2 * sigma ** 2))

    def refine(
        self,
        probabilities: np.ndarray,
        image: np.ndarray,
        num_iterations: Optional[int] = None,
    ) -> np.ndarray:
        """Refine segmentation mask using dense CRF.

        Performs mean-field inference iterations:
            Q_i(x) ∝ exp(-ψ_u(x) - Σ_{j≠i} Σ_{x'} Q_j(x') * ψ_p(x, x'))

        Args:
            probabilities: Class probabilities of shape (C, H, W) or (H, W).
            image: Original image of shape (H, W, 3).
            num_iterations: Override CRF iterations.

        Returns:
            Refined class probabilities.
        """
        if num_iterations is None:
            num_iterations = self.config.crf_iterations

        if probabilities.ndim == 2:
            # Binary case: expand to (2, H, W)
            probs = np.stack([1 - probabilities, probabilities], axis=0)
        else:
            probs = probabilities.copy()

        c, h, w = probs.shape

        # Clip probabilities for numerical stability
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / np.sum(probs, axis=0, keepdims=True)

        # Initialize Q from unary (model predictions)
        Q = probs.copy()

        # Subsample positions for efficiency
        step = max(1, min(h, w) // 50)
        ys, xs = np.mgrid[0:h:step, 0:w:step]
        positions = np.column_stack([ys.ravel(), xs.ravel()]).astype(np.float64)
        n_samples = len(positions)

        # Pre-compute kernels
        bilateral_weight = self.config.crf_bilateral_weight
        spatial_sigma = self.config.crf_spatial_sigma
        color_sigma = self.config.crf_color_sigma

        bilateral_kernel = self._compute_bilateral_kernel(
            image, positions, spatial_sigma, color_sigma
        )
        spatial_kernel = self._compute_spatial_kernel(positions, sigma=spatial_sigma * 3)

        # Mean-field iterations
        compat_weight = self.config.crf_compatibility_weight

        for iteration in range(num_iterations):
            # Sample Q at kernel positions
            Q_sampled = Q[:, positions[:, 0].astype(int), positions[:, 1].astype(int)]  # (C, N)

            # Message passing: sum of Q * kernel for each class
            messages_bilateral = np.zeros_like(Q_sampled)  # (C, N)
            messages_spatial = np.zeros_like(Q_sampled)

            for cls in range(c):
                q_cls = Q_sampled[cls]  # (N,)
                # Bilateral message
                msg_b = bilateral_kernel @ q_cls
                messages_bilateral[cls] = msg_b
                # Spatial message
                msg_s = spatial_kernel @ q_cls
                messages_spatial[cls] = msg_s

            # Compatibility transform (Potts model: penalize different labels)
            pairwise = np.zeros_like(Q_sampled)
            for cls in range(c):
                # Sum of all other classes' messages
                other_sum = np.sum(messages_bilateral, axis=0) - messages_bilateral[cls]
                other_sum_s = np.sum(messages_spatial, axis=0) - messages_spatial[cls]
                pairwise[cls] = -compat_weight * (bilateral_weight * other_sum + other_sum_s)

            # Update Q at sampled positions
            unary_sampled = -np.log(np.clip(Q_sampled, 1e-8, 1.0))  # Negative log-prob
            updated = -unary_sampled + pairwise  # Minimize energy
            updated = np.exp(updated - np.max(updated, axis=0, keepdims=True))
            updated = updated / np.sum(updated, axis=0, keepdims=True)

            # Propagate back to full resolution (nearest neighbor)
            for idx, (py, px) in enumerate(positions.astype(int)):
                py = min(py, h - 1)
                px = min(px, w - 1)
                # Apply update to local region
                y_start = max(0, py - step // 2)
                y_end = min(h, py + step // 2 + 1)
                x_start = max(0, px - step // 2)
                x_end = min(w, px + step // 2 + 1)
                Q[:, y_start:y_end, x_start:x_end] = updated[:, idx:idx+1]

        return Q

    def refine_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        num_classes: int = 2,
    ) -> np.ndarray:
        """Refine a hard segmentation mask using CRF.

        Args:
            mask: Hard segmentation mask (H, W) with class indices.
            image: Original image (H, W, 3).
            num_classes: Number of classes.

        Returns:
            Refined mask with class indices.
        """
        # Convert mask to soft probabilities
        h, w = mask.shape
        probs = np.zeros((num_classes, h, w), dtype=np.float32)
        for c in range(num_classes):
            probs[c] = (mask == c).astype(np.float32) * 0.9 + 0.05

        # Normalize
        probs = probs / np.sum(probs, axis=0, keepdims=True)

        # Refine
        refined_probs = self.refine(probs, image, num_iterations=3)

        # Convert back to hard mask
        refined_mask = np.argmax(refined_probs, axis=0).astype(np.uint8)
        return refined_mask


# ---------------------------------------------------------------------------
# Morphological Operations
# ---------------------------------------------------------------------------

class MorphologicalProcessor:
    """Morphological operations for segmentation mask refinement.

    Provides configurable pipeline of morphological operations:
        1. Close (fill small gaps)
        2. Open (remove small protrusions)
        3. Fill holes
        4. Boundary smoothing
    """

    def __init__(
        self,
        kernel_size: int = 5,
        close_iterations: int = 1,
        open_iterations: int = 1,
        fill_holes: bool = True,
        smooth_boundaries: bool = False,
        smooth_sigma: float = 1.0,
    ) -> None:
        """Initialize morphological processor.

        Args:
            kernel_size: Kernel size for morphological operations.
            close_iterations: Number of closing iterations.
            open_iterations: Number of opening iterations.
            fill_holes: Whether to fill holes in masks.
            smooth_boundaries: Whether to smooth mask boundaries.
            smooth_sigma: Sigma for boundary smoothing.
        """
        self.kernel_size = kernel_size
        self.close_iterations = close_iterations
        self.open_iterations = open_iterations
        self.fill_holes = fill_holes
        self.smooth_boundaries = smooth_boundaries
        self.smooth_sigma = smooth_sigma

        # Pre-compute kernel
        self._kernel = self._create_kernel(kernel_size)

    @staticmethod
    def _create_kernel(size: int) -> np.ndarray:
        """Create morphological kernel (cross shape).

        Args:
            size: Kernel size (must be odd).

        Returns:
            Binary kernel array.
        """
        if size % 2 == 0:
            size += 1
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        kernel[center, :] = 1
        kernel[:, center] = 1
        # Also fill diagonals for larger kernels
        if size >= 5:
            for i in range(size):
                for j in range(size):
                    if abs(i - center) + abs(j - center) <= center:
                        kernel[i, j] = 1
        return kernel

    def dilate(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Apply morphological dilation.

        Args:
            mask: Binary mask.
            iterations: Number of dilation iterations.

        Returns:
            Dilated mask.
        """
        result = mask.copy()
        h, w = result.shape
        kh, kw = self._kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        for _ in range(iterations):
            padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
            new_result = np.zeros_like(result)
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    new_result[i, j] = np.max(region * self._kernel) if np.any(self._kernel) else result[i, j]
            result = new_result

        return result

    def erode(self, mask: np.ndarray, iterations: int = 1) -> np.ndarray:
        """Apply morphological erosion.

        Args:
            mask: Binary mask.
            iterations: Number of erosion iterations.

        Returns:
            Eroded mask.
        """
        result = mask.copy()
        h, w = result.shape
        kh, kw = self._kernel.shape
        pad_h, pad_w = kh // 2, kw // 2

        for _ in range(iterations):
            padded = np.pad(result, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
            new_result = np.zeros_like(result)
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kh, j:j+kw]
                    # Erosion: all kernel positions must be 1
                    kernel_pixels = region[self._kernel > 0]
                    new_result[i, j] = 1 if np.all(kernel_pixels > 0) else 0
            result = new_result

        return result

    def close(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological closing (dilation then erosion).

        Fills small gaps and holes in the mask.

        Args:
            mask: Binary mask.

        Returns:
            Closed mask.
        """
        result = mask.copy()
        for _ in range(self.close_iterations):
            result = self.dilate(result, iterations=1)
            result = self.erode(result, iterations=1)
        return result

    def open(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening (erosion then dilation).

        Removes small protrusions and noise.

        Args:
            mask: Binary mask.

        Returns:
            Opened mask.
        """
        result = mask.copy()
        for _ in range(self.open_iterations):
            result = self.erode(result, iterations=1)
            result = self.dilate(result, iterations=1)
        return result

    def fill_holes_op(self, mask: np.ndarray) -> np.ndarray:
        """Fill holes in binary mask using flood fill from borders.

        Args:
            mask: Binary mask.

        Returns:
            Mask with holes filled.
        """
        if not self.fill_holes:
            return mask

        result = mask.copy()
        h, w = result.shape

        # Create inverse mask
        inv_mask = (result == 0).astype(np.uint8)

        # Flood fill from all border pixels
        visited = np.zeros_like(inv_mask, dtype=bool)
        queue: List[Tuple[int, int]] = []

        # Add border pixels to queue
        for i in range(h):
            for j in [0, w - 1]:
                if inv_mask[i, j] > 0 and not visited[i, j]:
                    visited[i, j] = True
                    queue.append((i, j))
        for j in range(w):
            for i in [0, h - 1]:
                if inv_mask[i, j] > 0 and not visited[i, j]:
                    visited[i, j] = True
                    queue.append((i, j))

        # BFS flood fill
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and inv_mask[ny, nx] > 0:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        # Holes are unvisited background pixels
        holes = (inv_mask > 0) & (~visited)
        result[holes] = 1

        return result

    def process(self, mask: np.ndarray) -> np.ndarray:
        """Apply full morphological processing pipeline.

        Args:
            mask: Binary mask.

        Returns:
            Processed mask.
        """
        result = mask.copy()

        # Close (fill gaps)
        result = self.close(result)

        # Open (remove noise)
        result = self.open(result)

        # Fill holes
        result = self.fill_holes_op(result)

        return result


# ---------------------------------------------------------------------------
# Small Region Removal
# ---------------------------------------------------------------------------

class SmallRegionRemover:
    """Removes small disconnected regions from segmentation masks.

    Uses connected component analysis to identify and remove regions
    smaller than a threshold, reducing false positive noise.
    """

    def __init__(self, min_area: int = 256) -> None:
        """Initialize small region remover.

        Args:
            min_area: Minimum area in pixels for a region to keep.
        """
        self.min_area = min_area

    def remove(self, mask: np.ndarray) -> np.ndarray:
        """Remove small regions from binary mask.

        Args:
            mask: Binary mask (H, W).

        Returns:
            Cleaned mask.
        """
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        result = np.zeros_like(mask)

        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0 and not visited[i, j]:
                    # BFS to find connected component
                    component_pixels: List[Tuple[int, int]] = []
                    queue = [(i, j)]
                    visited[i, j] = True

                    while queue:
                        cy, cx = queue.pop(0)
                        component_pixels.append((cy, cx))
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx] > 0:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                    # Keep only large components
                    if len(component_pixels) >= self.min_area:
                        for y, x in component_pixels:
                            result[y, x] = 1

        return result

    def remove_per_class(
        self, mask: np.ndarray, num_classes: int
    ) -> np.ndarray:
        """Remove small regions for each class independently.

        Args:
            mask: Segmentation mask with class indices (H, W).
            num_classes: Number of classes.

        Returns:
            Cleaned mask.
        """
        result = np.zeros_like(mask)

        for cls in range(num_classes):
            class_mask = (mask == cls).astype(np.uint8)
            cleaned = self.remove(class_mask)
            result[cleaned > 0] = cls

        return result


# ---------------------------------------------------------------------------
# Main Mask Generator
# ---------------------------------------------------------------------------

class MaskGenerator:
    """Complete mask generation pipeline from model output to final mask.

    Orchestrates the full refinement pipeline:
        1. Convert logits to probabilities (softmax)
        2. Apply confidence filtering
        3. Refine with CRF (optional)
        4. Apply morphological operations (optional)
        5. Remove small regions (optional)
        6. Smooth boundaries (optional)

    Example:
        >>> config = MaskGeneratorConfig(use_crf=True, use_morphology=True)
        >>> generator = MaskGenerator(config)
        >>> image = np.random.randint(0, 255, (512, 1024, 3), dtype=np.uint8)
        >>> logits = np.random.randn(19, 512, 1024).astype(np.float32)
        >>> mask = generator.generate(logits, image)
        >>> probs = generator.generate_probabilities(logits)
    """

    def __init__(self, config: Optional[MaskGeneratorConfig] = None) -> None:
        """Initialize mask generator.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or MaskGeneratorConfig()
        self._crf = CRFRefiner(self.config) if self.config.use_crf else None
        self._morph = MorphologicalProcessor(
            kernel_size=self.config.morph_kernel_size,
            close_iterations=self.config.morph_close_iterations,
            open_iterations=self.config.morph_open_iterations,
            fill_holes=self.config.fill_holes,
            smooth_boundaries=self.config.smooth_boundaries,
            smooth_sigma=self.config.boundary_smooth_sigma,
        ) if self.config.use_morphology else None
        self._small_region_remover = SmallRegionRemover(
            self.config.min_region_area
        ) if self.config.remove_small_regions else None

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Apply numerically stable softmax.

        Args:
            logits: Raw logits of shape (C, H, W) or (N, C, H, W).

        Returns:
            Class probabilities.
        """
        if logits.ndim == 3:
            shifted = logits - np.max(logits, axis=0, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
        elif logits.ndim == 4:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_vals = np.exp(shifted)
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return logits

    def generate_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to class probabilities.

        Args:
            logits: Raw model output of shape (C, H, W) or (N, C, H, W).

        Returns:
            Class probabilities with same shape.
        """
        return self.softmax(logits)

    def generate_mask(
        self, logits: np.ndarray, image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate segmentation mask from model logits.

        Args:
            logits: Raw model output of shape (C, H, W).
            image: Optional original image for CRF refinement.

        Returns:
            Segmentation mask with class indices (H, W).
        """
        # Convert to probabilities
        probs = self.generate_probabilities(logits)

        # Confidence filtering
        max_probs = np.max(probs, axis=0)
        mask = np.argmax(probs, axis=0).astype(np.uint8)

        # Apply confidence threshold
        if self.config.confidence_threshold > 0:
            low_confidence = max_probs < self.config.confidence_threshold
            mask[low_confidence] = 0  # Assign to background

        # CRF refinement
        if self._crf is not None and image is not None:
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            probs_refined = self._crf.refine(probs, image)
            mask = np.argmax(probs_refined, axis=0).astype(np.uint8)

        # Morphological operations (per-class)
        if self._morph is not None:
            num_classes = logits.shape[0] if logits.ndim == 3 else logits.shape[1]
            # Apply to each class independently
            for cls in range(1, num_classes):  # Skip background
                class_mask = (mask == cls).astype(np.uint8)
                if np.sum(class_mask) > 0:
                    class_mask = self._morph.process(class_mask)
                    mask[class_mask > 0] = cls

        # Small region removal
        if self._small_region_remover is not None:
            num_classes = logits.shape[0] if logits.ndim == 3 else logits.shape[1]
            mask = self._small_region_remover.remove_per_class(mask, num_classes)

        return mask

    def generate_batch(
        self,
        logits_batch: np.ndarray,
        images: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate masks for a batch of model outputs.

        Args:
            logits_batch: Batch of logits (N, C, H, W).
            images: Optional batch of images (N, H, W, 3).

        Returns:
            Batch of segmentation masks (N, H, W).
        """
        n = logits_batch.shape[0]
        masks = []

        for i in range(n):
            logits = logits_batch[i]
            image = images[i] if images is not None else None
            mask = self.generate_mask(logits, image)
            masks.append(mask)

        return np.stack(masks, axis=0)

    def generate_probability_map(
        self,
        logits: np.ndarray,
        target_class: int,
    ) -> np.ndarray:
        """Generate probability map for a specific class.

        Args:
            logits: Model output logits (C, H, W).
            target_class: Target class index.

        Returns:
            Probability map for the target class (H, W).
        """
        probs = self.generate_probabilities(logits)
        if target_class < probs.shape[0]:
            return probs[target_class]
        return np.zeros(logits.shape[1:], dtype=np.float32)

    def generate_confidence_map(
        self, logits: np.ndarray
    ) -> np.ndarray:
        """Generate confidence map (max class probability per pixel).

        Args:
            logits: Model output logits (C, H, W).

        Returns:
            Confidence map (H, W) with values in [0, 1].
        """
        probs = self.generate_probabilities(logits)
        return np.max(probs, axis=0)

    def generate_uncertainty_map(
        self, logits: np.ndarray, method: str = "entropy"
    ) -> np.ndarray:
        """Generate uncertainty map from model output.

        Args:
            logits: Model output logits (C, H, W).
            method: Uncertainty estimation method ('entropy' or 'margin').

        Returns:
            Uncertainty map (H, W) with values in [0, 1].
        """
        probs = self.generate_probabilities(logits)

        if method == "entropy":
            # Normalized entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)
            max_entropy = np.log(probs.shape[0])
            uncertainty = entropy / max_entropy
        elif method == "margin":
            # Margin between top-2 predictions
            sorted_probs = np.sort(probs, axis=0)
            margin = sorted_probs[-1] - sorted_probs[-2]
            uncertainty = 1.0 - margin
        else:
            uncertainty = 1.0 - np.max(probs, axis=0)

        return uncertainty.astype(np.float32)
