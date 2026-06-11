"""
TensorBoard Logger Module for Training Visualization.

Provides a unified logging interface for TensorBoard:
- Scalar logging (loss, accuracy, learning rate)
- Image logging (predictions, augmentations, attention maps)
- Histogram logging (weights, gradients, activations)
- Text logging (configuration, metrics summaries)
- Hyperparameter logging
- Custom plot generation
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class LoggerConfig:
    """Configuration for the TensorBoard logger."""
    log_dir: str = "./logs"
    enabled: bool = True
    flush_secs: int = 30
    max_queue: int = 100
    write_scalar_every: int = 1
    write_image_every: int = 100
    write_histogram_every: int = 500
    write_text_every: int = 100
    image_max_outputs: int = 4
    histogram_bins: int = 50


class TensorBoardLogger:
    """
    TensorBoard logger with rate-limited writing.

    Provides a high-level interface for logging training metrics
    to TensorBoard, with configurable write frequencies to
    control I/O overhead.

    Example:
        >>> logger = TensorBoardLogger(log_dir="./logs")
        >>> logger.log_scalar("train/loss", 0.5, step=100)
        >>> logger.log_scalars(step=100, metrics={"train/loss": 0.5, "train/acc": 0.9})
        >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        enabled: bool = True,
        config: Optional[LoggerConfig] = None,
    ) -> None:
        """
        Initialize the TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs.
            enabled: Whether logging is enabled.
            config: Logger configuration.
        """
        self.config = config or LoggerConfig(log_dir=log_dir, enabled=enabled)
        self.enabled = enabled and self.config.enabled
        self._writer = None
        self._step = 0
        self._logged_scalars: Dict[str, List[Tuple[int, float]]] = {}

        if self.enabled:
            self._setup_writer()

    def _setup_writer(self) -> None:
        """Initialize the TensorBoard SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(
                log_dir=self.config.log_dir,
                flush_secs=self.config.flush_secs,
                max_queue=self.config.max_queue,
            )
            print(f"[TensorBoard] Logging to {self.config.log_dir}")
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
                self._writer = SummaryWriter(logdir=self.config.log_dir)
                print(f"[TensorBoardX] Logging to {self.config.log_dir}")
            except ImportError:
                print("[TensorBoard] Not available. Logging disabled.")
                self.enabled = False
                self._writer = None

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a scalar value.

        Args:
            tag: Tag name for the scalar (e.g., "train/loss").
            value: Scalar value to log.
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        if step % self.config.write_scalar_every != 0:
            return

        self._writer.add_scalar(tag, value, step)

        # Track for summaries
        if tag not in self._logged_scalars:
            self._logged_scalars[tag] = []
        self._logged_scalars[tag].append((step, value))

    def log_scalars(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log multiple scalar values.

        Args:
            metrics: Dictionary of tag -> value pairs.
            step: Global step number.
        """
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Log an image.

        Args:
            tag: Tag name for the image.
            image: Image array (H, W, C) or (H, W), values in [0, 255] uint8 or [0, 1] float.
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        if step % self.config.write_image_every != 0:
            return

        try:
            import torch
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = image[..., np.newaxis]
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                tensor = torch.from_numpy(image).permute(2, 0, 1)
            else:
                tensor = image
            self._writer.add_image(tag, tensor, step)
        except ImportError:
            pass

    def log_images(
        self,
        tag: str,
        images: np.ndarray,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a batch of images.

        Args:
            tag: Tag name.
            images: Batch of images (N, H, W, C).
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        max_outputs = self.config.image_max_outputs
        for i in range(min(len(images), max_outputs)):
            self.log_image(f"{tag}/{i}", images[i], step)

    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None,
        bins: Optional[int] = None,
    ) -> None:
        """
        Log a histogram of values.

        Args:
            tag: Tag name.
            values: Array of values to histogram.
            step: Global step number.
            bins: Number of histogram bins.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        if step % self.config.write_histogram_every != 0:
            return

        bins = bins or self.config.histogram_bins

        try:
            import torch
            if isinstance(values, np.ndarray):
                tensor = torch.from_numpy(values)
            else:
                tensor = values
            self._writer.add_histogram(tag, tensor, step, bins=bins)
        except ImportError:
            pass

    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None,
    ) -> None:
        """
        Log text data.

        Args:
            tag: Tag name.
            text: Text string to log.
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        if step % self.config.write_text_every != 0:
            return

        self._writer.add_text(tag, text, step)

    def log_hyperparams(
        self,
        hparams: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log hyperparameters and optionally their associated metrics.

        Args:
            hparams: Dictionary of hyperparameter names to values.
            metrics: Optional dictionary of metric names to values.
        """
        if not self.enabled or self._writer is None:
            return

        # Convert all values to types supported by TensorBoard
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (bool, int, float, str)):
                clean_hparams[k] = v
            elif isinstance(v, (list, tuple)):
                clean_hparams[k] = str(v)
            else:
                clean_hparams[k] = str(v)

        if metrics:
            try:
                self._writer.add_hparams(clean_hparams, metrics)
            except Exception:
                pass
        else:
            self.log_text("hyperparams", str(clean_hparams))

    def log_model_graph(
        self,
        model: Any,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> None:
        """
        Log the model computation graph.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape for tracing.
        """
        if not self.enabled or self._writer is None:
            return

        try:
            import torch
            dummy_input = torch.randn(*input_shape)
            self._writer.add_graph(model, dummy_input)
        except Exception as e:
            print(f"[TensorBoard] Failed to log model graph: {e}")

    def log_learning_rate(
        self,
        lr: float,
        step: Optional[int] = None,
    ) -> None:
        """Log current learning rate."""
        self.log_scalar("train/learning_rate", lr, step)

    def log_gradients(
        self,
        model: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Log gradient histograms for all model parameters.

        Args:
            model: PyTorch model.
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(
                    f"gradients/{name}",
                    param.grad.detach().cpu().numpy(),
                    step,
                )

    def log_weights(
        self,
        model: Any,
        step: Optional[int] = None,
    ) -> None:
        """
        Log weight histograms for all model parameters.

        Args:
            model: PyTorch model.
            step: Global step number.
        """
        if not self.enabled or self._writer is None:
            return

        step = step or self._step
        for name, param in model.named_parameters():
            self.log_histogram(
                f"weights/{name}",
                param.detach().cpu().numpy(),
                step,
            )

    def get_scalar_summary(self, tag: str, last_n: int = 100) -> Dict[str, float]:
        """
        Get summary statistics for a logged scalar.

        Args:
            tag: Scalar tag name.
            last_n: Number of recent values to summarize.

        Returns:
            Dictionary with mean, std, min, max.
        """
        if tag not in self._logged_scalars:
            return {}

        recent = self._logged_scalars[tag][-last_n:]
        values = [v for _, v in recent]

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }

    def step(self) -> None:
        """Increment the global step counter."""
        self._step += 1

    def flush(self) -> None:
        """Flush all pending events to disk."""
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the logger and flush remaining events."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None
