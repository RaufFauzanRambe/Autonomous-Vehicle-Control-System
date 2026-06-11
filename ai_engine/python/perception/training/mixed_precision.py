"""
Mixed Precision Training Module.

Provides automatic mixed precision (AMP) training utilities:
- Gradient scaling for FP16 training
- Loss scaling policies (dynamic, static)
- Precision policy management
- Numerical stability monitoring
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class AMPConfig:
    """Configuration for mixed precision training."""
    enabled: bool = True
    dtype: str = "float16"  # float16, bfloat16
    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    max_scale: float = 2.0 ** 24
    min_scale: float = 1.0
    enabled_for_optimizer: bool = True
    enabled_for_loss: bool = True
    skip_inf_nan: bool = True


class GradientScaler:
    """
    Custom gradient scaler for mixed precision training.

    Manages loss scaling to prevent gradient underflow in FP16
    training. Implements dynamic scaling with growth and backoff.
    """

    def __init__(self, config: Optional[AMPConfig] = None) -> None:
        self.config = config or AMPConfig()
        self._scale = self.config.init_scale
        self._growth_tracker = 0
        self._found_inf = False
        self._step_count = 0

        # Statistics
        self._scale_history: List[float] = []
        self._inf_count = 0
        self._total_steps = 0

    @property
    def scale(self) -> float:
        """Current scale factor."""
        return self._scale

    @property
    def found_inf(self) -> bool:
        """Whether inf/nan was found in the last step."""
        return self._found_inf

    def scale_loss(self, loss: Any) -> Any:
        """
        Scale the loss value.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss tensor.
        """
        if not self.config.enabled:
            return loss

        return loss * self._scale

    def unscale_gradients(self, optimizer: Any) -> None:
        """
        Unscale gradients by dividing by the current scale.

        Also checks for inf/nan in gradients.

        Args:
            optimizer: The optimizer with gradients to unscale.
        """
        if not self.config.enabled:
            return

        self._found_inf = False

        try:
            import torch
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        grad_data = param.grad.data
                        if torch.isinf(grad_data).any() or torch.isnan(grad_data).any():
                            self._found_inf = True
                        else:
                            param.grad.data = grad_data / self._scale
        except ImportError:
            pass

    def update(self) -> None:
        """
        Update the scale factor based on gradient health.

        If no inf/nan was found, grow the scale.
        If inf/nan was found, backoff the scale.
        """
        self._total_steps += 1
        self._scale_history.append(self._scale)

        if self._found_inf:
            # Backoff: reduce scale
            self._scale = max(self._scale * self.config.backoff_factor, self.config.min_scale)
            self._growth_tracker = 0
            self._inf_count += 1
        else:
            # Growth: increase scale after stable interval
            self._growth_tracker += 1
            if self._growth_tracker >= self.config.growth_interval:
                self._scale = min(self._scale * self.config.growth_factor, self.config.max_scale)
                self._growth_tracker = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get scaler statistics."""
        return {
            "current_scale": self._scale,
            "inf_count": self._inf_count,
            "total_steps": self._total_steps,
            "inf_rate": self._inf_count / max(self._total_steps, 1),
        }


class PrecisionPolicy:
    """
    Manages precision policies for different parts of the model.

    Allows fine-grained control over which operations use FP16
    vs FP32, enabling safe mixed precision for numerical-sensitive
    operations like softmax, layer norm, etc.
    """

    # Operations that should always run in FP32
    FP32_OPS = {
        "softmax", "log_softmax", "layer_norm", "batch_norm",
        "cross_entropy", "nll_loss", "mse_loss",
        "sin", "cos", "exp", "log", "pow",
    }

    # Operations safe for FP16
    FP16_OPS = {
        "conv2d", "linear", "matmul", "bmm",
        "relu", "silu", "gelu", "tanh",
        "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
    }

    def __init__(
        self,
        default_precision: str = "mixed",
        custom_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.default_precision = default_precision
        self.custom_overrides = custom_overrides or {}
        self._op_precision: Dict[str, str] = {}

        # Build precision map
        for op in self.FP32_OPS:
            self._op_precision[op] = "fp32"
        for op in self.FP16_OPS:
            self._op_precision[op] = "fp16"

        # Apply overrides
        self._op_precision.update(self.custom_overrides)

    def get_precision(self, op_name: str) -> str:
        """Get the precision for an operation."""
        if op_name in self._op_precision:
            return self._op_precision[op_name]
        return self.default_precision

    def should_cast_to_fp16(self, op_name: str) -> bool:
        """Check if an operation should run in FP16."""
        precision = self.get_precision(op_name)
        return precision == "fp16" or (precision == "mixed" and op_name in self.FP16_OPS)


def create_grad_scaler(dtype: str = "float16") -> Optional[Any]:
    """
    Create a gradient scaler for mixed precision training.

    Args:
        dtype: Target precision (float16, bfloat16).

    Returns:
        Gradient scaler instance, or None if AMP not available.
    """
    if dtype == "bfloat16":
        # BF16 doesn't need loss scaling
        return None

    try:
        import torch.cuda.amp
        return torch.cuda.amp.GradScaler(
            init_scale=2.0 ** 16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
    except ImportError:
        # Use custom scaler
        return GradientScaler(AMPConfig(enabled=True, dtype=dtype))


def get_autocast_context(dtype: str = "float16", enabled: bool = True) -> Any:
    """
    Get an autocast context manager for mixed precision.

    Args:
        dtype: Target precision.
        enabled: Whether AMP is enabled.

    Returns:
        Autocast context manager.
    """
    if not enabled:
        # Return a no-op context manager
        class NoOpContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NoOpContext()

    try:
        import torch
        if dtype == "bfloat16":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)
        else:
            return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    except (ImportError, AttributeError):
        class NoOpContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return NoOpContext()
