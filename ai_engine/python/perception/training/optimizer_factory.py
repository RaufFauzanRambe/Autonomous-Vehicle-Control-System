"""
Optimizer Factory Module for Training.

Creates optimizers with proper configuration:
- SGD with momentum and Nesterov acceleration
- Adam with configurable betas
- AdamW with decoupled weight decay
- LAMB for large-batch training
- Layer-wise learning rate decay support
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


class OptimizerFactory:
    """
    Factory for creating optimizers with standard configurations.

    Provides a unified interface for creating various optimizer types
    with sensible defaults and configurable parameters.

    Example:
        >>> optimizer = create_optimizer(model.parameters(), "adamw", lr=1e-3, weight_decay=1e-4)
    """

    # Default configurations for each optimizer type
    DEFAULTS = {
        "sgd": {
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "nesterov": True,
        },
        "adam": {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
        },
        "adamw": {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-2,
            "amsgrad": False,
        },
        "lamb": {
            "lr": 1e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0.01,
            "adam": False,
        },
    }

    @staticmethod
    def create(
        params: Iterable,
        optimizer_type: str = "adamw",
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        momentum: float = 0.9,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        nesterov: bool = True,
        amsgrad: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Create an optimizer.

        Args:
            params: Model parameters to optimize.
            optimizer_type: Type of optimizer (sgd, adam, adamw, lamb).
            lr: Learning rate (overrides default).
            weight_decay: Weight decay (overrides default).
            momentum: Momentum for SGD.
            betas: Beta values for Adam/AdamW/LAMB.
            eps: Epsilon for numerical stability.
            nesterov: Use Nesterov momentum for SGD.
            amsgrad: Use AMSGrad variant for Adam/AdamW.
            **kwargs: Additional optimizer-specific arguments.

        Returns:
            Configured optimizer instance.

        Raises:
            ValueError: If optimizer_type is not recognized.
        """
        try:
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required for optimizer creation")

        defaults = OptimizerFactory.DEFAULTS.get(optimizer_type, {})
        lr = lr if lr is not None else defaults.get("lr", 1e-3)
        weight_decay = weight_decay if weight_decay is not None else defaults.get("weight_decay", 0.0)

        if optimizer_type == "sgd":
            return optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        elif optimizer_type == "adam":
            return optim.Adam(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        elif optimizer_type == "adamw":
            return optim.AdamW(
                params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        elif optimizer_type == "lamb":
            return OptimizerFactory._create_lamb(params, lr, betas, eps, weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. "
                             f"Supported: sgd, adam, adamw, lamb")

    @staticmethod
    def _create_lamb(
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
    ) -> Any:
        """
        Create a LAMB optimizer.

        LAMB (Large Batch Optimization for Deep Learning) enables
        training with very large batch sizes by normalizing the
        update ratio.

        Reference: You et al., "Large Batch Optimization for Deep Learning:
        Training BERT in 76 minutes" (2019)
        """
        try:
            from torch_optimizer import Lamb
            return Lamb(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        except ImportError:
            # Fallback to AdamW with a warning
            import torch.optim as optim
            print("[Optimizer] LAMB not available (install torch-optimizer). "
                  "Falling back to AdamW.")
            return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @staticmethod
    def create_with_param_groups(
        model: Any,
        optimizer_type: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        backbone_lr_scale: float = 0.1,
        bias_lr_scale: float = 2.0,
        no_weight_decay_keys: Optional[List[str]] = None,
    ) -> Any:
        """
        Create optimizer with parameter groups for layer-wise LR scaling.

        Applies different learning rates and weight decay to different
        parameter groups (e.g., backbone vs. head, bias vs. weights).

        Args:
            model: The model whose parameters to optimize.
            optimizer_type: Optimizer type.
            lr: Base learning rate.
            weight_decay: Base weight decay.
            backbone_lr_scale: LR multiplier for backbone parameters.
            bias_lr_scale: LR multiplier for bias parameters.
            no_weight_decay_keys: Parameter name patterns that should not have weight decay.

        Returns:
            Configured optimizer with parameter groups.
        """
        try:
            import torch.optim as optim
        except ImportError:
            raise ImportError("PyTorch is required")

        no_wd_keys = no_weight_decay_keys or ["bias", "LayerNorm.weight", "BatchNorm"]

        # Separate parameters into groups
        backbone_params = []
        head_params = []
        no_wd_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if parameter should have no weight decay
            has_no_wd = any(key in name for key in no_wd_keys)

            if has_no_wd:
                no_wd_params.append(param)
            elif "backbone" in name or "encoder" in name or "feature" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {
                "params": backbone_params,
                "lr": lr * backbone_lr_scale,
                "weight_decay": weight_decay,
                "name": "backbone",
            },
            {
                "params": head_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": "head",
            },
            {
                "params": no_wd_params,
                "lr": lr * bias_lr_scale,
                "weight_decay": 0.0,
                "name": "no_weight_decay",
            },
        ]

        # Remove empty groups
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        if optimizer_type == "sgd":
            return optim.SGD(param_groups, momentum=0.9, nesterov=True)
        elif optimizer_type == "adamw":
            return optim.AdamW(param_groups)
        else:
            return optim.Adam(param_groups)


# Convenience function
def create_optimizer(
    params: Iterable,
    optimizer_type: str = "adamw",
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Create an optimizer using the OptimizerFactory."""
    return OptimizerFactory.create(params, optimizer_type, lr, weight_decay, **kwargs)
