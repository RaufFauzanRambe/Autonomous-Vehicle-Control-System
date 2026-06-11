"""
Learning Rate Scheduler Module for Training.

Implements various LR scheduling strategies:
- Cosine annealing with warm restarts
- Step decay with configurable milestones
- Linear warmup with various post-warmup schedules
- OneCycle policy (Super-convergence)
- Custom schedules with callable functions
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class CosineAnnealingScheduler:
    """
    Cosine annealing learning rate scheduler.

    LR(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

    Supports warm restarts (SGDR) for periodic LR resets.
    """

    def __init__(
        self,
        optimizer: Any,
        lr_max: float = 1e-3,
        lr_min: float = 1e-6,
        T_max: int = 100,
        warmup_epochs: int = 5,
        warmup_lr: float = 1e-6,
        restart_period: Optional[int] = None,
        restart_mult: float = 2.0,
    ) -> None:
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.restart_period = restart_period
        self.restart_mult = restart_mult
        self._current_epoch = 0
        self._restart_count = 0

    def step(self, epoch: Optional[int] = None) -> float:
        """
        Update learning rate.

        Args:
            epoch: Current epoch (auto-incremented if None).

        Returns:
            Current learning rate.
        """
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

        # Warmup phase
        if self._current_epoch < self.warmup_epochs:
            warmup_factor = (self._current_epoch + 1) / self.warmup_epochs
            lr = self.warmup_lr + (self.lr_max - self.warmup_lr) * warmup_factor
        else:
            # Cosine annealing
            if self.restart_period is not None:
                # SGDR with warm restarts
                period = self.restart_period * (self.restart_mult ** self._restart_count)
                t = (self._current_epoch - self.warmup_epochs) % period
                if t == 0 and self._current_epoch > self.warmup_epochs:
                    self._restart_count += 1
                progress = t / period
            else:
                progress = (self._current_epoch - self.warmup_epochs) / max(1, self.T_max - self.warmup_epochs)

            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))

        # Apply LR to optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


class StepScheduler:
    """
    Step decay learning rate scheduler.

    Reduces LR by a factor at specified milestone epochs.
    """

    def __init__(
        self,
        optimizer: Any,
        initial_lr: float = 1e-3,
        milestones: List[int] = [30, 60, 90],
        gamma: float = 0.1,
        warmup_epochs: int = 5,
    ) -> None:
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self._current_epoch = 0

    def step(self, epoch: Optional[int] = None) -> float:
        """Update learning rate."""
        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

        if self._current_epoch < self.warmup_epochs:
            warmup_factor = (self._current_epoch + 1) / self.warmup_epochs
            lr = self.initial_lr * warmup_factor
        else:
            lr = self.initial_lr
            for milestone in self.milestones:
                if self._current_epoch >= milestone:
                    lr *= self.gamma

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class LinearWarmupScheduler:
    """
    Linear warmup followed by various decay schedules.

    During warmup: LR linearly increases from 0 to base_lr.
    After warmup: Apply cosine, step, or constant schedule.
    """

    def __init__(
        self,
        optimizer: Any,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        post_warmup_schedule: str = "cosine",  # cosine, linear, constant
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.post_warmup_schedule = post_warmup_schedule
        self._current_step = 0

    def step(self) -> float:
        """Update learning rate."""
        self._current_step += 1

        if self._current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self._current_step / self.warmup_steps
        else:
            # Post-warmup schedule
            progress = (self._current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)

            if self.post_warmup_schedule == "cosine":
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            elif self.post_warmup_schedule == "linear":
                lr = self.base_lr - (self.base_lr - self.min_lr) * progress
            else:  # constant
                lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class OneCycleScheduler:
    """
    OneCycle learning rate scheduler (Super-convergence).

    Implements the 1Cycle policy from Smith & Topin (2018):
    1. Increase LR from initial to max over first 45% of training
    2. Decrease LR from max to min over remaining 55%
    3. Optional final annealing phase

    This policy can achieve comparable accuracy in far fewer epochs.
    """

    def __init__(
        self,
        optimizer: Any,
        max_lr: float = 1e-2,
        total_steps: int = 100000,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        three_phase: bool = False,
    ) -> None:
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.three_phase = three_phase
        self._current_step = 0

        # Compute phase boundaries
        self._step_up = int(total_steps * pct_start)
        self._step_down = total_steps if not three_phase else int(total_steps * 0.9)

    def step(self) -> float:
        """Update learning rate."""
        self._current_step += 1

        if self._current_step <= self._step_up:
            # Phase 1: Increase LR
            progress = self._current_step / max(1, self._step_up)
            if self.anneal_strategy == "cos":
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (
                    1 - math.cos(math.pi * progress)
                ) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        elif self._current_step <= self._step_down:
            # Phase 2: Decrease LR
            progress = (self._current_step - self._step_up) / max(1, self._step_down - self._step_up)
            if self.anneal_strategy == "cos":
                lr = self.max_lr - (self.max_lr - self.final_lr) * (
                    1 - math.cos(math.pi * progress)
                ) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * progress
        else:
            # Phase 3: Final annealing (if three_phase)
            progress = (self._current_step - self._step_down) / max(1, self.total_steps - self._step_down)
            lr = self.final_lr * (1 - progress)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]


def create_scheduler(
    optimizer: Any,
    scheduler_type: str = "cosine",
    epochs: int = 100,
    steps_per_epoch: int = 1000,
    warmup_epochs: int = 5,
    lr: float = 1e-3,
    min_lr: float = 1e-6,
    **kwargs: Any,
) -> Any:
    """
    Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler.
        epochs: Total training epochs.
        steps_per_epoch: Steps per epoch.
        warmup_epochs: Number of warmup epochs.
        lr: Base learning rate.
        min_lr: Minimum learning rate.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        Configured scheduler instance.
    """
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == "cosine":
        try:
            import torch.optim.lr_scheduler as lr_scheduler
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr
            )
            if warmup_epochs > 0:
                scheduler = lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[
                        lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
                        scheduler,
                    ],
                    milestones=[warmup_epochs],
                )
            return scheduler
        except ImportError:
            return CosineAnnealingScheduler(
                optimizer, lr_max=lr, lr_min=min_lr, T_max=epochs,
                warmup_epochs=warmup_epochs,
            )

    elif scheduler_type == "step":
        milestones = kwargs.get("milestones", [epochs // 3, 2 * epochs // 3])
        gamma = kwargs.get("gamma", 0.1)
        try:
            import torch.optim.lr_scheduler as lr_scheduler
            return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        except ImportError:
            return StepScheduler(optimizer, initial_lr=lr, milestones=milestones, gamma=gamma)

    elif scheduler_type == "warmup":
        return LinearWarmupScheduler(
            optimizer, warmup_steps=warmup_steps, total_steps=total_steps,
            base_lr=lr, min_lr=min_lr,
        )

    elif scheduler_type == "onecycle":
        try:
            import torch.optim.lr_scheduler as lr_scheduler
            return lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps,
                pct_start=0.3, anneal_strategy="cos",
            )
        except ImportError:
            return OneCycleScheduler(
                optimizer, max_lr=lr, total_steps=total_steps,
            )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
