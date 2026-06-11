"""
Checkpoint Manager Module for Training.

Handles saving, loading, and managing model checkpoints:
- Periodic checkpoint saving
- Best model tracking
- Auto-resume from latest checkpoint
- Checkpoint pruning (keep last N)
- Training state preservation
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""
    path: str
    epoch: int
    global_step: int
    metric: float
    timestamp: float
    is_best: bool = False
    file_size_mb: float = 0.0


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Features:
    - Save periodic and best-model checkpoints
    - Auto-resume from latest checkpoint
    - Keep only N most recent checkpoints
    - Track training state for exact resumption
    - Atomic writes to prevent corrupted checkpoints

    Example:
        >>> mgr = CheckpointManager("./checkpoints", keep_last_n=3)
        >>> mgr.save(model, optimizer, epoch=10, metrics={"val_loss": 0.5}, step=1000)
        >>> mgr.save_best(model, optimizer, epoch=10, metric=0.3, step=1000)
        >>> checkpoint = mgr.load_latest(model, optimizer)
    """

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        keep_last_n: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_rng_state: bool = True,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            save_dir: Directory for saving checkpoints.
            keep_last_n: Number of recent checkpoints to keep.
            save_optimizer: Whether to save optimizer state.
            save_scheduler: Whether to save scheduler state.
            save_rng_state: Whether to save RNG state for reproducibility.
        """
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_rng_state = save_rng_state

        self._best_metric = -float("inf")
        self._best_checkpoint_path: Optional[str] = None
        self._checkpoint_history: List[CheckpointInfo] = []

        os.makedirs(save_dir, exist_ok=True)

    def save(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        global_step: int = 0,
        scheduler: Any = None,
        extra_state: Optional[Dict] = None,
    ) -> str:
        """
        Save a training checkpoint.

        Args:
            model: The model to save.
            optimizer: The optimizer state.
            epoch: Current epoch number.
            metrics: Current metrics dictionary.
            global_step: Global training step.
            scheduler: LR scheduler state.
            extra_state: Additional state to preserve.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_dir = os.path.join(self.save_dir, f"epoch_{epoch:04d}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics or {},
            "best_metric": self._best_metric,
            "timestamp": time.time(),
        }

        # Save model
        try:
            import torch
            checkpoint["model_state_dict"] = model.state_dict()

            if self.save_optimizer and optimizer is not None:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            if self.save_scheduler and scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            if self.save_rng_state:
                checkpoint["rng_state"] = {
                    "torch": torch.random.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(),
                }

            # Atomic write
            temp_path = os.path.join(checkpoint_dir, "checkpoint.tmp")
            torch.save(checkpoint, temp_path)
            final_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            os.replace(temp_path, final_path)

        except ImportError:
            # Fallback: save as JSON
            json_checkpoint = {
                k: v for k, v in checkpoint.items()
                if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state")
            }
            with open(os.path.join(checkpoint_dir, "checkpoint.json"), "w") as f:
                json.dump(json_checkpoint, f, indent=2)

        # Save metadata
        file_size = 0
        for f in os.listdir(checkpoint_dir):
            file_size += os.path.getsize(os.path.join(checkpoint_dir, f))

        info = CheckpointInfo(
            path=checkpoint_dir,
            epoch=epoch,
            global_step=global_step,
            metric=metrics.get("val_loss", float("inf")) if metrics else float("inf"),
            timestamp=time.time(),
            file_size_mb=file_size / (1024 * 1024),
        )
        self._checkpoint_history.append(info)

        # Prune old checkpoints
        self._prune_checkpoints()

        return checkpoint_dir

    def save_best(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: int = 0,
        metric: float = 0.0,
        global_step: int = 0,
    ) -> str:
        """
        Save as best model if metric improved.

        Args:
            model: The model to save.
            optimizer: The optimizer state.
            epoch: Current epoch number.
            metric: Metric value (lower is better for loss).
            global_step: Global training step.

        Returns:
            Path to the saved checkpoint.
        """
        # For loss-type metrics, lower is better
        is_better = metric < self._best_metric if self._best_metric != -float("inf") else True

        if is_better:
            self._best_metric = metric
            best_dir = os.path.join(self.save_dir, "best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            os.makedirs(best_dir, exist_ok=True)

            try:
                import torch
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metric": self._best_metric,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                    "timestamp": time.time(),
                }
                torch.save(checkpoint, os.path.join(best_dir, "checkpoint.pt"))
            except ImportError:
                with open(os.path.join(best_dir, "best_info.json"), "w") as f:
                    json.dump({"epoch": epoch, "metric": metric, "step": global_step}, f, indent=2)

            self._best_checkpoint_path = best_dir
            print(f"[Checkpoint] New best model saved (metric={metric:.4f})")

        return self._best_checkpoint_path or ""

    def load(
        self,
        path: str,
        model: Any = None,
        optimizer: Any = None,
        scheduler: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint from disk.

        Args:
            path: Path to checkpoint directory or file.
            model: Model to load weights into.
            optimizer: Optimizer to load state into.
            scheduler: Scheduler to load state into.

        Returns:
            Checkpoint metadata dictionary, or None if failed.
        """
        # Resolve checkpoint file path
        if os.path.isdir(path):
            pt_path = os.path.join(path, "checkpoint.pt")
            json_path = os.path.join(path, "checkpoint.json")
            if os.path.exists(pt_path):
                path = pt_path
            elif os.path.exists(json_path):
                path = json_path

        if not os.path.exists(path):
            print(f"[Checkpoint] File not found: {path}")
            return None

        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

            if model is not None and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[Checkpoint] Model loaded from {path}")

            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Restore RNG state
            if "rng_state" in checkpoint and self.save_rng_state:
                rng_state = checkpoint["rng_state"]
                if rng_state.get("torch") is not None:
                    torch.random.set_rng_state(rng_state["torch"])
                if rng_state.get("numpy") is not None:
                    np.random.set_state(rng_state["numpy"])

            self._best_metric = checkpoint.get("best_metric", -float("inf"))

            return {
                "epoch": checkpoint.get("epoch", 0),
                "global_step": checkpoint.get("global_step", 0),
                "best_metric": checkpoint.get("best_metric", -float("inf")),
                "metrics": checkpoint.get("metrics", {}),
            }

        except ImportError:
            # Try JSON fallback
            if path.endswith(".json"):
                with open(path, "r") as f:
                    data = json.load(f)
                return data
            return None

    def load_latest(self, model: Any = None, optimizer: Any = None) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        latest = self._find_latest_checkpoint()
        if latest:
            return self.load(latest, model, optimizer)
        return None

    def load_best(self, model: Any = None, optimizer: Any = None) -> Optional[Dict]:
        """Load the best checkpoint."""
        best_dir = os.path.join(self.save_dir, "best")
        if os.path.exists(best_dir):
            return self.load(best_dir, model, optimizer)
        return None

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint directory."""
        if not os.path.exists(self.save_dir):
            return None

        checkpoints = []
        for name in os.listdir(self.save_dir):
            if name.startswith("epoch_"):
                full_path = os.path.join(self.save_dir, name)
                if os.path.isdir(full_path):
                    checkpoints.append(full_path)

        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return checkpoints[0]

    def _prune_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return

        checkpoint_dirs = []
        for name in os.listdir(self.save_dir):
            if name.startswith("epoch_"):
                full_path = os.path.join(self.save_dir, name)
                if os.path.isdir(full_path):
                    checkpoint_dirs.append(full_path)

        if len(checkpoint_dirs) <= self.keep_last_n:
            return

        # Sort by timestamp
        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x))

        # Remove oldest
        to_remove = checkpoint_dirs[:-self.keep_last_n]
        for path in to_remove:
            try:
                shutil.rmtree(path)
                print(f"[Checkpoint] Removed old checkpoint: {path}")
            except OSError as e:
                print(f"[Checkpoint] Failed to remove {path}: {e}")

    def get_checkpoint_info(self) -> List[Dict]:
        """Get information about all saved checkpoints."""
        info_list = []
        for cp_info in self._checkpoint_history:
            info_list.append({
                "path": cp_info.path,
                "epoch": cp_info.epoch,
                "global_step": cp_info.global_step,
                "metric": cp_info.metric,
                "timestamp": cp_info.timestamp,
                "is_best": cp_info.is_best,
                "size_mb": cp_info.file_size_mb,
            })
        return info_list
