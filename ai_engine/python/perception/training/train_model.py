"""
Main Training Script for Autonomous Vehicle AI Models.

Provides a unified entry point for training various model types:
- Object detection
- Semantic segmentation
- Depth estimation
- Lane detection
- End-to-end driving

Features:
- Distributed training launch
- Comprehensive argument parsing
- Automatic checkpointing and resumption
- Mixed precision training
- Multi-GPU support
"""

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrainArgs:
    """Training arguments."""
    # Model
    model_name: str = "resnet50"
    model_type: str = "detection"  # detection, segmentation, depth, lane, e2e
    pretrained: bool = True
    num_classes: int = 10

    # Data
    dataset: str = "kitti"
    data_root: str = "./data"
    train_split: str = "train"
    val_split: str = "val"
    num_workers: int = 4
    pin_memory: bool = True

    # Training
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "adamw"  # sgd, adam, adamw, lamb
    scheduler: str = "cosine"  # step, cosine, warmup, onecycle

    # Augmentation
    augmentation: str = "auto"  # none, light, medium, heavy, auto
    mosaic: bool = True
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # Mixed precision
    amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16

    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_url: str = "env://"
    dist_backend: str = "nccl"

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume: Optional[str] = None
    save_freq: int = 5  # Save every N epochs
    save_best: bool = True
    keep_last_n: int = 3

    # Logging
    log_dir: str = "./logs"
    log_freq: int = 50  # Log every N batches
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "av-control"

    # Evaluation
    eval_freq: int = 1  # Evaluate every N epochs
    eval_only: bool = False

    # Early stopping
    early_stop: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # Hyperparameter search
    hpo: bool = False
    hpo_framework: str = "optuna"  # optuna, grid, random
    hpo_trials: int = 50
    hpo_study_name: str = "av_hpo"

    # Export
    export_onnx: bool = False
    onnx_path: Optional[str] = None
    onnx_simplify: bool = True
    onnx_dynamic: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    benchmark: bool = True

    # System
    device: str = "auto"
    gpu_id: Optional[int] = None
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


def parse_args() -> TrainArgs:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train autonomous vehicle AI models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument("--model-name", type=str, default="resnet50")
    parser.add_argument("--model-type", type=str, default="detection",
                        choices=["detection", "segmentation", "depth", "lane", "e2e"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--num-classes", type=int, default=10)

    # Data arguments
    parser.add_argument("--dataset", type=str, default="kitti",
                        choices=["coco", "kitti", "custom", "nuscenes", "waymo"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--num-workers", type=int, default=4)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["sgd", "adam", "adamw", "lamb"])
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["step", "cosine", "warmup", "onecycle"])

    # Augmentation
    parser.add_argument("--augmentation", type=str, default="auto")
    parser.add_argument("--mosaic", action="store_true", default=True)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--amp-dtype", type=str, default="float16")

    # Distributed
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-freq", type=int, default=5)
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--keep-last-n", type=int, default=3)

    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true", default=False)

    # Early stopping
    parser.add_argument("--early-stop", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=15)

    # HPO
    parser.add_argument("--hpo", action="store_true", default=False)
    parser.add_argument("--hpo-framework", type=str, default="optuna")
    parser.add_argument("--hpo-trials", type=int, default=50)

    # Export
    parser.add_argument("--export-onnx", action="store_true", default=False)
    parser.add_argument("--onnx-path", type=str, default=None)
    parser.add_argument("--onnx-simplify", action="store_true", default=True)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", default=False)

    # System
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    args = parser.parse_args()

    # Convert to TrainArgs dataclass
    train_args = TrainArgs(
        model_name=args.model_name,
        model_type=args.model_type,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        dataset=args.dataset,
        data_root=args.data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        augmentation=args.augmentation,
        mosaic=args.mosaic,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        distributed=args.distributed,
        world_size=args.world_size,
        rank=args.rank,
        local_rank=args.local_rank,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        save_freq=args.save_freq,
        save_best=args.save_best,
        keep_last_n=args.keep_last_n,
        log_dir=args.log_dir,
        log_freq=args.log_freq,
        tensorboard=args.tensorboard,
        wandb=args.wandb,
        eval_freq=args.eval_freq,
        eval_only=args.eval_only,
        early_stop=args.early_stop,
        patience=args.patience,
        hpo=args.hpo,
        hpo_framework=args.hpo_framework,
        hpo_trials=args.hpo_trials,
        export_onnx=args.export_onnx,
        onnx_path=args.onnx_path,
        onnx_simplify=args.onnx_simplify,
        seed=args.seed,
        deterministic=args.deterministic,
        device=args.device,
        gpu_id=args.gpu_id,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
    )

    return train_args


def setup_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def setup_device(device: str, gpu_id: Optional[int] = None) -> str:
    """Resolve and setup compute device."""
    try:
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if gpu_id is not None and device == "cuda":
            torch.cuda.set_device(gpu_id)
        return device
    except ImportError:
        return "cpu"


class Trainer:
    """
    Main training orchestrator.

    Coordinates all training components including data loading,
    model creation, optimization, logging, and checkpointing.
    """

    def __init__(self, args: TrainArgs) -> None:
        self.args = args
        self.device = setup_device(args.device, args.gpu_id)
        setup_seed(args.seed, args.deterministic)

        self._start_epoch = 0
        self._best_metric = -float("inf")
        self._patience_counter = 0
        self._global_step = 0
        self._training_history: List[Dict[str, Any]] = []

        # Initialize components (lazy imports for optional deps)
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._train_loader = None
        self._val_loader = None
        self._logger = None
        self._checkpoint_mgr = None

        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._should_stop = False

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle interrupt signals for graceful shutdown."""
        print(f"\n[Trainer] Received signal {signum}. Saving checkpoint and exiting...")
        self._should_stop = True

    def setup(self) -> None:
        """Initialize all training components."""
        print("[Trainer] Setting up training...")

        # Setup data loaders
        self._setup_data()

        # Setup model
        self._setup_model()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Setup mixed precision
        self._setup_amp()

        # Setup logging
        self._setup_logging()

        # Setup checkpointing
        self._setup_checkpointing()

        # Resume from checkpoint if specified
        if self.args.resume:
            self._resume_checkpoint()

        print(f"[Trainer] Setup complete. Device: {self.device}")

    def _setup_data(self) -> None:
        """Setup training and validation data loaders."""
        from .dataset_loader import create_dataloader, DatasetConfig

        config = DatasetConfig(
            dataset=self.args.dataset,
            data_root=self.args.data_root,
            train_split=self.args.train_split,
            val_split=self.args.val_split,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            augmentation=self.args.augmentation,
        )

        self._train_loader = create_dataloader(config, split="train")
        self._val_loader = create_dataloader(config, split="val")

        print(f"[Trainer] Train samples: {len(self._train_loader.dataset) if hasattr(self._train_loader, 'dataset') else 'N/A'}")
        print(f"[Trainer] Val samples: {len(self._val_loader.dataset) if hasattr(self._val_loader, 'dataset') else 'N/A'}")

    def _setup_model(self) -> None:
        """Setup the model architecture."""
        try:
            import torch
            import torch.nn as nn

            # Simplified model creation based on type
            if self.args.model_type == "detection":
                self._model = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, self.args.num_classes),
                )
            else:
                self._model = nn.Sequential(
                    nn.Linear(48, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, self.args.num_classes),
                )

            self._model = self._model.to(self.device)
            param_count = sum(p.numel() for p in self._model.parameters())
            print(f"[Trainer] Model parameters: {param_count:,}")
        except ImportError:
            print("[Trainer] PyTorch not available. Model setup skipped.")

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        try:
            import torch

            if self._model is None:
                return

            from .optimizer_factory import create_optimizer
            from .scheduler import create_scheduler

            self._optimizer = create_optimizer(
                self._model.parameters(),
                optimizer_type=self.args.optimizer,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )

            self._scheduler = create_scheduler(
                self._optimizer,
                scheduler_type=self.args.scheduler,
                epochs=self.args.epochs,
                steps_per_epoch=1000,  # Will be updated
                warmup_epochs=5,
            )

        except ImportError:
            print("[Trainer] PyTorch not available. Optimizer setup skipped.")

    def _setup_amp(self) -> None:
        """Setup automatic mixed precision."""
        if self.args.amp:
            try:
                from .mixed_precision import create_grad_scaler
                self._scaler = create_grad_scaler(self.args.amp_dtype)
                print(f"[Trainer] AMP enabled ({self.args.amp_dtype})")
            except ImportError:
                print("[Trainer] AMP not available. Training in full precision.")
                self._scaler = None
        else:
            self._scaler = None

    def _setup_logging(self) -> None:
        """Setup logging infrastructure."""
        from .tensorboard_logger import TensorBoardLogger

        self._logger = TensorBoardLogger(
            log_dir=self.args.log_dir,
            enabled=self.args.tensorboard,
        )

    def _setup_checkpointing(self) -> None:
        """Setup checkpoint management."""
        from .checkpoint_manager import CheckpointManager

        self._checkpoint_mgr = CheckpointManager(
            save_dir=self.args.checkpoint_dir,
            keep_last_n=self.args.keep_last_n,
        )

    def _resume_checkpoint(self) -> None:
        """Resume training from a checkpoint."""
        if self._checkpoint_mgr:
            checkpoint = self._checkpoint_mgr.load(self.args.resume, self._model, self._optimizer)
            if checkpoint:
                self._start_epoch = checkpoint.get("epoch", 0)
                self._best_metric = checkpoint.get("best_metric", -float("inf"))
                self._global_step = checkpoint.get("global_step", 0)
                print(f"[Trainer] Resumed from epoch {self._start_epoch}")

    def train(self) -> Dict[str, Any]:
        """
        Run the complete training loop.

        Returns:
            Dictionary of training results and statistics.
        """
        print(f"\n{'=' * 60}")
        print(f"Training: {self.args.model_name} ({self.args.model_type})")
        print(f"Epochs: {self.args.epochs}, Batch size: {self.args.batch_size}")
        print(f"Optimizer: {self.args.optimizer}, LR: {self.args.learning_rate}")
        print(f"Device: {self.device}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        for epoch in range(self._start_epoch, self.args.epochs):
            if self._should_stop:
                break

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Evaluate
            val_metrics = {}
            if (epoch + 1) % self.args.eval_freq == 0:
                val_metrics = self._validate_epoch(epoch)

                # Check for improvement
                current_metric = val_metrics.get("val_loss", float("inf"))
                if current_metric < self._best_metric or self._best_metric == -float("inf"):
                    self._best_metric = current_metric
                    self._patience_counter = 0
                    if self.args.save_best and self._checkpoint_mgr:
                        self._checkpoint_mgr.save_best(
                            self._model, self._optimizer, epoch,
                            current_metric, self._global_step,
                        )
                else:
                    self._patience_counter += 1

                # Early stopping
                if self.args.early_stop and self._patience_counter >= self.args.patience:
                    print(f"[Trainer] Early stopping at epoch {epoch}")
                    break

            # Save periodic checkpoint
            if (epoch + 1) % self.args.save_freq == 0 and self._checkpoint_mgr:
                self._checkpoint_mgr.save(
                    self._model, self._optimizer, epoch,
                    val_metrics, self._global_step,
                )

            # Logging
            self._log_epoch(epoch, train_metrics, val_metrics)

            # Step scheduler
            if self._scheduler:
                self._scheduler.step()

        # Final export
        if self.args.export_onnx and self._model is not None:
            self._export_model()

        elapsed = time.time() - start_time
        results = {
            "total_epochs": epoch + 1,
            "best_metric": self._best_metric,
            "total_time": elapsed,
            "global_step": self._global_step,
        }

        print(f"\n[Trainer] Training complete. Best metric: {self._best_metric:.4f}")
        return results

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        try:
            import torch

            self._model.train()
            total_loss = 0.0
            n_batches = 0

            for batch_idx, batch in enumerate(self._train_loader):
                if self._should_stop:
                    break

                # Forward pass
                if isinstance(batch, dict):
                    inputs = batch["image"].to(self.device) if "image" in batch else batch.get("obs", torch.zeros(1, 48))
                    targets = batch.get("label", batch.get("action", torch.zeros(1, self.args.num_classes)))
                else:
                    inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

                if self._scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self._model(inputs)
                        loss = torch.nn.functional.cross_entropy(outputs, targets.argmax(dim=1) if targets.dim() > 1 else targets.long())
                    self._scaler.scale(loss).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    outputs = self._model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, targets.argmax(dim=1) if targets.dim() > 1 else targets.long())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.args.max_grad_norm)
                    self._optimizer.step()

                self._optimizer.zero_grad()
                total_loss += loss.item()
                n_batches += 1
                self._global_step += 1

            avg_loss = total_loss / max(n_batches, 1)
            return {"train_loss": avg_loss}

        except Exception as e:
            print(f"[Trainer] Training error: {e}")
            return {"train_loss": float("inf")}

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one validation epoch."""
        try:
            import torch

            self._model.eval()
            total_loss = 0.0
            n_batches = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in self._val_loader:
                    if isinstance(batch, dict):
                        inputs = batch["image"].to(self.device) if "image" in batch else batch.get("obs", torch.zeros(1, 48))
                        targets = batch.get("label", batch.get("action", torch.zeros(1, self.args.num_classes)))
                    else:
                        inputs, targets = batch[0].to(self.device), batch[1].to(self.device)

                    outputs = self._model(inputs)
                    loss = torch.nn.functional.cross_entropy(outputs, targets.argmax(dim=1) if targets.dim() > 1 else targets.long())
                    total_loss += loss.item()
                    n_batches += 1

                    pred = outputs.argmax(dim=1)
                    target_idx = targets.argmax(dim=1) if targets.dim() > 1 else targets.long()
                    correct += (pred == target_idx).sum().item()
                    total += targets.size(0)

            avg_loss = total_loss / max(n_batches, 1)
            accuracy = correct / max(total, 1)
            return {"val_loss": avg_loss, "val_accuracy": accuracy}

        except Exception as e:
            print(f"[Trainer] Validation error: {e}")
            return {"val_loss": float("inf"), "val_accuracy": 0.0}

    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict) -> None:
        """Log epoch metrics."""
        if self._logger:
            self._logger.log_scalars(
                step=self._global_step,
                metrics={**train_metrics, **val_metrics},
            )

        if (epoch + 1) % 5 == 0:
            train_loss = train_metrics.get("train_loss", 0)
            val_loss = val_metrics.get("val_loss", "N/A")
            print(f"[Epoch {epoch + 1}/{self.args.epochs}] "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss}")

    def _export_model(self) -> None:
        """Export model to ONNX format."""
        from .export_onnx import export_to_onnx

        onnx_path = self.args.onnx_path or os.path.join(
            self.args.checkpoint_dir, f"{self.args.model_name}.onnx"
        )
        try:
            export_to_onnx(
                model=self._model,
                output_path=onnx_path,
                input_shape=(1, 3, 224, 224),
                dynamic_shapes=self.args.onnx_dynamic,
                simplify=self.args.onnx_simplify,
            )
        except Exception as e:
            print(f"[Trainer] ONNX export failed: {e}")


def main() -> None:
    """Main entry point for training."""
    args = parse_args()

    # Distributed setup
    if args.distributed:
        from .distributed_training import setup_distributed
        args = setup_distributed(args)

    # Create trainer
    trainer = Trainer(args)
    trainer.setup()

    if args.eval_only:
        # Run evaluation only
        val_metrics = trainer._validate_epoch(0)
        print(f"Evaluation results: {val_metrics}")
        return

    # HPO mode
    if args.hpo:
        from .hyperparameter_search import run_hpo
        best_params = run_hpo(args)
        print(f"Best hyperparameters: {best_params}")
        return

    # Train
    results = trainer.train()
    print(f"Training results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
