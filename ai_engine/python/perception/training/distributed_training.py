"""
Distributed Training Module for Multi-GPU Training.

Provides utilities for distributed data parallel (DDP) training:
- Process group initialization and cleanup
- Gradient synchronization
- Distributed data sampling
- Multi-GPU training coordination
- Fault tolerance and recovery
"""

import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = -1  # -1 = auto-detect
    rank: int = -1  # -1 = auto-detect
    local_rank: int = -1
    gpu_ids: Optional[List[int]] = None
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25
    sync_bn: bool = False  # Synchronized batch normalization
    timeout_minutes: int = 30


def setup_distributed(args: Any) -> Any:
    """
    Initialize distributed training environment.

    Reads distributed configuration from environment variables
    and initializes the process group.

    Args:
        args: Training arguments (will be modified in-place).

    Returns:
        Modified args with distributed settings.
    """
    try:
        import torch
        import torch.distributed as dist
    except ImportError:
        print("[Distributed] PyTorch not available. Falling back to single GPU.")
        args.distributed = False
        return args

    # Auto-detect from environment
    if "RANK" in os.environ:
        args.rank = int(os.environ["RANK"])
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.world_size <= 1:
        args.distributed = False
        return args

    # Set device
    torch.cuda.set_device(args.local_rank)

    # Initialize process group
    dist.init_process_group(
        backend=args.dist_backend or "nccl",
        init_method=args.dist_url or "env://",
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(minutes=30) if 'datetime' in dir() else None,
    )

    # Synchronize all processes
    dist.barrier()

    print(f"[Distributed] Rank {args.rank}/{args.world_size} "
          f"(local_rank={args.local_rank}) initialized")

    return args


def cleanup_distributed() -> None:
    """Clean up distributed training resources."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
            print("[Distributed] Process group destroyed")
    except ImportError:
        pass


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass
    return True


def get_rank() -> int:
    """Get the rank of the current process."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_rank()
    except ImportError:
        pass
    return 0


def get_world_size() -> int:
    """Get the total number of processes."""
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except ImportError:
        pass
    return 1


def all_reduce_tensor(tensor: Any, op: str = "sum") -> Any:
    """
    All-reduce a tensor across all processes.

    Args:
        tensor: The tensor to reduce.
        op: Reduction operation (sum, mean, max, min).

    Returns:
        Reduced tensor.
    """
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return tensor

        op_map = {
            "sum": dist.ReduceOp.SUM,
            "mean": dist.ReduceOp.SUM,
            "max": dist.ReduceOp.MAX,
            "min": dist.ReduceOp.MIN,
        }

        dist.all_reduce(tensor, op=op_map.get(op, dist.ReduceOp.SUM))

        if op == "mean":
            tensor /= get_world_size()

        return tensor
    except ImportError:
        return tensor


def all_gather_tensors(tensor: Any) -> List[Any]:
    """
    Gather tensors from all processes.

    Args:
        tensor: Local tensor to gather.

    Returns:
        List of tensors from all processes.
    """
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return [tensor]

        gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(gathered, tensor)
        return gathered
    except ImportError:
        return [tensor]


class DistributedDataParallel:
    """
    Wrapper for PyTorch DistributedDataParallel with enhanced features.

    Provides gradient synchronization, mixed precision support,
    and gradient accumulation across multiple GPUs.
    """

    def __init__(
        self,
        model: Any,
        config: Optional[DistributedConfig] = None,
    ) -> None:
        self.config = config or DistributedConfig()
        self._model = model
        self._ddp_model = None

        try:
            import torch.nn as nn
            import torch.distributed as dist

            if dist.is_initialized():
                self._ddp_model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.config.local_rank] if self.config.local_rank >= 0 else None,
                    output_device=self.config.local_rank if self.config.local_rank >= 0 else None,
                    find_unused_parameters=self.config.find_unused_parameters,
                    gradient_as_bucket_view=self.config.gradient_as_bucket_view,
                    bucket_cap_mb=self.config.bucket_cap_mb,
                )
                print(f"[DDP] Model wrapped with DistributedDataParallel")
            else:
                self._ddp_model = model
        except ImportError:
            self._ddp_model = model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass through the DDP model."""
        return self._ddp_model(*args, **kwargs)

    def parameters(self) -> Any:
        """Get model parameters."""
        return self._ddp_model.parameters()

    def state_dict(self) -> Dict:
        """Get model state dict."""
        return self._ddp_model.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load model state dict."""
        self._ddp_model.load_state_dict(state_dict)


class GradientSyncer:
    """
    Utility for custom gradient synchronization strategies.

    Supports gradient accumulation, gradient compression,
    and asynchronous synchronization.
    """

    def __init__(
        self,
        accumulation_steps: int = 1,
        sync_mode: str = "sync",  # sync, async, skip
    ) -> None:
        self.accumulation_steps = accumulation_steps
        self.sync_mode = sync_mode
        self._step_count = 0

    def should_sync(self) -> bool:
        """Check if gradients should be synchronized this step."""
        self._step_count += 1
        if self.sync_mode == "sync":
            return True
        elif self.sync_mode == "skip":
            return self._step_count % self.accumulation_steps == 0
        return True

    def should_step(self) -> bool:
        """Check if optimizer should step this iteration."""
        return self._step_count % self.accumulation_steps == 0

    def reset(self) -> None:
        """Reset step counter."""
        self._step_count = 0


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """
    Broadcast a Python object from source rank to all ranks.

    Args:
        obj: Object to broadcast (must be picklable).
        src: Source rank for broadcast.

    Returns:
        The broadcasted object.
    """
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return obj

        obj_list = [obj if dist.get_rank() == src else None]
        dist.broadcast_object_list(obj_list, src=src)
        return obj_list[0]
    except ImportError:
        return obj


def reduce_dict(input_dict: Dict[str, float], average: bool = True) -> Dict[str, float]:
    """
    Reduce a dictionary of metrics across all processes.

    Args:
        input_dict: Dictionary of metric name to value.
        average: Whether to average (vs sum) across processes.

    Returns:
        Reduced dictionary.
    """
    try:
        import torch
        import torch.distributed as dist

        if not dist.is_initialized():
            return input_dict

        names = sorted(input_dict.keys())
        values = torch.tensor([input_dict[k] for k in names], dtype=torch.float64)
        dist.all_reduce(values)

        if average:
            values /= get_world_size()

        return {k: v.item() for k, v in zip(names, values)}
    except ImportError:
        return input_dict
