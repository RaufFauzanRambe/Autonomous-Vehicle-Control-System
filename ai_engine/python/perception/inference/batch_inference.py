"""
Batch Inference Module for Autonomous Vehicle AI.

High-throughput batch inference processing:
- Dynamic batching with configurable timeout and size limits
- Throughput optimization with adaptive batch sizing
- Memory-aware batch size calculation based on available GPU/CPU memory
- Multi-model pipeline batching for coordinated inference
- Request queuing with priority and deadline awareness

Optimized for autonomous driving workloads where multiple camera
streams and sensor inputs must be processed concurrently.
"""

import os
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class BatchStrategy(Enum):
    """Batch formation strategy."""
    FIXED_SIZE = "fixed_size"           # Wait until batch is full
    TIMEOUT = "timeout"                 # Send on timeout regardless of size
    ADAPTIVE = "adaptive"               # Adjust based on throughput
    MEMORY_AWARE = "memory_aware"       # Size limited by available memory


@dataclass
class BatchRequest:
    """A single inference request in the batch queue."""
    request_id: str
    input_data: np.ndarray
    model_name: str = "default"
    priority: int = 0
    deadline_ms: float = 100.0
    callback: Optional[Callable] = None
    timestamp: float = 0.0
    result: Optional[Dict[str, np.ndarray]] = None
    completed: bool = False

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def elapsed_ms(self) -> float:
        """Time elapsed since request submission."""
        return (time.time() - self.timestamp) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded its deadline."""
        return self.elapsed_ms > self.deadline_ms


@dataclass
class BatchConfig:
    """Configuration for batch inference."""
    max_batch_size: int = 8
    min_batch_size: int = 1
    batch_timeout_ms: float = 10.0
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_queue_size: int = 500
    num_workers: int = 2
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    max_memory_mb: float = 4096.0
    per_sample_memory_mb: float = 5.0  # Estimated memory per sample
    throughput_target_fps: float = 100.0
    adaptive_adjust_interval: int = 50  # Batches between adjustments
    enable_priority: bool = True
    drop_expired: bool = True


@dataclass
class BatchResult:
    """Result from a batch inference operation."""
    batch_size: int = 0
    total_latency_ms: float = 0.0
    per_sample_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    dropped_expired: int = 0
    success: bool = True


class DynamicBatcher:
    """
    Dynamic batch formation engine.

    Collects individual inference requests and forms optimal batches
    based on the configured strategy. Supports timeout-based flushing,
    priority ordering, and deadline-aware request management.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self._pending: List[BatchRequest] = []
        self._lock = threading.Lock()
        self._last_flush_time = time.time()
        self._batch_counter = 0

    def add(self, request: BatchRequest) -> bool:
        """
        Add a request to the pending batch.

        Args:
            request: Inference request to add.

        Returns:
            True if the request was accepted.
        """
        with self._lock:
            if len(self._pending) >= self.config.max_queue_size:
                return False
            self._pending.append(request)
            return True

    def should_flush(self) -> bool:
        """
        Determine if the current pending requests should be flushed as a batch.

        Returns:
            True if the batch should be processed now.
        """
        with self._lock:
            if not self._pending:
                return False

            # Flush if batch is full
            if len(self._pending) >= self.config.max_batch_size:
                return True

            # Flush on timeout
            elapsed_ms = (time.time() - self._last_flush_time) * 1000
            if elapsed_ms >= self.config.batch_timeout_ms and self._pending:
                return True

            # Flush if any request is near deadline
            if self.config.drop_expired or self.config.enable_priority:
                for req in self._pending:
                    if req.elapsed_ms > req.deadline_ms * 0.8:
                        return True

            # Flush if minimum batch size reached and timeout is partially elapsed
            if (len(self._pending) >= self.config.min_batch_size
                    and elapsed_ms >= self.config.batch_timeout_ms * 0.5):
                return True

            return False

    def get_batch(self) -> List[BatchRequest]:
        """
        Get the current batch of requests and reset the pending list.

        Returns:
            List of requests forming the current batch.
        """
        with self._lock:
            if not self._pending:
                return []

            # Remove expired requests
            expired = []
            valid = []
            for req in self._pending:
                if req.is_expired and self.config.drop_expired:
                    expired.append(req)
                else:
                    valid.append(req)

            # Sort by priority (higher priority value = higher priority)
            if self.config.enable_priority:
                valid.sort(key=lambda r: -r.priority)

            # Limit to max batch size
            batch = valid[:self.config.max_batch_size]
            remaining = valid[self.config.max_batch_size:]

            # Mark expired requests as completed with empty result
            for req in expired:
                req.completed = True

            self._pending = remaining
            self._last_flush_time = time.time()
            self._batch_counter += 1

            return batch

    @property
    def pending_count(self) -> int:
        """Number of pending requests."""
        return len(self._pending)


class MemoryAwareBatchSizer:
    """
    Memory-aware dynamic batch size calculator.

    Monitors available system/GPU memory and adjusts batch sizes
    to avoid out-of-memory errors while maximizing throughput.
    Supports GPU memory tracking via PyTorch CUDA.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self._current_batch_size = config.max_batch_size
        self._memory_history: deque = deque(maxlen=100)
        self._oom_count = 0

    def compute_batch_size(self) -> int:
        """
        Compute the optimal batch size based on available memory.

        Returns:
            Recommended batch size.
        """
        available_mb = self._get_available_memory_mb()
        self._memory_history.append(available_mb)

        # Calculate max batch that fits in memory
        if self.config.per_sample_memory_mb > 0:
            memory_limited = int(available_mb / self.config.per_sample_memory_mb)
        else:
            memory_limited = self.config.max_batch_size

        # Apply safety margin (80% of available)
        safe_batch = int(memory_limited * 0.8)

        # Clamp to configured range
        batch_size = max(
            self.config.min_batch_size,
            min(safe_batch, self.config.max_batch_size)
        )

        # Reduce if we've had OOM events recently
        if self._oom_count > 0:
            batch_size = max(self.config.min_batch_size, batch_size // 2)

        self._current_batch_size = batch_size
        return batch_size

    def report_oom(self) -> None:
        """Report an out-of-memory event to trigger batch size reduction."""
        self._oom_count += 1
        self._current_batch_size = max(
            self.config.min_batch_size,
            self._current_batch_size // 2,
        )
        print(f"[BatchSizer] OOM detected, reduced batch size to {self._current_batch_size}")

    def report_success(self) -> None:
        """Report a successful batch to gradually increase batch size."""
        if self._oom_count > 0:
            self._oom_count = max(0, self._oom_count - 1)
        # Gradually increase batch size
        if self._current_batch_size < self.config.max_batch_size:
            self._current_batch_size = min(
                self._current_batch_size + 1,
                self.config.max_batch_size,
            )

    def _get_available_memory_mb(self) -> float:
        """Get available GPU or CPU memory in MB."""
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free / (1024 * 1024)
        except ImportError:
            pass

        # Fallback: estimate from system memory
        try:
            import psutil
            return psutil.virtual_memory().available / (1024 * 1024)
        except ImportError:
            return self.config.max_memory_mb

    @property
    def current_batch_size(self) -> int:
        """Current recommended batch size."""
        return self._current_batch_size


class ThroughputOptimizer:
    """
    Adaptive throughput optimizer for batch inference.

    Adjusts batch size and timeout parameters to maximize throughput
    while respecting latency constraints. Uses a feedback loop based
    on measured throughput and latency history.
    """

    def __init__(self, config: BatchConfig) -> None:
        self.config = config
        self._throughput_history: deque = deque(maxlen=200)
        self._latency_history: deque = deque(maxlen=200)
        self._optimal_batch_size = config.max_batch_size
        self._adjustment_count = 0

    def record_batch(self, batch_size: int, latency_ms: float) -> None:
        """
        Record metrics from a completed batch.

        Args:
            batch_size: Number of samples in the batch.
            latency_ms: Total batch processing latency.
        """
        throughput = batch_size / max(latency_ms / 1000.0, 0.001)
        per_sample_ms = latency_ms / max(batch_size, 1)

        self._throughput_history.append(throughput)
        self._latency_history.append(per_sample_ms)

    def get_optimal_batch_size(self) -> int:
        """
        Compute the optimal batch size based on throughput history.

        Returns:
            Recommended batch size.
        """
        if len(self._throughput_history) < 10:
            return self._optimal_batch_size

        # Analyze throughput by batch size
        throughput_by_size: Dict[int, List[float]] = {}
        for i, tp in enumerate(self._throughput_history):
            if i >= len(self._latency_history):
                break
            # Estimate batch size from recent history
            idx = len(self._throughput_history) - len(self._throughput_history) + i
            # Group by rounded throughput ranges
            size_bucket = max(1, round(tp / 20)) * 4
            size_bucket = min(size_bucket, self.config.max_batch_size)
            throughput_by_size.setdefault(size_bucket, []).append(tp)

        # Find batch size with best throughput
        best_size = self.config.max_batch_size
        best_throughput = 0.0
        for size, throughputs in throughput_by_size.items():
            avg_tp = np.mean(throughputs)
            if avg_tp > best_throughput:
                best_throughput = avg_tp
                best_size = size

        # Check if latency is within bounds
        recent_latency = list(self._latency_history)[-20:]
        if recent_latency:
            avg_latency = np.mean(recent_latency)
            # If latency is too high, reduce batch size
            max_per_sample_ms = self.config.batch_timeout_ms
            if avg_latency > max_per_sample_ms:
                best_size = max(self.config.min_batch_size, best_size // 2)

        self._optimal_batch_size = max(
            self.config.min_batch_size,
            min(best_size, self.config.max_batch_size)
        )

        return self._optimal_batch_size

    @property
    def avg_throughput(self) -> float:
        """Average throughput in samples per second."""
        if not self._throughput_history:
            return 0.0
        return float(np.mean(list(self._throughput_history)))


class BatchInferenceEngine:
    """
    Batch inference engine with dynamic batching and throughput optimization.

    Coordinates batch formation, memory-aware sizing, and multi-model
    pipeline execution for high-throughput inference in autonomous
    driving perception systems.

    Example:
        >>> engine = BatchInferenceEngine(BatchConfig())
        >>> engine.add_model("detector", model_fn)
        >>> engine.start()
        >>> request_id = engine.submit(image_data, model_name="detector")
        >>> result = engine.get_result(request_id)
        >>> engine.stop()
    """

    def __init__(self, config: BatchConfig = BatchConfig()) -> None:
        self.config = config
        self._batcher = DynamicBatcher(config)
        self._memory_sizer = MemoryAwareBatchSizer(config)
        self._throughput_optimizer = ThroughputOptimizer(config)
        self._models: Dict[str, Callable] = {}
        self._results: Dict[str, BatchRequest] = {}
        self._results_lock = threading.Lock()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._batch_count = 0
        self._stats_lock = threading.Lock()
        self._total_processed = 0
        self._total_dropped = 0

    def add_model(self, name: str, infer_fn: Callable) -> None:
        """
        Register a model with the batch engine.

        Args:
            name: Model name for routing requests.
            infer_fn: Callable that accepts a batched numpy array
                      and returns a dictionary of output arrays.
        """
        self._models[name] = infer_fn
        print(f"[BatchEngine] Registered model: {name}")

    def start(self) -> None:
        """Start the batch inference workers."""
        if self._running:
            return

        self._running = True
        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"batch_worker_{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        print(f"[BatchEngine] Started {self.config.num_workers} workers")

    def stop(self) -> None:
        """Stop the batch inference engine."""
        self._running = False
        for worker in self._workers:
            worker.join(timeout=2.0)
        self._workers.clear()
        print("[BatchEngine] Stopped")

    def submit(
        self,
        input_data: np.ndarray,
        model_name: str = "default",
        priority: int = 0,
        deadline_ms: float = 100.0,
        callback: Optional[Callable] = None,
    ) -> Optional[str]:
        """
        Submit a request for batch inference.

        Args:
            input_data: Input numpy array.
            model_name: Target model name.
            priority: Request priority (higher = more important).
            deadline_ms: Maximum acceptable latency.
            callback: Optional callback for result delivery.

        Returns:
            Request ID, or None if the queue is full.
        """
        request_id = f"batch_{self._total_processed}_{int(time.time() * 1000)}"

        request = BatchRequest(
            request_id=request_id,
            input_data=input_data,
            model_name=model_name,
            priority=priority,
            deadline_ms=deadline_ms,
            callback=callback,
        )

        if not self._batcher.add(request):
            return None

        return request_id

    def get_result(self, request_id: str, timeout: float = 0.1) -> Optional[Dict[str, np.ndarray]]:
        """
        Retrieve the result of a submitted request.

        Args:
            request_id: The request ID to look up.
            timeout: Maximum wait time in seconds.

        Returns:
            Output dictionary, or None if not yet available.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._results_lock:
                if request_id in self._results:
                    req = self._results.pop(request_id)
                    return req.result
            time.sleep(0.001)
        return None

    def _worker_loop(self) -> None:
        """Worker thread main loop for batch processing."""
        while self._running:
            # Wait for batch to be ready
            if not self._batcher.should_flush():
                time.sleep(0.001)
                continue

            batch = self._batcher.get_batch()
            if not batch:
                continue

            # Group by model for efficient processing
            model_groups: Dict[str, List[BatchRequest]] = {}
            for req in batch:
                model_groups.setdefault(req.model_name, []).append(req)

            # Process each model group
            for model_name, requests in model_groups.items():
                self._process_model_batch(model_name, requests)

            with self._stats_lock:
                self._batch_count += 1

    def _process_model_batch(self, model_name: str, requests: List[BatchRequest]) -> None:
        """Process a batch of requests for a single model."""
        infer_fn = self._models.get(model_name)
        if infer_fn is None:
            for req in requests:
                req.completed = True
            return

        # Determine batch size based on strategy
        if self.config.strategy == BatchStrategy.MEMORY_AWARE:
            batch_size = self._memory_sizer.compute_batch_size()
            requests = requests[:batch_size]
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            batch_size = self._throughput_optimizer.get_optimal_batch_size()
            requests = requests[:batch_size]

        # Stack inputs into a batch
        try:
            inputs = [req.input_data for req in requests]
            batch_input = np.stack(inputs)

            # Run inference
            start_time = time.time()
            batch_output = infer_fn(batch_input)
            latency_ms = (time.time() - start_time) * 1000

            # Split outputs back to individual requests
            if isinstance(batch_output, dict):
                for i, req in enumerate(requests):
                    individual_output = {}
                    for key, arr in batch_output.items():
                        if arr.ndim > 0 and arr.shape[0] == len(requests):
                            individual_output[key] = arr[i:i+1]
                        else:
                            individual_output[key] = arr
                    req.result = individual_output
                    req.completed = True
            else:
                for i, req in enumerate(requests):
                    req.result = {"output": batch_output[i:i+1] if batch_output.ndim > 0 else batch_output}
                    req.completed = True

            # Record metrics
            self._throughput_optimizer.record_batch(len(requests), latency_ms)
            self._memory_sizer.report_success()

            with self._stats_lock:
                self._total_processed += len(requests)

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                self._memory_sizer.report_oom()
                # Retry with smaller batch
                if len(requests) > 1:
                    half = len(requests) // 2
                    self._process_model_batch(model_name, requests[:half])
                    self._process_model_batch(model_name, requests[half:])
                    return

            for req in requests:
                req.result = None
                req.completed = True

        except Exception as e:
            print(f"[BatchEngine] Error processing batch for {model_name}: {e}")
            for req in requests:
                req.result = None
                req.completed = True

        # Store results and invoke callbacks
        with self._results_lock:
            for req in requests:
                self._results[req.request_id] = req
                if req.callback:
                    try:
                        req.callback(req.request_id, req.result)
                    except Exception:
                        pass

        # Clean up old results
        with self._results_lock:
            if len(self._results) > 500:
                expired_keys = [
                    k for k, v in self._results.items()
                    if v.elapsed_ms > 5000
                ]
                for k in expired_keys[:200]:
                    del self._results[k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get batch engine statistics."""
        return {
            "total_processed": self._total_processed,
            "total_batches": self._batch_count,
            "pending_requests": self._batcher.pending_count,
            "avg_throughput_fps": self._throughput_optimizer.avg_throughput,
            "current_batch_size": self._memory_sizer.current_batch_size,
            "models_registered": list(self._models.keys()),
        }
