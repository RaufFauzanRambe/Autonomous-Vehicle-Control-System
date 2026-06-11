"""
Real-Time Inference Module for Autonomous Vehicle AI.

Provides low-latency inference capabilities:
- Asynchronous inference pipeline with thread pool
- Frame skipping for overload management
- Priority queue for critical vs. non-critical inference
- Adaptive scheduling based on system load
- Ring buffer for recent inference history
"""

import threading
import time
import queue
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .inference_engine import InferenceEngine, InferenceResult, PipelineConfig


class Priority(Enum):
    """Inference request priority levels."""
    CRITICAL = 0    # Safety-critical (collision avoidance, emergency brake)
    HIGH = 1        # Primary perception (detection, segmentation)
    NORMAL = 2      # Standard inference (depth, lane)
    LOW = 3         # Background tasks (logging, analytics)


@dataclass
class InferenceRequest:
    """A real-time inference request."""
    request_id: str
    inputs: Dict[str, np.ndarray]
    priority: Priority = Priority.NORMAL
    callback: Optional[Callable] = None
    deadline_ms: float = 50.0  # Maximum acceptable latency
    timestamp: float = 0.0
    dropped: bool = False

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def __lt__(self, other: "InferenceRequest") -> bool:
        """Priority queue ordering (lower priority value = higher priority)."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


@dataclass
class SchedulerConfig:
    """Configuration for the real-time inference scheduler."""
    max_workers: int = 4
    max_queue_size: int = 200
    frame_skip_threshold: float = 0.8  # Skip frames when queue is this full
    adaptive_scheduling: bool = True
    target_fps: float = 20.0
    max_latency_ms: float = 50.0
    history_size: int = 1000
    enable_priority_queue: bool = True
    deadline_aware: bool = True
    stats_window: int = 100


class FrameSkipper:
    """
    Adaptive frame skipping for overload management.

    When the inference pipeline is overloaded, frames are
    selectively skipped to maintain real-time performance.
    Skip decisions are based on queue depth, latency history,
    and frame priority.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._skip_counter = 0
        self._total_frames = 0
        self._skipped_frames = 0

    def should_skip(self, request: InferenceRequest, queue_depth: int) -> bool:
        """
        Determine if a frame should be skipped.

        Args:
            request: The inference request.
            queue_depth: Current queue depth.

        Returns:
            True if the frame should be skipped.
        """
        self._total_frames += 1

        # Never skip critical requests
        if request.priority == Priority.CRITICAL:
            return False

        # Skip if queue is overloaded
        queue_ratio = queue_depth / max(self.config.max_queue_size, 1)
        if queue_ratio > self.config.frame_skip_threshold:
            # Skip based on priority - lower priority gets skipped more
            skip_probability = {
                Priority.LOW: 0.9,
                Priority.NORMAL: 0.5,
                Priority.HIGH: 0.1,
                Priority.CRITICAL: 0.0,
            }[request.priority]

            if np.random.random() < skip_probability:
                self._skipped_frames += 1
                return True

        # Skip if past deadline
        if self.config.deadline_aware:
            elapsed = (time.time() - request.timestamp) * 1000
            if elapsed > request.deadline_ms:
                self._skipped_frames += 1
                return True

        return False

    @property
    def skip_rate(self) -> float:
        """Frame skip rate."""
        return self._skipped_frames / max(self._total_frames, 1)


class RealtimeInferenceEngine:
    """
    Real-time inference engine with priority scheduling.

    Extends the base InferenceEngine with real-time capabilities
    including priority queues, frame skipping, and adaptive scheduling.

    Example:
        >>> engine = RealtimeInferenceEngine(SchedulerConfig())
        >>> engine.start()
        >>> request_id = engine.submit(image_data, priority=Priority.HIGH)
        >>> result = engine.get_result(request_id, timeout=0.05)
        >>> engine.stop()
    """

    def __init__(
        self,
        config: SchedulerConfig = SchedulerConfig(),
        pipeline_config: PipelineConfig = PipelineConfig(),
    ) -> None:
        self.config = config
        self._engine = InferenceEngine(pipeline_config)
        self._frame_skipper = FrameSkipper(config)

        # Priority queue
        self._request_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=config.max_queue_size
        )
        self._result_store: Dict[str, InferenceResult] = {}
        self._result_lock = threading.Lock()

        # Worker threads
        self._workers: List[threading.Thread] = []
        self._running = False

        # Performance tracking
        self._latency_history: deque = deque(maxlen=config.history_size)
        self._fps_history: deque = deque(maxlen=config.history_size)
        self._last_frame_time = 0.0
        self._request_counter = 0

    def add_model(self, model_config: Any) -> None:
        """Add a model to the inference engine."""
        self._engine.add_model(model_config)

    def initialize(self) -> None:
        """Initialize the engine."""
        self._engine.initialize()

    def start(self) -> None:
        """Start the real-time inference workers."""
        if self._running:
            return

        self._running = True
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"inference_worker_{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

        print(f"[RealtimeEngine] Started {self.config.max_workers} workers")

    def stop(self) -> None:
        """Stop the real-time inference engine."""
        self._running = False

        # Signal workers to stop
        for _ in self._workers:
            try:
                self._request_queue.put_nowait(None)
            except queue.Full:
                pass

        # Wait for workers
        for worker in self._workers:
            worker.join(timeout=2.0)

        self._workers.clear()
        print("[RealtimeEngine] Stopped")

    def submit(
        self,
        inputs: Dict[str, np.ndarray],
        priority: Priority = Priority.NORMAL,
        callback: Optional[Callable] = None,
        deadline_ms: float = 50.0,
    ) -> Optional[str]:
        """
        Submit an inference request.

        Args:
            inputs: Input data dictionary.
            priority: Request priority.
            callback: Optional result callback.
            deadline_ms: Maximum acceptable latency.

        Returns:
            Request ID, or None if the request was skipped.
        """
        request_id = f"rt_{self._request_counter}_{int(time.time() * 1000)}"
        self._request_counter += 1

        request = InferenceRequest(
            request_id=request_id,
            inputs=inputs,
            priority=priority,
            callback=callback,
            deadline_ms=deadline_ms,
        )

        # Check if frame should be skipped
        queue_depth = self._request_queue.qsize()
        if self._frame_skipper.should_skip(request, queue_depth):
            return None

        try:
            self._request_queue.put_nowait(request)
            return request_id
        except queue.Full:
            return None

    def get_result(
        self,
        request_id: str,
        timeout: float = 0.1,
    ) -> Optional[Dict[str, InferenceResult]]:
        """
        Get the result of a submitted request.

        Args:
            request_id: The request ID to look up.
            timeout: Maximum time to wait for result.

        Returns:
            Inference result dictionary, or None if not available.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._result_lock:
                if request_id in self._result_store:
                    return self._result_store.pop(request_id)
            time.sleep(0.001)
        return None

    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while self._running:
            try:
                request = self._request_queue.get(timeout=0.1)
                if request is None:
                    break

                # Check if request is past deadline
                elapsed_ms = (time.time() - request.timestamp) * 1000
                if elapsed_ms > request.deadline_ms:
                    continue  # Skip expired request

                # Run inference
                results = self._engine.infer(request.inputs)

                # Store result
                with self._result_lock:
                    for model_name, result in results.items():
                        key = request.request_id
                        self._result_store[key] = results

                # Track latency
                total_latency = (time.time() - request.timestamp) * 1000
                self._latency_history.append(total_latency)

                # Track FPS
                now = time.time()
                if self._last_frame_time > 0:
                    fps = 1.0 / (now - self._last_frame_time)
                    self._fps_history.append(fps)
                self._last_frame_time = now

                # Callback
                if request.callback:
                    try:
                        request.callback(request.request_id, results)
                    except Exception as e:
                        print(f"[RealtimeEngine] Callback error: {e}")

                # Clean up old results (keep last 100)
                with self._result_lock:
                    if len(self._result_store) > 100:
                        keys = list(self._result_store.keys())
                        for k in keys[:-100]:
                            del self._result_store[k]

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[RealtimeEngine] Worker error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get real-time inference statistics."""
        latencies = list(self._latency_history)
        fps_values = list(self._fps_history)

        return {
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "avg_fps": float(np.mean(fps_values)) if fps_values else 0.0,
            "min_fps": float(np.min(fps_values)) if fps_values else 0.0,
            "skip_rate": self._frame_skipper.skip_rate,
            "queue_depth": self._request_queue.qsize(),
            "pending_results": len(self._result_store),
            "total_requests": self._request_counter,
        }
