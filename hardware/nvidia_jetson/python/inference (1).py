#!/usr/bin/env python3
# =============================================================================
# File: python/inference.py
# Brief: TensorRT InferenceEngine — async GPU inference with buffer pooling,
#        FP16/INT8 support, batched execution, CUDA stream pipelining.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""TensorRT inference engine for the Jetson autonomous vehicle stack.

This module wraps the TensorRT Python API in a thin, type-safe layer that
handles buffer allocation, async execution, and CUDA stream management.
It is intended to be the single entry point for any neural network
inference on the Jetson platform.

Typical usage:

    from python.inference import InferenceEngine
    import numpy as np

    engine = InferenceEngine("/models/yolov5s.engine", max_batch_size=8)
    inp = np.random.randn(1, 3, 640, 640).astype(np.float32)
    out = engine.infer({"images": inp})
    print(out["output0"].shape)

For maximum throughput, use the async context manager:

    async with engine.stream() as ctx:
        future = ctx.enqueue({"images": inp})
        # ... do CPU work while GPU runs ...
        out = await future
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401 — must be imported before pycuda.driver
    import pycuda.driver as cuda
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:  # pragma: no cover — runtime fallback for unit tests
    _TRT_AVAILABLE = False
    trt = None  # type: ignore
    cuda = None  # type: ignore


# -----------------------------------------------------------------------------
# Constants & type aliases
# -----------------------------------------------------------------------------
TRT_LOGGER_LEVEL = trt.Logger.INFO if _TRT_AVAILABLE else None
ArrayLike = Union[np.ndarray, "np.ndarray"]
Shape = Tuple[int, ...]


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class InferenceError(RuntimeError):
    """Raised when an inference call fails."""


class EngineLoadError(InferenceError):
    """Raised when an engine file cannot be loaded or deserialized."""


class BufferAllocationError(InferenceError):
    """Raised when GPU memory for I/O tensors cannot be allocated."""


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BindingInfo:
    """Describes a single TensorRT engine I/O binding."""

    name: str
    dtype: np.dtype
    shape: Shape
    volume: int
    is_input: bool
    host_buffer: Optional[np.ndarray] = None
    device_ptr: Optional[int] = None

    def allocate(self) -> None:
        """Allocate host (pinned) and device buffers for this binding."""
        if self.host_buffer is None:
            self.host_buffer = np.zeros(self.volume, dtype=self.dtype)
        if self.device_ptr is None:
            try:
                self.device_ptr = int(
                    cuda.mem_alloc(self.host_buffer.nbytes))
            except cuda.LogicError as exc:  # pragma: no cover
                raise BufferAllocationError(
                    f"cudaMalloc failed for {self.name}: {exc}") from exc


@dataclass
class InferenceResult:
    """Result of a single inference call."""

    outputs: Dict[str, np.ndarray]
    latency_ms: float
    batch_size: int
    timestamp: float = field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Buffer pool — avoids re-allocating pinned host memory every frame.
# -----------------------------------------------------------------------------
class _BufferPool:
    """Thread-safe pool of preallocated device + pinned host buffers."""

    def __init__(self, binding: BindingInfo, pool_size: int = 4) -> None:
        self._binding = binding
        self._pool: Deque[Tuple[np.ndarray, int]] = deque()
        self._lock = threading.Lock()
        for _ in range(pool_size):
            host = np.zeros(binding.volume, dtype=binding.dtype)
            try:
                # Use page-locked memory for fast async H2D/D2H.
                pinned = cuda.pagelocked_empty(binding.volume,
                                                dtype=binding.dtype)
                dev = int(cuda.mem_alloc(pinned.nbytes))
                self._pool.append((pinned, dev))
            except cuda.LogicError:
                # Fallback to non-pinned memory (slower but functional).
                dev = int(cuda.mem_alloc(host.nbytes))
                self._pool.append((host, dev))

    def acquire(self) -> Tuple[np.ndarray, int]:
        with self._lock:
            return self._pool.popleft() if self._pool else (
                cuda.pagelocked_empty(self._binding.volume,
                                       dtype=self._binding.dtype),
                int(cuda.mem_alloc(self._binding.host_buffer.nbytes)))

    def release(self, host: np.ndarray, dev: int) -> None:
        with self._lock:
            self._pool.append((host, dev))

    def __del__(self) -> None:
        # Free device buffers on teardown.
        while self._pool:
            _, dev = self._pool.popleft()
            try:
                cuda.mem_free(dev)
            except Exception:  # noqa: BLE001 — best effort
                pass


# -----------------------------------------------------------------------------
# InferenceEngine — the main public class.
# -----------------------------------------------------------------------------
class InferenceEngine:
    """Wraps a TensorRT engine and executes inference on a CUDA stream.

    Features:
        * FP16 / INT8 engine support (transparent — set at build time).
        * Async inference using CUDA streams.
        * Buffer pooling to avoid per-frame allocations.
        * Batched inference up to ``max_batch_size``.
        * Thread-safe — multiple threads can call :meth:`infer_async`
          concurrently.

    Args:
        engine_path: Path to a serialized ``.engine`` file.
        max_batch_size: Maximum batch size supported by the engine.
        device_id: CUDA device index (always 0 on Jetson).
        pool_size: Number of preallocated buffer sets to keep in the pool.
        enable_fp16: Hint that the engine is FP16 (only affects numpy dtype
            promotion of host buffers; the engine itself dictates precision).
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        max_batch_size: int = 1,
        device_id: int = 0,
        pool_size: int = 4,
        enable_fp16: bool = True,
    ) -> None:
        if not _TRT_AVAILABLE:
            raise InferenceError(
                "TensorRT and/or PyCUDA are not installed. "
                "Run on a Jetson device with JetPack 5.x+.")

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise EngineLoadError(f"Engine file not found: {self.engine_path}")

        self.max_batch_size = max_batch_size
        self.device_id = device_id
        self.pool_size = pool_size
        self.enable_fp16 = enable_fp16

        # CUDA context — created by pycuda.autoinit on import.
        self._ctx = cuda.Device(device_id).make_context()

        # TRT logger + runtime.
        self._logger = trt.Logger(TRT_LOGGER_LEVEL)
        self._runtime = trt.Runtime(self._logger)

        # Load engine.
        with open(self.engine_path, "rb") as f:
            engine_data = f.read()
        self._engine = self._runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise EngineLoadError(
                f"Failed to deserialize engine {self.engine_path}. "
                "This usually means a TensorRT version mismatch — rebuild "
                "the engine from ONNX with the local trtexec.")

        # Create execution context.
        # TRT 8.x uses execute_async_v2; TRT 10.x uses execute_v3.
        self._trt_version = int(trt.__version__.split(".")[0])
        self._context = self._engine.create_execution_context()

        # Inspect bindings.
        self._bindings: List[BindingInfo] = []
        self._input_idx: Dict[str, int] = {}
        self._output_idx: Dict[str, int] = {}
        self._analyze_bindings()

        # CUDA stream + buffer pools per input/output binding.
        self._stream = cuda.Stream()
        self._pools: Dict[str, _BufferPool] = {
            b.name: _BufferPool(b, pool_size=self.pool_size)
            for b in self._bindings
        }

        # Async helpers.
        self._lock = threading.RLock()
        self._engine_hash = hashlib.sha256(engine_data).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _analyze_bindings(self) -> None:
        """Inspect the engine's I/O bindings and allocate pool slots."""
        n = (
            self._engine.num_io_tensors  # TRT 10+ API
            if hasattr(self._engine, "num_io_tensors")
            else self._engine.num_bindings  # TRT 8.x API
        )
        for i in range(n):
            if hasattr(self._engine, "get_tensor_name"):
                name = self._engine.get_tensor_name(i)
                is_input = self._engine.get_tensor_mode(name) == \
                    trt.TensorIOMode.INPUT
                shape = tuple(self._engine.get_tensor_shape(name))
                trt_dtype = self._engine.get_tensor_dtype(name)
            else:  # pragma: no cover — TRT 8.x legacy path
                name = self._engine.get_binding_name(i)
                is_input = self._engine.binding_is_input(i)
                shape = tuple(self._engine.get_binding_shape(i))
                trt_dtype = self._engine.get_binding_dtype(i)

            dtype = self._trt_to_np_dtype(trt_dtype)
            # Replace dynamic dims (== -1) with 1 so we can pre-allocate.
            static_shape = tuple(d if d != -1 else 1 for d in shape)
            volume = int(np.prod(static_shape))

            binding = BindingInfo(
                name=name, dtype=dtype, shape=static_shape,
                volume=volume, is_input=is_input)
            binding.allocate()
            self._bindings.append(binding)
            (self._input_idx if is_input else self._output_idx)[name] = i

    @staticmethod
    def _trt_to_np_dtype(trt_dtype: Any) -> np.dtype:
        """Map a TensorRT DataType to a numpy dtype."""
        mapping = {
            trt.float32: np.float32,
            trt.float16: np.float16,
            trt.int32: np.int32,
            trt.int8: np.int8,
            trt.bool: np.bool_,
            trt.uint8: np.uint8,
        }
        if trt_dtype not in mapping:
            raise InferenceError(f"Unsupported TRT dtype: {trt_dtype}")
        return np.dtype(mapping[trt_dtype])

    # ------------------------------------------------------------------ #
    # Public API — synchronous inference
    # ------------------------------------------------------------------ #
    def infer(self, inputs: Dict[str, ArrayLike]) -> Dict[str, np.ndarray]:
        """Run a synchronous inference and return the output tensors.

        Args:
            inputs: Mapping of input binding name to numpy array. The
                array must be contiguous and match the binding's dtype.

        Returns:
            Mapping of output binding name to numpy array (CPU side).

        Raises:
            InferenceError: If input shapes mismatch or inference fails.
        """
        result = self.infer_async(inputs)
        return self.sync()["outputs"]

    # ------------------------------------------------------------------ #
    # Public API — async inference
    # ------------------------------------------------------------------ #
    def infer_async(self, inputs: Dict[str, ArrayLike]) -> int:
        """Enqueue an inference request on the CUDA stream.

        Args:
            inputs: Mapping of input binding name to numpy array.

        Returns:
            A unique request ID that can be passed to :meth:`sync`.

        Raises:
            InferenceError: On shape/dtype mismatch.
        """
        with self._lock:
            request_id = int(time.time_ns())
            buffers: Dict[str, Tuple[np.ndarray, int]] = {}

            # 1. Copy inputs to pinned host memory, then H2D.
            for name, arr in inputs.items():
                if name not in self._input_idx:
                    raise InferenceError(
                        f"Unknown input binding '{name}'. "
                        f"Available: {list(self._input_idx)}")
                binding = self._bindings[self._input_idx[name]]
                arr = np.ascontiguousarray(arr, dtype=binding.dtype)
                expected = binding.shape
                if arr.shape != expected:
                    # Update dynamic dims if supported.
                    self._set_dynamic_shape(name, arr.shape)
                    expected = arr.shape
                if arr.size != binding.volume:
                    raise InferenceError(
                        f"Input '{name}' size mismatch: "
                        f"got {arr.size}, expected {binding.volume}")

                host, dev = self._pools[name].acquire()
                host.reshape(-1)[:arr.size] = arr.reshape(-1)
                cuda.memcpy_htod_async(
                    int(dev), host, self._stream)
                buffers[name] = (host, dev)

            # 2. Allocate output buffers (D2H happens after enqueue).
            for name, idx in self._output_idx.items():
                binding = self._bindings[idx]
                host, dev = self._pools[name].acquire()
                buffers[name] = (host, dev)

            # 3. Build the device-pointer list in binding order.
            dev_ptrs: List[int] = []
            for binding in self._bindings:
                dev_ptrs.append(buffers[binding.name][1])

            # 4. Enqueue inference.
            try:
                if self._trt_version >= 10:
                    # TRT 10.x — set tensor addresses and call execute_v3.
                    for binding, ptr in zip(self._bindings, dev_ptrs):
                        self._context.set_tensor_address(binding.name, ptr)
                    self._context.execute_async_v3(self._stream.handle())
                else:
                    # TRT 8.x — use the bindings array form.
                    self._context.execute_async_v2(
                        dev_ptrs, self._stream.handle())
            except Exception as exc:  # noqa: BLE001
                raise InferenceError(
                    f"execute_async failed: {exc}") from exc

            # 5. Schedule D2H copies for outputs.
            for name, idx in self._output_idx.items():
                host, dev = buffers[name]
                binding = self._bindings[idx]
                cuda.memcpy_dtoh_async(
                    host, int(dev), self._stream)

            # 6. Stash the request so sync() can harvest it.
            self._pending = getattr(self, "_pending", {})
            self._pending[request_id] = (buffers,)
            return request_id

    def _set_dynamic_shape(self, name: str, shape: Shape) -> None:
        """Update a dynamic input dimension on the execution context."""
        try:
            if self._trt_version >= 10:
                self._context.set_input_shape(name, shape)
            else:
                idx = self._input_idx[name]
                self._context.set_binding_shape(idx, shape)
                # Update the cached binding volume.
                binding = self._bindings[idx]
                binding.shape = shape
                binding.volume = int(np.prod(shape))
                binding.allocate()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(
                f"Cannot set dynamic shape {shape} for {name}: {exc}") from exc

    def sync(self, timeout_s: float = 10.0) -> InferenceResult:
        """Block until the most recent async inference completes.

        Args:
            timeout_s: Maximum time to wait for the stream to finish.

        Returns:
            :class:`InferenceResult` with the output tensors.
        """
        t0 = time.perf_counter()
        self._stream.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            if not getattr(self, "_pending", None):
                raise InferenceError("No pending inference to sync.")
            # Take the most recent request.
            request_id = list(self._pending.keys())[-1]
            (buffers,) = self._pending.pop(request_id)

            outputs: Dict[str, np.ndarray] = {}
            batch_size = 1
            for name, idx in self._output_idx.items():
                binding = self._bindings[idx]
                host, dev = buffers[name]
                shape = (
                    self._context.get_tensor_shape(name)
                    if self._trt_version >= 10
                    else self._context.get_binding_shape(idx)
                )
                shape = tuple(shape)
                # Dynamic batch dim — replace -1 with the host buffer length.
                if -1 in shape:
                    inferred = host.size
                    for s in shape:
                        if s > 0:
                            inferred //= s
                    shape = (inferred,) + tuple(
                        s if s > 0 else 1 for s in shape[1:])
                outputs[name] = host.reshape(shape).copy()
                if not name.endswith("_meta"):
                    batch_size = shape[0]
                self._pools[name].release(host, dev)

            # Release input buffers back to the pool.
            for name in self._input_idx:
                if name in buffers:
                    host, dev = buffers[name]
                    self._pools[name].release(host, dev)

        return InferenceResult(
            outputs=outputs, latency_ms=latency_ms,
            batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Batched inference
    # ------------------------------------------------------------------ #
    def infer_batch(
        self,
        inputs: Dict[str, List[ArrayLike]],
        show_progress: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Run inference over a list of inputs in batches.

        Args:
            inputs: Mapping of input name to list of numpy arrays.
            show_progress: If True, print progress to stderr.

        Returns:
            Mapping of output name to concatenated numpy array (axis 0).
        """
        input_names = list(inputs.keys())
        n_samples = len(inputs[input_names[0]])
        results: Dict[str, List[np.ndarray]] = {}
        iterator = range(0, n_samples, self.max_batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="batch-infer")
            except ImportError:
                pass

        for start in iterator:
            end = min(start + self.max_batch_size, n_samples)
            batch_inputs: Dict[str, np.ndarray] = {}
            for name in input_names:
                arr_list = inputs[name][start:end]
                # Pad the last batch if needed.
                pad = self.max_batch_size - len(arr_list)
                if pad > 0:
                    pad_arr = np.zeros_like(arr_list[0])[None]
                    arr_list = arr_list + [pad_arr] * pad
                batch_inputs[name] = np.stack(arr_list, axis=0)
            self.infer_async(batch_inputs)
            out = self.sync().outputs
            for k, v in out.items():
                results.setdefault(k, []).append(v[:end - start])

        return {k: np.concatenate(v, axis=0) for k, v in results.items()}

    # ------------------------------------------------------------------ #
    # Async context manager (cooperative, not preemptive)
    # ------------------------------------------------------------------ #
    class _StreamContext:
        """Async context manager returned by :meth:`stream`."""

        def __init__(self, engine: "InferenceEngine") -> None:
            self._engine = engine
            self._loop = asyncio.new_event_loop()

        async def enqueue(
            self, inputs: Dict[str, ArrayLike]
        ) -> "asyncio.Future[InferenceResult]":
            """Enqueue an inference request as a coroutine."""
            req_id = self._engine.infer_async(inputs)
            fut: asyncio.Future = self._loop.create_future()

            def _done(*_: Any) -> None:
                try:
                    result = self._engine.sync()
                    self._loop.call_soon_threadsafe(fut.set_result, result)
                except Exception as exc:  # noqa: BLE001
                    self._loop.call_soon_threadsafe(fut.set_exception, exc)

            # Polling fallback (PyCUDA does not expose stream callbacks).
            threading.Timer(0.001, _done).start()
            return fut

        async def __aenter__(self) -> "InferenceEngine._StreamContext":
            return self

        async def __aexit__(self, *exc_info: Any) -> None:
            self._engine._stream.synchronize()

    def stream(self) -> "InferenceEngine._StreamContext":
        """Return an async context manager for pipelined inference."""
        return InferenceEngine._StreamContext(self)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def input_names(self) -> List[str]:
        return list(self._input_idx.keys())

    @property
    def output_names(self) -> List[str]:
        return list(self._output_idx.keys())

    @property
    def engine_hash(self) -> str:
        """Short hash of the serialized engine bytes."""
        return self._engine_hash

    def __repr__(self) -> str:
        return (
            f"<InferenceEngine path={self.engine_path.name} "
            f"hash={self._engine_hash} inputs={self.input_names} "
            f"outputs={self.output_names} max_batch={self.max_batch_size}>")

    # ------------------------------------------------------------------ #
    # Teardown
    # ------------------------------------------------------------------ #
    def __del__(self) -> None:  # noqa: D401
        """Release GPU resources on object destruction."""
        try:
            for pool in getattr(self, "_pools", {}).values():
                del pool
            if hasattr(self, "_context"):
                del self._context
            if hasattr(self, "_engine"):
                del self._engine
            if hasattr(self, "_ctx"):
                self._ctx.detach()
        except Exception:  # noqa: BLE001
            pass


# -----------------------------------------------------------------------------
# CLI smoke test
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test InferenceEngine.")
    parser.add_argument("--engine", required=True, help="Path to .engine file")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    eng = InferenceEngine(args.engine, max_batch_size=1)
    print(f"Loaded: {eng}")
    in_name = eng.input_names[0]
    binding = eng._bindings[eng._input_idx[in_name]]
    dummy = np.random.randn(*binding.shape).astype(binding.dtype)

    latencies: List[float] = []
    for _ in range(args.iters):
        eng.infer_async({in_name: dummy})
        result = eng.sync()
        latencies.append(result.latency_ms)

    print(f"Latency over {args.iters} iters: "
          f"mean={np.mean(latencies):.2f} ms  "
          f"p95={np.percentile(latencies, 95):.2f} ms  "
          f"FPS={1000.0/np.mean(latencies):.1f}")
