#!/usr/bin/env python3
# =============================================================================
# File: python/benchmark.py
# Brief: BenchmarkSuite — measure latency / throughput / FPS / GPU memory /
#        power consumption (via jetson-stats) and emit a JSON + Markdown
#        report with comparison tables across engines.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""Benchmarking harness for TensorRT inference on Jetson.

The suite measures:

* **Latency** (per-inference, with p50/p95/p99 percentiles).
* **Throughput** (FPS, accounting for async pipelining).
* **GPU memory** (peak device memory during the run).
* **Power consumption** (via ``jetson-stats`` — samples GPU power draw
  at 10 Hz during the run).
* **Temperature** (peak GPU temp during the run).

The output is a JSON file plus a Markdown table comparing multiple
engines (e.g. FP16 vs INT8, batch 1 vs batch 8).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Defer InferenceEngine import so the module can be loaded even on a
# non-Jetson host (for report rendering / unit tests).
try:
    from python.inference import InferenceEngine
    _ENGINE_AVAILABLE = True
except Exception:  # noqa: BLE001 — runtime fallback
    _ENGINE_AVAILABLE = False
    InferenceEngine = None  # type: ignore

try:
    from jtop import jtop
    _JTOP_AVAILABLE = True
except Exception:  # noqa: BLE001
    _JTOP_AVAILABLE = False
    jtop = None  # type: ignore

try:
    import pycuda.driver as cuda
    _PYCUDA_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PYCUDA_AVAILABLE = False


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BenchmarkSample:
    """One sample of latency / memory / power at a point in time."""

    iteration: int
    latency_ms: float
    gpu_mem_mb: float
    gpu_power_w: float
    gpu_temp_c: float


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report for one engine configuration."""

    engine_path: str
    config: Dict[str, Any]
    num_iterations: int
    warmup_iterations: int
    samples: List[BenchmarkSample] = field(default_factory=list)
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    throughput_fps: float = 0.0
    gpu_mem_peak_mb: float = 0.0
    gpu_power_mean_w: float = 0.0
    gpu_power_peak_w: float = 0.0
    gpu_temp_peak_c: float = 0.0
    trt_version: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Drop the per-sample list for compactness — caller can opt in.
        if d.get("samples") is not None:
            d["samples"] = len(d["samples"])
        return d


# -----------------------------------------------------------------------------
# Power / temperature monitor (background thread)
# -----------------------------------------------------------------------------
class _PowerMonitor:
    """Background sampler for jetson-stats power/thermal metrics."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Tuple[float, float, float]] = []  # (mem, power, temp)

    def start(self) -> None:
        if not _JTOP_AVAILABLE:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            with jtop() as jetson:
                while not self._stop.is_set():
                    stats = jetson.stats
                    gpu_mem = float(stats.get("mem", {}).get("GPU", 0))
                    gpu_power = float(stats.get("power", 0))
                    gpu_temp = float(stats.get("temperature", {}).get("GPU", 0))
                    self.samples.append((gpu_mem, gpu_power, gpu_temp))
                    time.sleep(self.interval_s)
        except Exception:  # noqa: BLE001
            pass

    def stop(self) -> List[Tuple[float, float, float]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.samples


# -----------------------------------------------------------------------------
# BenchmarkSuite
# -----------------------------------------------------------------------------
class BenchmarkSuite:
    """Run latency / throughput benchmarks on a TensorRT engine.

    Args:
        engine_path: Path to the ``.engine`` file to benchmark.
        input_name: Name of the input binding (auto-detected if omitted).
        input_shape: Shape of the input tensor. Defaults to the binding
            shape reported by the engine.
        batch_size: Inference batch size (must be <= ``max_batch_size``).
        warmup_iters: Number of warmup iterations (excluded from stats).
        iters: Number of measured iterations.
        async_infer: If True, pipeline inference and pre/post processing
            using async ``enqueue`` + ``sync``.
        monitor_power: If True, sample GPU power/temperature via jetson-stats.
        device_id: CUDA device index.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        input_name: Optional[str] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        batch_size: int = 1,
        warmup_iters: int = 20,
        iters: int = 200,
        async_infer: bool = True,
        monitor_power: bool = True,
        device_id: int = 0,
        dtype: np.dtype = np.float32,
    ) -> None:
        if not _ENGINE_AVAILABLE:
            raise RuntimeError(
                "InferenceEngine is not available — cannot benchmark.")
        self.engine_path = str(engine_path)
        self.batch_size = batch_size
        self.warmup_iters = warmup_iters
        self.iters = iters
        self.async_infer = async_infer
        self.monitor_power = monitor_power
        self.device_id = device_id

        self.engine = InferenceEngine(
            engine_path, max_batch_size=batch_size, device_id=device_id)

        if input_name is None:
            input_name = self.engine.input_names[0]
        self.input_name = input_name

        if input_shape is None:
            binding = self.engine._bindings[self.engine._input_idx[input_name]]
            input_shape = binding.shape
        # Override batch dim.
        input_shape = (batch_size,) + tuple(input_shape[1:])
        self.input_shape = input_shape
        self.dtype = dtype

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> BenchmarkReport:
        """Run the configured benchmark and return a :class:`BenchmarkReport`."""
        # Prepare a dummy input — random data is sufficient for latency.
        dummy = np.random.randn(*self.input_shape).astype(self.dtype)

        report = BenchmarkReport(
            engine_path=self.engine_path,
            config={
                "input_name": self.input_name,
                "input_shape": list(self.input_shape),
                "batch_size": self.batch_size,
                "warmup_iters": self.warmup_iters,
                "iters": self.iters,
                "async_infer": self.async_infer,
                "dtype": str(self.dtype),
            },
            num_iterations=self.iters,
            warmup_iterations=self.warmup_iters,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Warmup.
        for _ in range(self.warmup_iters):
            self.engine.infer_async({self.input_name: dummy})
            self.engine.sync()

        # Monitor power in the background.
        monitor = _PowerMonitor(interval_s=0.1) if self.monitor_power else None
        if monitor:
            monitor.start()

        # Reset peak memory counter.
        if _PYCUDA_AVAILABLE:
            try:
                cuda.Context.synchronize()
                cuda.Context.reset_accumulated_memory()
            except Exception:  # noqa: BLE001
                pass

        samples: List[BenchmarkSample] = []
        for i in range(self.iters):
            t0 = time.perf_counter()
            if self.async_infer:
                self.engine.infer_async({self.input_name: dummy})
                result = self.engine.sync()
            else:
                result = self.engine.infer({self.input_name: dummy})
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Sample memory/power at the end of each iteration (best effort).
            mem_mb = self._gpu_mem_used_mb()
            power_w, temp_c = self._instant_power_temp()

            samples.append(BenchmarkSample(
                iteration=i,
                latency_ms=latency_ms,
                gpu_mem_mb=mem_mb,
                gpu_power_w=power_w,
                gpu_temp_c=temp_c,
            ))

        if monitor:
            monitor_samples = monitor.stop()
        else:
            monitor_samples = []

        self._aggregate(report, samples, monitor_samples)
        report.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        return report

    # ------------------------------------------------------------------ #
    # Internal aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(
        self,
        report: BenchmarkReport,
        samples: List[BenchmarkSample],
        monitor_samples: List[Tuple[float, float, float]],
    ) -> None:
        """Compute summary statistics from raw samples."""
        latencies = np.array([s.latency_ms for s in samples], dtype=np.float64)
        report.samples = samples
        report.latency_mean_ms = float(np.mean(latencies))
        report.latency_p50_ms = float(np.percentile(latencies, 50))
        report.latency_p95_ms = float(np.percentile(latencies, 95))
        report.latency_p99_ms = float(np.percentile(latencies, 99))
        report.latency_min_ms = float(np.min(latencies))
        report.latency_max_ms = float(np.max(latencies))
        # Throughput = batches per second.
        if report.latency_mean_ms > 0:
            report.throughput_fps = (
                self.batch_size * 1000.0 / report.latency_mean_ms)

        # GPU memory peak — from samples (or PyCUDA if available).
        if samples:
            report.gpu_mem_peak_mb = float(max(s.gpu_mem_mb for s in samples))
        if _PYCUDA_AVAILABLE:
            try:
                report.gpu_mem_peak_mb = max(
                    report.gpu_mem_peak_mb,
                    cuda.Context.get_accumulated_memory() / (1 << 20))
            except Exception:  # noqa: BLE001
                pass

        # Power / temp aggregated from the monitor thread (if running).
        if monitor_samples:
            powers = [s[1] for s in monitor_samples if s[1] > 0]
            temps = [s[2] for s in monitor_samples if s[2] > 0]
            if powers:
                report.gpu_power_mean_w = float(statistics.mean(powers))
                report.gpu_power_peak_w = float(max(powers))
            if temps:
                report.gpu_temp_peak_c = float(max(temps))

        # TensorRT version.
        try:
            import tensorrt as trt
            report.trt_version = trt.__version__
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _gpu_mem_used_mb(self) -> float:
        """Return current GPU memory usage in MB (best effort)."""
        if not _PYCUDA_AVAILABLE:
            return 0.0
        try:
            free, total = cuda.mem_get_info()
            return (total - free) / (1 << 20)
        except Exception:  # noqa: BLE001
            return 0.0

    def _instant_power_temp(self) -> Tuple[float, float]:
        """Sample GPU power (W) and temperature (°C) right now."""
        if not _JTOP_AVAILABLE:
            return 0.0, 0.0
        try:
            with jtop() as jetson:
                stats = jetson.stats
                power = float(stats.get("power", 0))
                temp = float(stats.get("temperature", {}).get("GPU", 0))
                return power, temp
        except Exception:  # noqa: BLE001
            return 0.0, 0.0

    # ------------------------------------------------------------------ #
    # Report rendering
    # ------------------------------------------------------------------ #
    @staticmethod
    def render_markdown(reports: Sequence[BenchmarkReport]) -> str:
        """Render a Markdown comparison table from multiple reports."""
        if not reports:
            return "_No benchmark reports._\n"
        headers = [
            "Engine", "Precision", "Batch",
            "Latency mean (ms)", "Latency p95 (ms)",
            "Throughput (FPS)", "GPU mem peak (MB)",
            "Power mean (W)", "Temp peak (°C)",
        ]
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join(["---"] * len(headers)) + "|"]
        for r in reports:
            prec = ("INT8" if r.config.get("int8") else
                    "FP16" if r.config.get("fp16") else "FP32")
            row = [
                Path(r.engine_path).name,
                prec,
                str(r.config.get("batch_size", 1)),
                f"{r.latency_mean_ms:.2f}",
                f"{r.latency_p95_ms:.2f}",
                f"{r.throughput_fps:.1f}",
                f"{r.gpu_mem_peak_mb:.1f}",
                f"{r.gpu_power_mean_w:.2f}",
                f"{r.gpu_temp_peak_c:.1f}",
            ]
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"

    @staticmethod
    def save_report(
        report: BenchmarkReport, path: Union[str, Path]
    ) -> None:
        """Write a JSON report to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorRT benchmark suite.")
    parser.add_argument("--engine", required=True, help="Path to .engine file")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--no-async", action="store_true")
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--shape", default=None,
                        help="Comma-separated input shape, e.g. 1,3,640,640")
    parser.add_argument("--out", default="benchmark_report.json")
    parser.add_argument("--md", default=None,
                        help="Optional path to write Markdown summary")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    shape = (tuple(int(x) for x in args.shape.split(","))
              if args.shape else None)
    suite = BenchmarkSuite(
        engine_path=args.engine,
        input_shape=shape,
        batch_size=args.batch,
        warmup_iters=args.warmup,
        iters=args.iters,
        async_infer=not args.no_async,
        monitor_power=not args.no_power,
    )
    report = suite.run()
    BenchmarkSuite.save_report(report, args.out)
    print(f"\nReport saved to {args.out}")
    print(BenchmarkSuite.render_markdown([report]))
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(BenchmarkSuite.render_markdown([report]))


if __name__ == "__main__":  # pragma: no cover
    main()
