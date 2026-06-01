"""
Benchmarking Module
====================

Performance benchmarking tools for object detection models:

- **FPS measurement**: Sustained and instantaneous frames-per-second
- **Latency statistics**: P50 / P95 / P99 percentiles
- **Memory profiling**: GPU / CPU memory consumption
- **Accuracy vs. Speed tradeoff**: mAP at various batch sizes / precisions
- **Model comparison**: Side-by-side benchmark of multiple models

Usage::

    from object_detection.benchmark import Benchmarker

    bench = Benchmarker(detector, input_size=(640, 640))
    bench.run(num_iterations=200)
    bench.print_report()
    bench.plot_latency()
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .object_detector import ObjectDetector

logger = logging.getLogger(__name__)


# ===================================================================
# Data Structures
# ===================================================================


@dataclass
class LatencyStats:
    """Statistical summary of latency measurements."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @classmethod
    def from_samples(cls, samples: List[float]) -> "LatencyStats":
        if not samples:
            return cls()
        sorted_s = sorted(samples)
        n = len(sorted_s)
        return cls(
            mean_ms=statistics.mean(sorted_s),
            median_ms=statistics.median(sorted_s),
            std_ms=statistics.stdev(sorted_s) if n > 1 else 0.0,
            min_ms=sorted_s[0],
            max_ms=sorted_s[-1],
            p90_ms=sorted_s[int(n * 0.90)] if n > 1 else sorted_s[0],
            p95_ms=sorted_s[int(n * 0.95)] if n > 1 else sorted_s[0],
            p99_ms=sorted_s[int(n * 0.99)] if n > 1 else sorted_s[0],
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


@dataclass
class MemoryStats:
    """Memory consumption snapshot."""

    cpu_rss_mb: float = 0.0
    cpu_vms_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_util_pct: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "cpu_rss_mb": round(self.cpu_rss_mb, 1),
            "cpu_vms_mb": round(self.cpu_vms_mb, 1),
            "gpu_used_mb": round(self.gpu_used_mb, 1),
            "gpu_total_mb": round(self.gpu_total_mb, 1),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one configuration."""

    model_name: str = ""
    device: str = ""
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 1
    precision: str = "fp32"
    num_iterations: int = 0
    warmup_iterations: int = 0

    # Latency breakdown
    preprocess_latency: Optional[LatencyStats] = None
    inference_latency: Optional[LatencyStats] = None
    postprocess_latency: Optional[LatencyStats] = None
    total_latency: Optional[LatencyStats] = None

    # Throughput
    avg_fps: float = 0.0
    peak_fps: float = 0.0

    # Memory
    memory_before: Optional[MemoryStats] = None
    memory_after: Optional[MemoryStats] = None

    # Accuracy (optional – requires ground truth)
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None

    # Raw samples
    _raw_total_ms: List[float] = field(default_factory=list)
    _raw_preprocess_ms: List[float] = field(default_factory=list)
    _raw_inference_ms: List[float] = field(default_factory=list)
    _raw_postprocess_ms: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_name": self.model_name,
            "device": self.device,
            "input_size": list(self.input_size),
            "batch_size": self.batch_size,
            "precision": self.precision,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "avg_fps": round(self.avg_fps, 1),
            "peak_fps": round(self.peak_fps, 1),
        }
        if self.total_latency:
            d["total_latency"] = self.total_latency.to_dict()
        if self.inference_latency:
            d["inference_latency"] = self.inference_latency.to_dict()
        if self.preprocess_latency:
            d["preprocess_latency"] = self.preprocess_latency.to_dict()
        if self.postprocess_latency:
            d["postprocess_latency"] = self.postprocess_latency.to_dict()
        if self.memory_before:
            d["memory_before"] = self.memory_before.to_dict()
        if self.memory_after:
            d["memory_after"] = self.memory_after.to_dict()
        if self.map_50 is not None:
            d["map_50"] = round(self.map_50, 4)
        if self.map_50_95 is not None:
            d["map_50_95"] = round(self.map_50_95, 4)
        return d


# ===================================================================
# Memory Profiler
# ===================================================================


def snapshot_memory(device: str = "cpu") -> MemoryStats:
    """Capture a memory usage snapshot.

    Parameters
    ----------
    device : str
        If contains ``"cuda"``, also query GPU memory.

    Returns
    -------
    MemoryStats
    """
    stats = MemoryStats()

    # CPU memory via psutil
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        stats.cpu_rss_mb = mem_info.rss / (1024 ** 2)
        stats.cpu_vms_mb = mem_info.vms / (1024 ** 2)
    except ImportError:
        pass

    # GPU memory
    if "cuda" in device.lower():
        try:
            import torch
            if torch.cuda.is_available():
                stats.gpu_used_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                stats.gpu_total_mb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
                stats.gpu_util_pct = (stats.gpu_used_mb / stats.gpu_total_mb * 100) if stats.gpu_total_mb > 0 else 0
        except ImportError:
            pass

    return stats


# ===================================================================
# Benchmarker
# ===================================================================


class Benchmarker:
    """Model performance benchmarking utility.

    Parameters
    ----------
    detector : ObjectDetector
        Instantiated detector (model must be loaded).
    input_size : tuple
        ``(H, W)`` for synthetic test images.
    device : str
        Device string (``"cpu"``, ``"cuda:0"``, etc.).
    """

    def __init__(
        self,
        detector: ObjectDetector,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu",
    ) -> None:
        self.detector = detector
        self.input_size = input_size
        self.device = device

    def run(
        self,
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        batch_size: int = 1,
        precision: str = "fp32",
        test_image: Optional[np.ndarray] = None,
        track_memory: bool = True,
    ) -> BenchmarkResult:
        """Execute the benchmark.

        Parameters
        ----------
        num_iterations : int
            Number of timed iterations.
        warmup_iterations : int
            Untimed warmup iterations.
        batch_size : int
            Batch size (1 = single-image mode).
        precision : str
            ``"fp32"``, ``"fp16"``, or ``"int8"``.
        test_image : np.ndarray, optional
            Test image (BGR uint8).  Random image used if None.
        track_memory : bool
            Record memory before and after benchmarking.

        Returns
        -------
        BenchmarkResult
        """
        # Generate test image
        if test_image is None:
            test_image = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)

        result = BenchmarkResult(
            model_name=getattr(self.detector, "__class__", type(self.detector)).__name__,
            device=self.device,
            input_size=self.input_size,
            batch_size=batch_size,
            precision=precision,
            num_iterations=num_iterations,
            warmup_iterations=warmup_iterations,
        )

        # Memory before
        if track_memory:
            result.memory_before = snapshot_memory(self.device)

        # Warmup
        logger.info("Warming up (%d iterations)...", warmup_iterations)
        for _ in range(warmup_iterations):
            self.detector.detect_image(test_image)

        # Timed run
        logger.info("Benchmarking (%d iterations)...", num_iterations)

        preprocess_samples: List[float] = []
        inference_samples: List[float] = []
        postprocess_samples: List[float] = []
        total_samples: List[float] = []

        for i in range(num_iterations):
            t0 = time.perf_counter()
            det_result = self.detector.detect_image(test_image)
            t1 = time.perf_counter()

            total_ms = (t1 - t0) * 1000
            total_samples.append(total_ms)
            preprocess_samples.append(det_result.preprocess_time_ms)
            inference_samples.append(det_result.inference_time_ms)
            postprocess_samples.append(det_result.postprocess_time_ms)

            if (i + 1) % 20 == 0:
                running_fps = 1000.0 / statistics.mean(total_samples[-20:])
                logger.info("  [%d/%d] running FPS: %.1f", i + 1, num_iterations, running_fps)

        # Compute stats
        result.total_latency = LatencyStats.from_samples(total_samples)
        result.preprocess_latency = LatencyStats.from_samples(preprocess_samples)
        result.inference_latency = LatencyStats.from_samples(inference_samples)
        result.postprocess_latency = LatencyStats.from_samples(postprocess_samples)

        result.avg_fps = 1000.0 / result.total_latency.mean_ms if result.total_latency.mean_ms > 0 else 0
        result.peak_fps = 1000.0 / result.total_latency.min_ms if result.total_latency.min_ms > 0 else 0

        # Memory after
        if track_memory:
            result.memory_after = snapshot_memory(self.device)

        # Store raw samples
        result._raw_total_ms = total_samples
        result._raw_preprocess_ms = preprocess_samples
        result._raw_inference_ms = inference_samples
        result._raw_postprocess_ms = postprocess_samples

        logger.info("Benchmark complete – avg FPS: %.1f, median latency: %.1f ms",
                     result.avg_fps, result.total_latency.median_ms)

        return result

    def run_sweep(
        self,
        batch_sizes: List[int] = (1, 4, 8),
        precisions: List[str] = ("fp32", "fp16"),
        num_iterations: int = 50,
        warmup_iterations: int = 5,
    ) -> List[BenchmarkResult]:
        """Run benchmarks across multiple batch sizes and precisions.

        Returns
        -------
        list of BenchmarkResult
        """
        results: List[BenchmarkResult] = []
        for precision in precisions:
            for bs in batch_sizes:
                logger.info("=== Sweep: batch=%d, precision=%s ===", bs, precision)
                r = self.run(
                    num_iterations=num_iterations,
                    warmup_iterations=warmup_iterations,
                    batch_size=bs,
                    precision=precision,
                )
                results.append(r)
        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(result: BenchmarkResult) -> None:
        """Print a formatted benchmark report to the logger."""
        print("\n" + "=" * 60)
        print(f"  BENCHMARK REPORT: {result.model_name}")
        print("=" * 60)
        print(f"  Device:       {result.device}")
        print(f"  Input size:   {result.input_size}")
        print(f"  Batch size:   {result.batch_size}")
        print(f"  Precision:    {result.precision}")
        print(f"  Iterations:   {result.num_iterations} (+{result.warmup_iterations} warmup)")
        print("-" * 60)

        if result.total_latency:
            s = result.total_latency
            print(f"  Total latency:")
            print(f"    Mean:   {s.mean_ms:.2f} ms")
            print(f"    Median: {s.median_ms:.2f} ms")
            print(f"    Std:    {s.std_ms:.2f} ms")
            print(f"    Min:    {s.min_ms:.2f} ms")
            print(f"    Max:    {s.max_ms:.2f} ms")
            print(f"    P95:    {s.p95_ms:.2f} ms")
            print(f"    P99:    {s.p99_ms:.2f} ms")

        if result.inference_latency:
            print(f"  Inference latency (median): {result.inference_latency.median_ms:.2f} ms")

        print(f"  Avg FPS:  {result.avg_fps:.1f}")
        print(f"  Peak FPS: {result.peak_fps:.1f}")

        if result.memory_before and result.memory_after:
            mem_delta = result.memory_after.cpu_rss_mb - result.memory_before.cpu_rss_mb
            print(f"  CPU RSS:  {result.memory_after.cpu_rss_mb:.0f} MB (Δ {mem_delta:+.0f} MB)")
            if result.memory_after.gpu_used_mb > 0:
                print(f"  GPU Mem:  {result.memory_after.gpu_used_mb:.0f} / {result.memory_after.gpu_total_mb:.0f} MB "
                      f"({result.memory_after.gpu_util_pct:.1f}%)")

        if result.map_50 is not None:
            print(f"  mAP@50:    {result.map_50:.4f}")
        if result.map_50_95 is not None:
            print(f"  mAP@50-95: {result.map_50_95:.4f}")

        print("=" * 60 + "\n")

    @staticmethod
    def save_report(result: BenchmarkResult, path: Union[str, Path]) -> None:
        """Save benchmark results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Benchmark report saved to %s", path)

    @staticmethod
    def compare_results(results: List[BenchmarkResult]) -> str:
        """Generate a comparison table from multiple BenchmarkResults.

        Returns
        -------
        str
            Markdown-formatted comparison table.
        """
        if not results:
            return "No results to compare."

        lines = [
            "| Model | Device | Precision | Batch | Median (ms) | P95 (ms) | FPS |",
            "|-------|--------|-----------|-------|-------------|----------|-----|",
        ]

        for r in results:
            med = r.total_latency.median_ms if r.total_latency else 0
            p95 = r.total_latency.p95_ms if r.total_latency else 0
            lines.append(
                f"| {r.model_name} | {r.device} | {r.precision} | {r.batch_size} "
                f"| {med:.1f} | {p95:.1f} | {r.avg_fps:.1f} |"
            )

        return "\n".join(lines)

    @staticmethod
    def plot_latency(
        result: BenchmarkResult,
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Plot latency distribution as a histogram.

        Requires matplotlib.  If *output_path* is given, saves the plot;
        otherwise displays it.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping plot")
            return

        samples = result._raw_total_ms
        if not samples:
            logger.warning("No raw samples available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Total latency histogram
        axes[0, 0].hist(samples, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0, 0].axvline(result.total_latency.median_ms, color="red", linestyle="--", label="median")
        axes[0, 0].set_title("Total Latency Distribution")
        axes[0, 0].set_xlabel("Latency (ms)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # Inference latency histogram
        if result._raw_inference_ms:
            axes[0, 1].hist(result._raw_inference_ms, bins=50, color="darkorange", edgecolor="black", alpha=0.7)
            axes[0, 1].set_title("Inference Latency Distribution")
            axes[0, 1].set_xlabel("Latency (ms)")

        # Time series
        axes[1, 0].plot(samples, color="steelblue", linewidth=0.5)
        axes[1, 0].set_title("Latency Over Iterations")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Latency (ms)")

        # CDF
        sorted_s = sorted(samples)
        cdf = [i / len(sorted_s) for i in range(len(sorted_s))]
        axes[1, 1].plot(sorted_s, cdf, color="green")
        axes[1, 1].set_title("Latency CDF")
        axes[1, 1].set_xlabel("Latency (ms)")
        axes[1, 1].set_ylabel("CDF")

        fig.suptitle(f"Benchmark: {result.model_name} ({result.device}, {result.precision})")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Latency plot saved to %s", output_path)
        else:
            plt.show()

        plt.close(fig)


# ===================================================================
# CLI
# ===================================================================


def main() -> None:
    """Command-line benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Object Detection Benchmark")
    parser.add_argument("--config", "-c", type=str, required=True, help="YAML config")
    parser.add_argument("--iterations", "-n", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", "-w", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--output", "-o", type=str, default="benchmark_result.json", help="Output JSON")
    parser.add_argument("--plot", action="store_true", help="Generate latency plot")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from .object_detector import ObjectDetectorFactory

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    detector = ObjectDetectorFactory.create(config.get("detector_type", "yolov8"), config)

    bench = Benchmarker(detector, device=config.get("device", "cpu"))
    result = bench.run(num_iterations=args.iterations, warmup_iterations=args.warmup)

    Benchmarker.print_report(result)
    Benchmarker.save_report(result, args.output)

    if args.plot:
        Benchmarker.plot_latency(result, args.output.replace(".json", ".png"))


if __name__ == "__main__":
    main()
