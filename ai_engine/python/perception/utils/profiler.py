"""
Profiling utility for the Autonomous Vehicle Control System.

Provides function timing decorators, memory profiling, line profiler
integration, and flame graph data generation for performance analysis
of safety-critical perception and control code paths.

Usage:
    from utils.profiler import profile_function, MemoryProfiler, FlameGraphCollector

    @profile_function
    def detect_objects(image):
        ...

    mem_profiler = MemoryProfiler()
    mem_profiler.start()
    result = run_inference()
    mem_profiler.stop()
    mem_profiler.report()

    flame = FlameGraphCollector()
    with flame.sample("perception"):
        detect_objects(image)
    flame.save_folded("perception_folded.txt")
"""

from __future__ import annotations

import cProfile
import functools
import io
import os
import pstats
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

try:
    import tracemalloc
    _TRACEMALLOC_AVAILABLE = True
except ImportError:
    _TRACEMALLOC_AVAILABLE = False

try:
    import line_profiler as _line_profiler_mod
    _LINE_PROFILER_AVAILABLE = True
except ImportError:
    _LINE_PROFILER_AVAILABLE = False

try:
    import memory_profiler as _memory_profiler_mod
    _MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    _MEMORY_PROFILER_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Function timing decorator
# ---------------------------------------------------------------------------

@dataclass
class TimingResult:
    """Stores the result of a profiled function call."""
    function_name: str
    module_name: str
    call_count: int = 0
    total_time_s: float = 0.0
    min_time_s: float = float("inf")
    max_time_s: float = 0.0
    last_time_s: float = 0.0

    @property
    def mean_time_s(self) -> float:
        return self.total_time_s / self.call_count if self.call_count else 0.0

    @property
    def mean_time_ms(self) -> float:
        return self.mean_time_s * 1000.0

    def summary(self) -> Dict[str, Any]:
        return {
            "function": self.function_name,
            "module": self.module_name,
            "calls": self.call_count,
            "total_s": round(self.total_time_s, 6),
            "mean_ms": round(self.mean_time_ms, 3),
            "min_ms": round(self.min_time_s * 1000, 3),
            "max_ms": round(self.max_time_s * 1000, 3),
            "last_ms": round(self.last_time_s * 1000, 3),
        }


class FunctionProfiler:
    """Registry that accumulates timing data for profiled functions.

    Use the :func:`profile_function` decorator to automatically register
    functions with this profiler.
    """

    def __init__(self) -> None:
        self._results: Dict[str, TimingResult] = {}
        self._lock = threading.Lock()

    def record(self, func_name: str, module_name: str, elapsed: float) -> None:
        """Record a single timing measurement."""
        key = f"{module_name}.{func_name}"
        with self._lock:
            if key not in self._results:
                self._results[key] = TimingResult(
                    function_name=func_name, module_name=module_name
                )
            result = self._results[key]
            result.call_count += 1
            result.total_time_s += elapsed
            result.min_time_s = min(result.min_time_s, elapsed)
            result.max_time_s = max(result.max_time_s, elapsed)
            result.last_time_s = elapsed

    def get(self, func_name: str, module_name: str = "") -> Optional[TimingResult]:
        key = f"{module_name}.{func_name}" if module_name else func_name
        return self._results.get(key)

    def all_results(self) -> Dict[str, TimingResult]:
        with self._lock:
            return dict(self._results)

    def summary(self, sort_by: str = "total_s", top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return a sorted summary of all profiled functions."""
        results = list(self._results.values())
        sort_key_map = {
            "total_s": lambda r: r.total_time_s,
            "mean_ms": lambda r: r.mean_time_ms,
            "calls": lambda r: r.call_count,
            "max_ms": lambda r: r.max_time_s * 1000,
        }
        key_fn = sort_key_map.get(sort_by, lambda r: r.total_time_s)
        results.sort(key=key_fn, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return [r.summary() for r in results]

    def reset(self) -> None:
        with self._lock:
            self._results.clear()


# Global profiler instance
_global_profiler = FunctionProfiler()


def profile_function(
    func: Optional[F] = None,
    *,
    profiler: Optional[FunctionProfiler] = None,
    log_result: bool = False,
) -> Any:
    """Decorator that profiles a function's wall-clock execution time.

    Can be used with or without arguments::

        @profile_function
        def my_func():
            ...

        @profile_function(profiler=custom_profiler, log_result=True)
        def my_func():
            ...
    """
    def decorator(fn: F) -> F:
        _profiler = profiler or _global_profiler

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                _profiler.record(fn.__name__, fn.__module__, elapsed)
                if log_result:
                    import logging
                    logging.getLogger("profiler").debug(
                        "%s.%s took %.3f ms",
                        fn.__module__, fn.__name__, elapsed * 1000,
                    )

        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator


def get_global_profiler() -> FunctionProfiler:
    """Return the global :class:`FunctionProfiler` instance."""
    return _global_profiler


# ---------------------------------------------------------------------------
# cProfile wrapper
# ---------------------------------------------------------------------------

class CProfileWrapper:
    """Convenient wrapper around ``cProfile`` for profiling code blocks.

    Usage::

        with CProfileWrapper() as prof:
            run_inference()
        prof.print_stats(sort_by="cumulative")
    """

    def __init__(self) -> None:
        self._profiler = cProfile.Profile()
        self._results: Optional[pstats.Stats] = None

    def __enter__(self) -> "CProfileWrapper":
        self._profiler.enable()
        return self

    def __exit__(self, *args: Any) -> None:
        self._profiler.disable()
        stream = io.StringIO()
        self._results = pstats.Stats(self._profiler, stream=stream)

    def print_stats(self, sort_by: str = "cumulative", top_n: int = 30) -> str:
        """Print profile statistics and return as string."""
        if self._results is None:
            return "No profiling data available"
        stream = io.StringIO()
        stats = pstats.Stats(self._profiler, stream=stream)
        stats.sort_stats(sort_by)
        stats.print_stats(top_n)
        return stream.getvalue()

    def get_stats(self) -> Optional[pstats.Stats]:
        return self._results


# ---------------------------------------------------------------------------
# Memory profiler
# ---------------------------------------------------------------------------

class MemoryProfiler:
    """Profile memory usage of Python code using ``tracemalloc``.

    Usage::

        mp = MemoryProfiler()
        mp.start()
        # ... code to profile ...
        mp.stop()
        mp.report()
    """

    def __init__(self, n_frames: int = 25) -> None:
        self.n_frames = n_frames
        self._snapshot_before: Optional[Any] = None
        self._snapshot_after: Optional[Any] = None
        self._peak_mb: float = 0.0

    def start(self) -> None:
        """Start memory tracing and take a baseline snapshot."""
        if not _TRACEMALLOC_AVAILABLE:
            raise ImportError("tracemalloc is required for memory profiling")
        tracemalloc.start(self.n_frames)
        self._snapshot_before = tracemalloc.take_snapshot()

    def stop(self) -> None:
        """Stop tracing and capture the final snapshot."""
        if not _TRACEMALLOC_AVAILABLE:
            return
        self._snapshot_after = tracemalloc.take_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        self._peak_mb = peak / (1024 * 1024)
        tracemalloc.stop()

    @contextmanager
    def measure(self):
        """Context manager for profiling a code block."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def report(self, top_n: int = 20) -> Dict[str, Any]:
        """Generate a memory usage report."""
        report: Dict[str, Any] = {
            "peak_mb": round(self._peak_mb, 2),
            "top_allocations": [],
        }

        if self._snapshot_before is not None and self._snapshot_after is not None:
            stats = self._snapshot_after.compare_to(self._snapshot_before, "lineno")
            for stat in stats[:top_n]:
                report["top_allocations"].append({
                    "location": str(stat),
                    "size_kb": round(stat.size / 1024, 2),
                    "count": stat.count,
                })

        return report

    def top_allocations(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the top memory allocations by size."""
        if self._snapshot_after is None:
            return []
        stats = self._snapshot_after.statistics("lineno")
        result = []
        for stat in stats[:top_n]:
            result.append({
                "file": stat.traceback._frames[0] if stat.traceback._frames else ("?", "?"),
                "size_kb": round(stat.size / 1024, 2),
                "count": stat.count,
            })
        return result


# ---------------------------------------------------------------------------
# Line profiler integration
# ---------------------------------------------------------------------------

class LineProfilerWrapper:
    """Wrapper around the ``line_profiler`` package for line-by-line profiling.

    Usage::

        lp = LineProfilerWrapper()
        lp.add_function(detect_objects)
        lp.add_function(postprocess)
        lp.start()
        result = detect_objects(image)
        lp.stop()
        lp.print_stats()
    """

    def __init__(self) -> None:
        if not _LINE_PROFILER_AVAILABLE:
            raise ImportError(
                "line_profiler is required. Install with: pip install line_profiler"
            )
        self._profiler = _line_profiler_mod.LineProfiler()
        self._active = False

    def add_function(self, func: Callable[..., Any]) -> None:
        """Add a function to the line profiler."""
        self._profiler.add_function(func)

    def start(self) -> None:
        """Enable line profiling."""
        self._profiler.enable_by_count()
        self._active = True

    def stop(self) -> None:
        """Disable line profiling."""
        self._profiler.disable_by_count()
        self._active = False

    @contextmanager
    def session(self):
        """Context manager for a profiling session."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def print_stats(self, sort_by: str = "time") -> str:
        """Return formatted line profiler statistics."""
        stream = io.StringIO()
        self._profiler.print_stats(stream=stream, sort=sort_by)
        return stream.getvalue()

    @property
    def is_active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# Flame graph data collector
# ---------------------------------------------------------------------------

@dataclass
class _FlameNode:
    """Node in a flame graph stack tree."""
    name: str
    value: int = 0
    children: Dict[str, "_FlameNode"] = field(default_factory=dict)


class FlameGraphCollector:
    """Collect stack samples for generating flame graph visualizations.

    Produces data in the "folded stack" format consumed by
    ``FlameGraph.pl`` and similar tools.

    Usage::

        fg = FlameGraphCollector()
        with fg.sample("perception"):
            with fg.sample("detection"):
                detect_objects(image)
            with fg.sample("tracking"):
                track_objects(detections)
        fg.save_folded("flamegraph.txt")
    """

    def __init__(self) -> None:
        self._root = _FlameNode(name="root")
        self._stack_lock = threading.Lock()
        self._thread_stacks: Dict[int, List[str]] = {}

    @contextmanager
    def sample(self, label: str):
        """Context manager that adds *label* to the current call stack."""
        tid = threading.get_ident()
        with self._stack_lock:
            if tid not in self._thread_stacks:
                self._thread_stacks[tid] = []
            self._thread_stacks[tid].append(label)

        try:
            yield
        finally:
            with self._stack_lock:
                stack = self._thread_stacks.get(tid, [])
                if stack:
                    # Record the completed stack
                    self._record_stack(stack)
                    stack.pop()

    def _record_stack(self, stack: List[str]) -> None:
        """Record a completed stack sample into the tree."""
        node = self._root
        for frame in stack:
            if frame not in node.children:
                node.children[frame] = _FlameNode(name=frame)
            node = node.children[frame]
        node.value += 1

    def to_folded(self) -> str:
        """Convert the collected stacks to folded stack format.

        Each line is: ``frame1;frame2;frame3 COUNT``
        """
        lines: List[str] = []
        self._fold_node(self._root, [], lines)
        return "\n".join(lines)

    def _fold_node(self, node: _FlameNode, stack: List[str], lines: List[str]) -> None:
        if node.value > 0 and stack:
            lines.append(f"{';'.join(stack)} {node.value}")
        for child_name, child_node in node.children.items():
            self._fold_node(child_node, stack + [child_name], lines)

    def save_folded(self, path: Union[str, Path]) -> None:
        """Save folded stack data to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_folded())
            fh.write("\n")

    def summary(self) -> Dict[str, Any]:
        """Return a summary of collected samples."""
        total_samples = self._count_samples(self._root)
        unique_stacks = self._count_leaves(self._root)
        return {
            "total_samples": total_samples,
            "unique_stacks": unique_stacks,
        }

    def _count_samples(self, node: _FlameNode) -> int:
        count = node.value
        for child in node.children.values():
            count += self._count_samples(child)
        return count

    def _count_leaves(self, node: _FlameNode) -> int:
        if not node.children:
            return 1 if node.value > 0 else 0
        return sum(self._count_leaves(child) for child in node.children.values())

    def reset(self) -> None:
        """Clear all collected data."""
        self._root = _FlameNode(name="root")
        self._thread_stacks.clear()
