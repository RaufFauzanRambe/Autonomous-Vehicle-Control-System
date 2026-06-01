"""
GPU monitoring utility for the Autonomous Vehicle Control System.

Provides real-time monitoring of NVIDIA GPU metrics including VRAM usage,
GPU utilization, temperature, power consumption, and per-process tracking.
Uses ``nvidia-smi`` under the hood and parses its output.

Usage:
    from utils.gpu_monitor import GPUMonitor

    monitor = GPUMonitor()
    stats = monitor.get_stats()
    print(f"GPU Temp: {stats[0].temperature}°C, VRAM: {stats[0].memory_used_pct:.1f}%")

    # Continuous monitoring
    monitor.start_monitoring(interval=1.0, callback=on_gpu_update)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes for GPU metrics
# ---------------------------------------------------------------------------

@dataclass
class GPUProcessInfo:
    """Information about a single GPU process."""
    pid: int
    name: str = ""
    used_memory_mb: float = 0.0
    gpu_utilization_pct: float = 0.0


@dataclass
class GPUStats:
    """Comprehensive statistics for a single GPU."""
    index: int = 0
    name: str = "Unknown"
    uuid: str = ""

    # Memory
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_used_pct: float = 0.0

    # Utilization
    gpu_utilization_pct: float = 0.0
    memory_utilization_pct: float = 0.0

    # Thermal
    temperature_c: float = 0.0
    temperature_threshold_slowdown_c: float = 0.0
    temperature_threshold_shutdown_c: float = 0.0

    # Power
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    power_used_pct: float = 0.0

    # Clock
    clock_sm_mhz: float = 0.0
    clock_memory_mhz: float = 0.0

    # Processes
    processes: List[GPUProcessInfo] = field(default_factory=list)

    # Fan
    fan_speed_pct: float = 0.0

    @property
    def is_thermal_throttling(self) -> bool:
        return self.temperature_c >= self.temperature_threshold_slowdown_c

    @property
    def is_memory_pressure(self) -> bool:
        return self.memory_used_pct > 90.0

    def summary(self) -> Dict[str, Any]:
        return {
            "gpu_index": self.index,
            "name": self.name,
            "memory_used_pct": round(self.memory_used_pct, 1),
            "gpu_utilization_pct": round(self.gpu_utilization_pct, 1),
            "temperature_c": round(self.temperature_c, 1),
            "power_draw_w": round(self.power_draw_w, 1),
            "processes": len(self.processes),
            "thermal_throttling": self.is_thermal_throttling,
            "memory_pressure": self.is_memory_pressure,
        }


# ---------------------------------------------------------------------------
# nvidia-smi command helpers
# ---------------------------------------------------------------------------

_NVIDIA_SMI_CMD = "nvidia-smi"

# Query arguments for nvidia-smi
_GPU_QUERY_ARGS = (
    "index,name,uuid,memory.total,memory.used,memory.free,"
    "utilization.gpu,utilization.memory,"
    "temperature.gpu,temperature.gpu.tlimit,"
    "power.draw,power.limit,"
    "clocks.current.sm,clocks.current.mem,"
    "fan.speed"
)

_PROCESS_QUERY_ARGS = "pid,process_name,used_memory"


def _run_nvidia_smi(*args: str, timeout: float = 5.0) -> str:
    """Execute nvidia-smi with given arguments and return stdout."""
    cmd = [_NVIDIA_SMI_CMD] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def _is_nvidia_smi_available() -> bool:
    """Check if nvidia-smi is available on the system."""
    return bool(_run_nvidia_smi("--help"))


def _parse_gpu_query(output: str) -> List[GPUStats]:
    """Parse the output of nvidia-smi --query-gpu --format=csv."""
    if not output:
        return []

    lines = output.strip().split("\n")
    if len(lines) < 2:
        return []

    # Parse header to get column mapping
    headers = [h.strip().lower() for h in lines[0].split(",")]
    stats_list: List[GPUStats] = []

    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(headers):
            continue

        row = dict(zip(headers, values))
        stats = GPUStats(
            index=int(row.get("index", 0)),
            name=row.get("name", "Unknown"),
            uuid=row.get("uuid", ""),
            memory_total_mb=_parse_float(row.get("memory.total [mib]", "0")),
            memory_used_mb=_parse_float(row.get("memory.used [mib]", "0")),
            memory_free_mb=_parse_float(row.get("memory.free [mib]", "0")),
            gpu_utilization_pct=_parse_float(row.get("utilization.gpu [%]", "0")),
            memory_utilization_pct=_parse_float(row.get("utilization.memory [%]", "0")),
            temperature_c=_parse_float(row.get("temperature.gpu [c]", "0")),
            temperature_threshold_slowdown_c=_parse_float(row.get("temperature.gpu.tlimit [c]", "0")),
            power_draw_w=_parse_float(row.get("power.draw [w]", "0")),
            power_limit_w=_parse_float(row.get("power.limit [w]", "0")),
            clock_sm_mhz=_parse_float(row.get("clocks.current.sm [mhz]", "0")),
            clock_memory_mhz=_parse_float(row.get("clocks.current.mem [mhz]", "0")),
            fan_speed_pct=_parse_float(row.get("fan.speed [%]", "0")),
        )

        if stats.memory_total_mb > 0:
            stats.memory_used_pct = (stats.memory_used_mb / stats.memory_total_mb) * 100.0
        if stats.power_limit_w > 0:
            stats.power_used_pct = (stats.power_draw_w / stats.power_limit_w) * 100.0

        stats_list.append(stats)

    return stats_list


def _parse_process_query(output: str) -> List[GPUProcessInfo]:
    """Parse nvidia-smi --query-compute-apps output."""
    if not output:
        return []

    lines = output.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = [h.strip().lower() for h in lines[0].split(",")]
    processes: List[GPUProcessInfo] = []

    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        if len(values) != len(headers):
            continue
        row = dict(zip(headers, values))
        processes.append(GPUProcessInfo(
            pid=int(row.get("pid", 0)),
            name=row.get("process_name", ""),
            used_memory_mb=_parse_float(row.get("used_memory [mib]", "0")),
        ))

    return processes


def _parse_float(value: str) -> float:
    """Parse a string to float, stripping units and handling [N/A]."""
    if not value or value.lower() in ("n/a", "[not supported]", "-"):
        return 0.0
    # Remove any trailing unit characters
    cleaned = re.sub(r"[^\d.\-eE]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# GPUMonitor
# ---------------------------------------------------------------------------

class GPUMonitor:
    """Monitor NVIDIA GPU(s) via nvidia-smi.

    Supports one-shot stats retrieval and continuous background monitoring
    with a callback.

    Args:
        gpu_index: Specific GPU index to monitor, or -1 for all GPUs.
    """

    def __init__(self, gpu_index: int = -1) -> None:
        self.gpu_index = gpu_index
        self._available = _is_nvidia_smi_available()
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._last_stats: List[GPUStats] = []

    @property
    def is_available(self) -> bool:
        """Whether nvidia-smi was found and responsive."""
        return self._available

    def get_stats(self) -> List[GPUStats]:
        """Retrieve current GPU statistics.

        Returns:
            List of :class:`GPUStats`, one per detected GPU.
        """
        if not self._available:
            return []

        # Query GPU stats
        output = _run_nvidia_smi(
            f"--query-gpu={_GPU_QUERY_ARGS}",
            "--format=csv,noheader,nounits",
        )
        stats_list = _parse_gpu_query(output)

        # Query per-GPU processes
        for stats in stats_list:
            proc_output = _run_nvidia_smi(
                f"-i {stats.index}",
                f"--query-compute-apps={_PROCESS_QUERY_ARGS}",
                "--format=csv,noheader,nounits",
            )
            stats.processes = _parse_process_query(proc_output)

        # Filter to specific GPU if requested
        if self.gpu_index >= 0:
            stats_list = [s for s in stats_list if s.index == self.gpu_index]

        self._last_stats = stats_list
        return stats_list

    def get_memory_info(self) -> Dict[int, Dict[str, float]]:
        """Quick memory-only query (faster than full stats).

        Returns:
            Dict mapping GPU index → {used_mb, total_mb, used_pct}.
        """
        output = _run_nvidia_smi(
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        )
        result: Dict[int, Dict[str, float]] = {}
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            idx = int(parts[0])
            used = _parse_float(parts[1])
            total = _parse_float(parts[2])
            result[idx] = {
                "used_mb": used,
                "total_mb": total,
                "used_pct": (used / total * 100.0) if total > 0 else 0.0,
            }
        return result

    def get_temperature(self) -> Dict[int, float]:
        """Quick temperature-only query. Returns {gpu_index: temp_c}."""
        output = _run_nvidia_smi(
            "--query-gpu=index,temperature.gpu",
            "--format=csv,noheader,nounits",
        )
        result: Dict[int, float] = {}
        for line in output.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            result[int(parts[0])] = _parse_float(parts[1])
        return result

    # -- Continuous monitoring --

    def start_monitoring(
        self,
        interval: float = 1.0,
        callback: Optional[Callable[[List[GPUStats]], None]] = None,
    ) -> None:
        """Start background monitoring.

        Args:
            interval: Seconds between polling cycles.
            callback: Called with the latest stats after each poll.
        """
        if self._monitoring:
            return
        self._monitoring = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval, callback),
            daemon=True,
            name="gpu-monitor",
        )
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _monitor_loop(
        self,
        interval: float,
        callback: Optional[Callable[[List[GPUStats]], None]],
    ) -> None:
        while self._monitoring:
            try:
                stats = self.get_stats()
                if callback is not None:
                    callback(stats)
            except Exception:
                pass  # Don't crash the monitor thread
            time.sleep(interval)

    @property
    def last_stats(self) -> List[GPUStats]:
        """Most recently queried stats (empty if never queried)."""
        return list(self._last_stats)

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring


# ---------------------------------------------------------------------------
# Convenience: global singleton
# ---------------------------------------------------------------------------

_global_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> GPUMonitor:
    """Return the process-wide default :class:`GPUMonitor`."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMonitor()
    return _global_monitor
