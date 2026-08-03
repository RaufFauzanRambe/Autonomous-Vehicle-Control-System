"""Forensic evidence collection utilities.

The :class:`ForensicTools` class bundles evidence-gathering helpers used after
an intrusion is detected: memory dumps, disk images, packet captures, CAN log
snapshots, process lists, and network state. Each collected artifact is
represented as an :class:`EvidenceItem`, hashed for integrity, and packaged
into a tarball suitable for handover to a SIEM or incident-response team.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import EvidenceType
from .utils import hash_file, timestamp_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A single piece of collected forensic evidence."""

    id: str
    type: EvidenceType
    path: str
    size_bytes: int
    sha256: str
    collected_at: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    collector: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "collected_at": self.collected_at,
            "description": self.description,
            "metadata": dict(self.metadata),
            "collector": self.collector,
        }


# ---------------------------------------------------------------------------
# ForensicTools
# ---------------------------------------------------------------------------


class ForensicTools:
    """Collect, hash, and package forensic evidence.

    All artifacts are stored under ``evidence_dir``. Each :meth:`collect_evidence`
    run creates a unique sub-directory per "case" identified by a timestamp.

    Heavy operations (memory dumps, disk images) shell out to standard Linux
    tooling; if the tool is missing the operation is logged and skipped rather
    than raising, so partial collections remain useful.
    """

    def __init__(
        self,
        evidence_dir: str = "/var/lib/avcs/ids/evidence",
        hash_algorithm: str = "sha256",
        max_size_mb: int = 4096,
        compress: bool = True,
    ) -> None:
        self.evidence_dir = str(evidence_dir)
        self.hash_algorithm = hash_algorithm
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.compress = bool(compress)
        self._lock = threading.RLock()
        self._next_id = 1
        self._manifest: List[EvidenceItem] = []
        Path(self.evidence_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Case management
    # ------------------------------------------------------------------

    def _new_case_dir(self, case_name: Optional[str] = None) -> Path:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        name = case_name or f"case-{ts}-{os.getpid()}"
        case_dir = Path(self.evidence_dir) / name
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir

    def _next_evidence_id(self) -> str:
        with self._lock:
            eid = f"ev-{self._next_id:06d}"
            self._next_id += 1
            return eid

    # ------------------------------------------------------------------
    # Individual collectors
    # ------------------------------------------------------------------

    def collect_memory_dump(self, case_dir: Path, pid: Optional[int] = None,
                            description: str = "") -> Optional[EvidenceItem]:
        """Collect a memory dump (full system via /dev/mem or a process via gcore)."""
        out_path = case_dir / ("memdump.bin" if pid is None else f"memdump.pid{pid}.bin")
        if pid is None:
            cmd = ["dd", "if=/dev/mem", f"of={out_path}", "bs=1M", "count=512"]
        else:
            cmd = ["gcore", "-o", str(case_dir / f"core.pid{pid}"), str(pid)]
        ok, error = self._run_command(cmd)
        # gcore writes core.<pid>; rename for consistency
        if pid is not None and ok:
            for cand in case_dir.glob(f"core.pid{pid}.*"):
                shutil.move(str(cand), str(out_path))
                break
        if not ok or not out_path.exists():
            logger.warning("Memory dump failed: %s", error)
            return None
        return self._finalize(out_path, EvidenceType.MEMORY_DUMP, description,
                              metadata={"pid": pid, "command": " ".join(cmd)})

    def collect_disk_image(self, case_dir: Path, device: str = "/dev/sda1",
                           description: str = "") -> Optional[EvidenceItem]:
        """Collect a disk image via dd (limited to max_size_bytes)."""
        out_path = case_dir / f"diskimage-{Path(device).name}.dd"
        bs = "1M"
        count = max(1, self.max_size_bytes // (1024 * 1024))
        cmd = ["dd", f"if={device}", f"of={out_path}", f"bs={bs}", f"count={count}"]
        ok, error = self._run_command(cmd)
        if not ok or not out_path.exists():
            logger.warning("Disk image failed: %s", error)
            return None
        return self._finalize(out_path, EvidenceType.DISK_IMAGE, description,
                              metadata={"device": device})

    def collect_packet_capture(self, case_dir: Path, interface: str = "eth0",
                               duration_sec: int = 10,
                               description: str = "") -> Optional[EvidenceItem]:
        """Collect a packet capture via tcpdump."""
        out_path = case_dir / f"capture-{interface}.pcap"
        cmd = ["tcpdump", "-i", interface, "-w", str(out_path), "-G",
               str(duration_sec), "-W", "1"]
        ok, error = self._run_command(cmd, timeout=duration_sec + 5)
        if not ok or not out_path.exists():
            logger.warning("Packet capture failed: %s", error)
            return None
        return self._finalize(out_path, EvidenceType.PACKET_CAPTURE, description,
                              metadata={"interface": interface, "duration_sec": duration_sec})

    def collect_can_log_snapshot(self, case_dir: Path, interface: str = "can0",
                                 duration_sec: int = 10,
                                 description: str = "") -> Optional[EvidenceItem]:
        """Collect a CAN frame log via candump."""
        out_path = case_dir / f"can-{interface}.log"
        cmd = ["candump", "-L", interface]
        try:
            with open(out_path, "wb") as fh:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.PIPE)
                proc.wait(timeout=duration_sec)
                proc.terminate()
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("CAN log snapshot failed: %s", exc)
            if 'proc' in locals():
                proc.terminate()
            return None
        if not out_path.exists() or out_path.stat().st_size == 0:
            return None
        return self._finalize(out_path, EvidenceType.CAN_LOG, description,
                              metadata={"interface": interface, "duration_sec": duration_sec})

    def collect_process_list(self, case_dir: Path,
                             description: str = "ps auxww snapshot") -> Optional[EvidenceItem]:
        """Collect a process-list snapshot (ps auxww)."""
        out_path = case_dir / "processes.txt"
        try:
            out = subprocess.check_output(["ps", "auxww"], timeout=10)
            out_path.write_bytes(out)
        except Exception as exc:
            logger.warning("Process list collection failed: %s", exc)
            return None
        return self._finalize(out_path, EvidenceType.PROCESS_LIST, description)

    def collect_network_state(self, case_dir: Path,
                              description: str = "ss -tulpn + ip addr + ip route") -> Optional[EvidenceItem]:
        """Collect network state (sockets, addresses, routes)."""
        out_path = case_dir / "network-state.txt"
        lines: List[str] = []
        for cmd in (["ss", "-tulpn"], ["ip", "addr"], ["ip", "route"], ["ip", "-s", "link"]):
            try:
                out = subprocess.check_output(cmd, timeout=10, stderr=subprocess.STDOUT)
                lines.append(f"$ {' '.join(cmd)}\n{out.decode('utf-8', 'replace')}\n")
            except Exception as exc:
                lines.append(f"$ {' '.join(cmd)}\n[ERROR] {exc}\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return self._finalize(out_path, EvidenceType.NETWORK_STATE, description)

    def collect_log_snapshot(self, case_dir: Path, log_paths: List[str],
                             description: str = "Log snapshot") -> Optional[EvidenceItem]:
        """Bundle a snapshot of the given log files into the case directory."""
        out_dir = case_dir / "logs"
        out_dir.mkdir(exist_ok=True)
        copied = []
        for p in log_paths:
            src = Path(p)
            if not src.exists():
                continue
            dst = out_dir / src.name
            try:
                shutil.copy2(src, dst)
                copied.append(str(dst))
            except OSError as exc:
                logger.warning("Failed to copy %s: %s", src, exc)
        if not copied:
            return None
        # Hash the bundle directory listing
        listing = "\n".join(copied).encode("utf-8")
        meta_path = out_dir / "MANIFEST.txt"
        meta_path.write_bytes(listing)
        return self._finalize(meta_path, EvidenceType.LOG_SNAPSHOT, description,
                              metadata={"files": copied})

    def collect_file(self, case_dir: Path, file_path: str,
                     description: str = "") -> Optional[EvidenceItem]:
        """Copy a single suspicious file into the evidence directory."""
        src = Path(file_path)
        if not src.exists() or not src.is_file():
            return None
        dst = case_dir / f"suspect-{src.name}"
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            logger.warning("Failed to copy %s: %s", src, exc)
            return None
        return self._finalize(dst, EvidenceType.OTHER, description or f"Copy of {file_path}",
                              metadata={"source": file_path})

    # ------------------------------------------------------------------
    # High-level orchestration
    # ------------------------------------------------------------------

    def snapshot_system(self, case_name: Optional[str] = None,
                        collect_memory: bool = True,
                        collect_disk: bool = False,
                        collect_capture: bool = True,
                        capture_interface: str = "eth0",
                        capture_duration: int = 5,
                        collect_can: bool = False,
                        can_interface: str = "can0",
                        collect_logs: Optional[List[str]] = None,
                        description: str = "Full system snapshot",
                        ) -> List[EvidenceItem]:
        """Run a battery of collectors and return the resulting evidence items."""
        case_dir = self._new_case_dir(case_name)
        items: List[EvidenceItem] = []
        if collect_memory:
            item = self.collect_memory_dump(case_dir, description="System memory dump")
            if item:
                items.append(item)
        if collect_disk:
            item = self.collect_disk_image(case_dir, description="Disk image")
            if item:
                items.append(item)
        if collect_capture:
            item = self.collect_packet_capture(
                case_dir, interface=capture_interface,
                duration_sec=capture_duration,
                description=f"Live capture on {capture_interface}",
            )
            if item:
                items.append(item)
        if collect_can:
            item = self.collect_can_log_snapshot(
                case_dir, interface=can_interface,
                duration_sec=capture_duration,
                description=f"CAN log on {can_interface}",
            )
            if item:
                items.append(item)
        # Always include process list + network state
        for collector in (self.collect_process_list, self.collect_network_state):
            item = collector(case_dir)
            if item:
                items.append(item)
        if collect_logs:
            item = self.collect_log_snapshot(case_dir, collect_logs)
            if item:
                items.append(item)
        # Write the manifest
        manifest_path = case_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({
                "case_dir": str(case_dir),
                "collected_at": timestamp_now(),
                "description": description,
                "items": [i.to_dict() for i in items],
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Snapshot complete: %d item(s) in %s", len(items), case_dir)
        return items

    def collect_evidence(self, case_name: Optional[str] = None,
                         **kwargs: Any) -> List[EvidenceItem]:
        """Alias for :meth:`snapshot_system`."""
        return self.snapshot_system(case_name=case_name, **kwargs)

    # ------------------------------------------------------------------
    # Hashing & packaging
    # ------------------------------------------------------------------

    def hash_evidence(self, items: List[EvidenceItem]) -> Dict[str, str]:
        """Re-hash a list of evidence items and return {id: sha256}."""
        result: Dict[str, str] = {}
        for item in items:
            try:
                result[item.id] = hash_file(item.path, self.hash_algorithm)
            except OSError as exc:
                logger.warning("Failed to hash %s: %s", item.path, exc)
        return result

    def package_evidence(self, case_dir: Path or str, dest: Optional[str] = None) -> str:
        """Package a case directory into a tar.gz archive."""
        case_dir = Path(case_dir)
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case dir not found: {case_dir}")
        dest = dest or f"{case_dir}.tar.gz"
        mode = "w:gz" if self.compress else "w"
        with tarfile.open(dest, mode) as tar:
            tar.add(case_dir, arcname=case_dir.name)
        logger.info("Packaged evidence to %s", dest)
        return dest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _finalize(self, path: Path, etype: EvidenceType, description: str,
                  metadata: Optional[Dict[str, Any]] = None) -> EvidenceItem:
        try:
            sha = hash_file(str(path), self.hash_algorithm)
            size = path.stat().st_size
        except OSError as exc:
            logger.error("Failed to finalize %s: %s", path, exc)
            raise
        item = EvidenceItem(
            id=self._next_evidence_id(),
            type=etype,
            path=str(path),
            size_bytes=size,
            sha256=sha,
            collected_at=timestamp_now(),
            description=description,
            metadata=metadata or {},
        )
        with self._lock:
            self._manifest.append(item)
        return item

    def _run_command(self, cmd: List[str], timeout: int = 60) -> tuple:
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
            if proc.returncode != 0:
                err = proc.stderr.decode("utf-8", "replace")[:500]
                return False, err
            return True, ""
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_manifest(self) -> List[EvidenceItem]:
        with self._lock:
            return list(self._manifest)

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "items_collected": len(self._manifest),
                "evidence_dir": self.evidence_dir,
                "by_type": {
                    t.value: sum(1 for i in self._manifest if i.type == t)
                    for t in EvidenceType
                },
            }


__all__ = ["ForensicTools", "EvidenceItem"]
