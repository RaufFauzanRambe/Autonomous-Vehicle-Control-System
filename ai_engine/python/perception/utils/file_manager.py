"""File management utilities for Thunders AI.

Supports reading and writing JSON, YAML, Pickle, and SafeTensors
formats with compression, decompression, and file watching.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class FileWatcher:
    """Simple file change watcher.

    Attributes:
        path: Watched file or directory path.
        callback: Function to call on change.
    """

    def __init__(
        self,
        path: str,
        callback: Callable[[str], None],
        poll_interval: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.callback = callback
        self.poll_interval = poll_interval
        self._last_mtime: Optional[float] = None
        self._running: bool = False
        self._watch_id = f"watch-{id(self)}"

    def check(self) -> bool:
        """Check if the watched path has changed.

        Returns:
            True if a change was detected and callback invoked.
        """
        try:
            current_mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return False

        if self._last_mtime is not None and current_mtime > self._last_mtime:
            self._last_mtime = current_mtime
            try:
                self.callback(str(self.path))
            except Exception as exc:
                logger.error("FileWatcher callback error: %s", exc)
            return True

        self._last_mtime = current_mtime
        return False


class FileManager:
    """File I/O with format detection, compression, and watching.

    Supports JSON, YAML, Pickle, and SafeTensors formats with
    automatic compression/decompression and file change monitoring.

    Attributes:
        base_dir: Default base directory for file operations.
    """

    SUPPORTED_FORMATS = ["json", "yaml", "yml", "pickle", "pkl", "safetensors"]

    def __init__(
        self,
        base_dir: str = ".",
        create_dirs: bool = True,
        default_format: str = "json",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.default_format = default_format
        self._watchers: Dict[str, FileWatcher] = {}

        if create_dirs:
            self.base_dir.mkdir(parents=True, exist_ok=True)

        logger.info("FileManager initialised: base_dir=%s", self.base_dir)

    def save(
        self,
        data: Any,
        path: str,
        format: Optional[str] = None,
        compress: bool = False,
        indent: int = 2,
        **kwargs: Any,
    ) -> str:
        """Save data to a file.

        Args:
            data: Data to save.
            path: File path (relative to base_dir or absolute).
            format: Force format; auto-detected from extension if None.
            compress: Gzip-compress the output.
            indent: Indentation for JSON/YAML.
            **kwargs: Format-specific options.

        Returns:
            Absolute path of the saved file.

        Raises:
            ValueError: If format is unsupported.
        """
        file_path = self._resolve_path(path)
        fmt = format or self._detect_format(file_path)

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: '{fmt}'; choose from {self.SUPPORTED_FORMATS}")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            content = json.dumps(data, indent=indent, default=str, **kwargs)
            self._write_bytes(content.encode("utf-8"), file_path, compress)
        elif fmt in ("yaml", "yml"):
            content = self._serialise_yaml(data, **kwargs)
            self._write_bytes(content.encode("utf-8"), file_path, compress)
        elif fmt in ("pickle", "pkl"):
            content = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            self._write_bytes(content, file_path, compress)
        elif fmt == "safetensors":
            self._save_safetensors(data, file_path, **kwargs)

        logger.info("Saved: %s (format=%s, compress=%s)", file_path, fmt, compress)
        return str(file_path)

    def load(
        self,
        path: str,
        format: Optional[str] = None,
        decompress: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Load data from a file.

        Args:
            path: File path.
            format: Force format; auto-detected if None.
            decompress: Gzip-decompress the input.
            **kwargs: Format-specific options.

        Returns:
            Loaded data.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If format is unsupported.
        """
        file_path = self._resolve_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        fmt = format or self._detect_format(file_path)

        if fmt not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: '{fmt}'")

        if fmt == "json":
            raw = self._read_bytes(file_path, decompress)
            return json.loads(raw.decode("utf-8"))
        elif fmt in ("yaml", "yml"):
            raw = self._read_bytes(file_path, decompress)
            return self._deserialise_yaml(raw.decode("utf-8"), **kwargs)
        elif fmt in ("pickle", "pkl"):
            raw = self._read_bytes(file_path, decompress)
            return pickle.loads(raw)
        elif fmt == "safetensors":
            return self._load_safetensors(file_path, **kwargs)

    def compress(
        self,
        source_path: str,
        dest_path: Optional[str] = None,
        level: int = 6,
    ) -> str:
        """Gzip-compress a file.

        Args:
            source_path: Path of the file to compress.
            dest_path: Output path; defaults to source_path + '.gz'.
            level: Compression level (1-9).

        Returns:
            Path of the compressed file.
        """
        src = self._resolve_path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        dst = self._resolve_path(dest_path) if dest_path else Path(f"{src}.gz")

        with open(src, "rb") as f_in:
            with gzip.open(dst, "wb", compresslevel=level) as f_out:
                shutil.copyfileobj(f_in, f_out)

        ratio = dst.stat().st_size / max(src.stat().st_size, 1)
        logger.info("Compressed %s → %s (ratio=%.2f)", src, dst, ratio)
        return str(dst)

    def decompress(
        self,
        source_path: str,
        dest_path: Optional[str] = None,
    ) -> str:
        """Gzip-decompress a file.

        Args:
            source_path: Path of the gzip file.
            dest_path: Output path; strips '.gz' if not specified.

        Returns:
            Path of the decompressed file.
        """
        src = self._resolve_path(source_path)
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        if dest_path:
            dst = self._resolve_path(dest_path)
        else:
            dst = src.with_suffix("") if src.suffix == ".gz" else src.with_name(src.stem + ".out")

        with gzip.open(src, "rb") as f_in:
            with open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        logger.info("Decompressed %s → %s", src, dst)
        return str(dst)

    def watch(
        self,
        path: str,
        callback: Callable[[str], None],
        poll_interval: float = 1.0,
    ) -> str:
        """Register a file watcher.

        Args:
            path: File or directory to watch.
            callback: Function called when the file changes.
            poll_interval: Seconds between checks.

        Returns:
            Watcher ID.
        """
        watcher = FileWatcher(path, callback, poll_interval)
        self._watchers[watcher._watch_id] = watcher
        logger.info("Watching: %s (interval=%.1fs)", path, poll_interval)
        return watcher._watch_id

    def check_watchers(self) -> List[str]:
        """Check all registered watchers for changes.

        Returns:
            List of paths that changed.
        """
        changed: List[str] = []
        for watcher in self._watchers.values():
            if watcher.check():
                changed.append(str(watcher.path))
        return changed

    def compute_hash(self, path: str, algorithm: str = "sha256") -> str:
        """Compute a file hash.

        Args:
            path: File path.
            algorithm: Hash algorithm ('sha256', 'md5', 'sha1').

        Returns:
            Hex digest string.
        """
        file_path = self._resolve_path(path)
        h = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- Internal helpers ---------------------------------------------------

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to base_dir if not absolute."""
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p

    def _detect_format(self, path: Path) -> str:
        """Detect format from file extension."""
        ext = path.suffix.lower().lstrip(".")
        if ext == "gz":
            ext = path.with_suffix("").suffix.lower().lstrip(".")
        return ext if ext in self.SUPPORTED_FORMATS else self.default_format

    def _write_bytes(self, data: bytes, path: Path, compress: bool) -> None:
        """Write bytes to a file, optionally gzipping."""
        if compress:
            with gzip.open(path, "wb") as f:
                f.write(data)
        else:
            path.write_bytes(data)

    def _read_bytes(self, path: Path, decompress: bool) -> bytes:
        """Read bytes from a file, optionally gunzipping."""
        if decompress or path.suffix == ".gz":
            with gzip.open(path, "rb") as f:
                return f.read()
        return path.read_bytes()

    def _serialise_yaml(self, data: Any, **kwargs: Any) -> str:
        """Serialise data to YAML string."""
        try:
            import yaml
            return yaml.dump(data, default_flow_style=False, **kwargs)
        except ImportError:
            raise ImportError("PyYAML required for YAML support")

    def _deserialise_yaml(self, text: str, **kwargs: Any) -> Any:
        """Deserialise YAML string."""
        try:
            import yaml
            return yaml.safe_load(text)
        except ImportError:
            raise ImportError("PyYAML required for YAML support")

    def _save_safetensors(self, data: Any, path: Path, **kwargs: Any) -> None:
        """Save data in SafeTensors format."""
        try:
            from safetensors.torch import save_file
            save_file(data, str(path))
        except ImportError:
            logger.warning("safetensors not installed; falling back to pickle")
            path.write_bytes(pickle.dumps(data))

    def _load_safetensors(self, path: Path, **kwargs: Any) -> Any:
        """Load data from SafeTensors format."""
        try:
            from safetensors.torch import load_file
            return load_file(str(path))
        except ImportError:
            logger.warning("safetensors not installed; falling back to pickle")
            return pickle.loads(path.read_bytes())
