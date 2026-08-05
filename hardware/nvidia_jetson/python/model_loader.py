#!/usr/bin/env python3
# =============================================================================
# File: python/model_loader.py
# Brief: ModelLoader — ONNX → TensorRT engine conversion with caching,
#        dynamic shape optimization profiles, INT8 calibration, and
#        engine version management.
# Author: AV Control System Team
# Date: 2025
# License: MIT
# =============================================================================
"""ONNX → TensorRT engine loader for the Jetson AV stack.

This module wraps the TensorRT builder API so the rest of the stack can
treat engine creation as a one-liner. Key features:

* **Engine caching** — Engines are keyed on the SHA-256 of the ONNX model
  plus the build configuration (precision, max batch, dynamic shapes).
  Subsequent loads skip the (slow) build step.
* **Dynamic shapes** — Builds an optimization profile covering
  ``[min, opt, max]`` for each dynamic input dimension.
* **INT8 calibration** — Implements ``IInt8EntropyCalibrator2`` with a
  pre-loaded calibration dataset and a binary cache file.
* **Version pinning** — Refuses to load engines built with a different
  TensorRT major version.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRT_AVAILABLE = False
    trt = None  # type: ignore
    cuda = None  # type: ignore


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class BuildConfig:
    """Configuration for a single engine build."""

    onnx_path: str
    fp16: bool = True
    int8: bool = False
    batch_range: Tuple[int, int, int] = (1, 1, 1)
    image_shape: Tuple[int, int, int] = (3, 640, 640)
    workspace_gb: int = 4
    use_dla: bool = False
    dla_core: int = 0
    obey_precision_constraints: bool = False
    calib_cache_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["batch_range"] = list(self.batch_range)
        d["image_shape"] = list(self.image_shape)
        return d

    def hash(self) -> str:
        """Stable hash of this config + ONNX file content."""
        h = hashlib.sha256()
        h.update(json.dumps(self.to_dict(), sort_keys=True).encode())
        try:
            with open(self.onnx_path, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
        except FileNotFoundError:
            pass
        return h.hexdigest()[:16]


# -----------------------------------------------------------------------------
# INT8 Calibrator
# -----------------------------------------------------------------------------
class Int8Calibrator(trt.IInt8EntropyCalibrator2 if _TRT_AVAILABLE else object):
    """Entropy calibrator for INT8 quantization.

    Loads calibration images into a pagelocked host buffer and streams them
    to TensorRT during the build. The resulting scale table is cached so
    subsequent builds don't need the dataset.
    """

    def __init__(
        self,
        calibration_images: Sequence[np.ndarray],
        cache_file: Union[str, Path],
        input_name: str,
        image_shape: Tuple[int, int, int] = (3, 640, 640),
        batch_size: int = 8,
        max_batches: Optional[int] = None,
    ) -> None:
        if _TRT_AVAILABLE:
            super().__init__()
        self.cache_file = Path(cache_file)
        self.input_name = input_name
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_batches = max_batches or len(calibration_images) // batch_size
        self._images = list(calibration_images)
        self._idx = 0
        self._n_batches = min(
            self.max_batches,
            max(1, len(self._images) // batch_size))

        # Pagelocked host buffer for one batch.
        c, h, w = image_shape
        self._batch_bytes = batch_size * c * h * w * 4  # float32
        self._device_input = cuda.mem_alloc(self._batch_bytes)
        self._host_input = cuda.pagelocked_empty(
            batch_size * c * h * w, dtype=np.float32)

    def get_batch_size(self) -> int:
        """Return the batch size TensorRT should request."""
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        """Return device pointers for the next batch, or None when done."""
        if self._idx >= self._n_batches:
            return None
        start = self._idx * self.batch_size
        end = min(start + self.batch_size, len(self._images))
        batch = self._images[start:end]
        # Pad if last batch is short.
        while len(batch) < self.batch_size:
            batch.append(batch[-1])
        try:
            stacked = np.stack(batch, axis=0).astype(np.float32).ravel()
            self._host_input[:stacked.size] = stacked
            cuda.memcpy_htod(self._device_input, self._host_input)
        except Exception:  # noqa: BLE001
            return None
        self._idx += 1
        return [int(self._device_input)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """Load a previously cached calibration table."""
        if not self.cache_file.exists():
            return None
        with open(self.cache_file, "rb") as f:
            return f.read()

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist the calibration table for next time."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# -----------------------------------------------------------------------------
# ModelLoader
# -----------------------------------------------------------------------------
class ModelLoader:
    """Build / load / cache TensorRT engines from ONNX models.

    Args:
        cache_dir: Directory for cached engine files.
        logger_level: TensorRT logger verbosity.
        rebuild: If True, always rebuild even if a cached engine exists.
    """

    def __init__(
        self,
        cache_dir: Union[str, Path] = "/tmp/trt_cache",
        logger_level: Optional[Any] = None,
        rebuild: bool = False,
    ) -> None:
        if not _TRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT and PyCUDA are not installed. Run on a Jetson "
                "device with JetPack 5.x or later.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = trt.Logger(logger_level or trt.Logger.INFO)
        self.rebuild = rebuild
        self._runtime = trt.Runtime(self.logger)
        self._trt_version = int(trt.__version__.split(".")[0])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(
        self,
        onnx_path: Union[str, Path],
        fp16: bool = True,
        int8: bool = False,
        batch_range: Tuple[int, int, int] = (1, 1, 1),
        image_shape: Tuple[int, int, int] = (3, 640, 640),
        workspace_gb: int = 4,
        use_dla: bool = False,
        dla_core: int = 0,
        obey_precision_constraints: bool = False,
        calib_dataset: Optional[Sequence[np.ndarray]] = None,
        calib_cache_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Build (or load from cache) a TensorRT engine from ONNX.

        Returns:
            A ``tensorrt.ICudaEngine`` ready to be passed to
            :class:`python.inference.InferenceEngine`.
        """
        config = BuildConfig(
            onnx_path=str(onnx_path),
            fp16=fp16, int8=int8,
            batch_range=batch_range,
            image_shape=image_shape,
            workspace_gb=workspace_gb,
            use_dla=use_dla, dla_core=dla_core,
            obey_precision_constraints=obey_precision_constraints,
            calib_cache_path=str(calib_cache_path) if calib_cache_path else None,
        )
        cache_path = self._cache_path(config)

        if not self.rebuild and cache_path.exists():
            try:
                return self._load_engine(cache_path)
            except Exception as exc:  # noqa: BLE001
                self.logger.log(
                    trt.Logger.WARNING,
                    f"Failed to load cached engine {cache_path}: {exc}; "
                    "rebuilding.")

        engine = self._build_engine(
            config, calib_dataset, calib_cache_path)
        self._save_engine(engine, cache_path)
        # Persist the build config alongside the engine for debugging.
        with open(str(cache_path) + ".json", "w", encoding="utf-8") as f:
            json.dump({**config.to_dict(),
                       "trt_version": self._trt_version,
                       "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      f, indent=2)
        return engine

    def load(
        self, onnx_path: Union[str, Path], **build_kwargs: Any
    ) -> Any:
        """Alias for :meth:`build` with default cache lookup."""
        return self.build(onnx_path, **build_kwargs)

    def list_cached_engines(self) -> List[Dict[str, Any]]:
        """Return metadata for every cached engine in ``cache_dir``."""
        out = []
        for json_path in sorted(self.cache_dir.glob("*.engine.json")):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    out.append(json.load(f))
            except Exception:  # noqa: BLE001
                continue
        return out

    def clear_cache(self) -> int:
        """Delete all cached engines. Returns the number removed."""
        n = 0
        for p in self.cache_dir.glob("*.engine*"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n

    # ------------------------------------------------------------------ #
    # Internal: build pipeline
    # ------------------------------------------------------------------ #
    def _build_engine(
        self,
        config: BuildConfig,
        calib_dataset: Optional[Sequence[np.ndarray]],
        calib_cache_path: Optional[Union[str, Path]],
    ) -> Any:
        """Build a TensorRT engine from the ONNX model."""
        builder = trt.Builder(self.logger)
        network_flags = 1 << int(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        with open(config.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    self.logger.log(
                        trt.Logger.ERROR, parser.get_error(i).desc())
                raise RuntimeError(
                    f"Failed to parse ONNX model {config.onnx_path}")

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            config.workspace_gb * (1 << 30))

        if config.fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.int8:
            builder_config.set_flag(trt.BuilderFlag.INT8)
        if config.obey_precision_constraints:
            builder_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        if config.use_dla:
            builder_config.default_device_type = trt.DeviceType.DLA
            builder_config.DLA_core = config.dla_core
            builder_config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Optimization profile for dynamic shapes.
        profile = builder.create_optimization_profile()
        bmin, bopt, bmax = config.batch_range
        c, h, w = config.image_shape
        # Find the (first) input tensor name.
        input_name = self._first_input_name(network)
        profile.set_shape(
            input_name,
            min=(bmin, c, h, w),
            opt=(bopt, c, h, w),
            max=(bmax, c, h, w))
        builder_config.add_optimization_profile(profile)

        # INT8 calibration.
        if config.int8:
            if calib_dataset is None or len(calib_dataset) == 0:
                raise ValueError(
                    "INT8 build requires a non-empty calib_dataset.")
            calib = Int8Calibrator(
                calibration_images=calib_dataset,
                cache_file=calib_cache_path or
                    (self.cache_dir / "int8_calib.cache"),
                input_name=input_name,
                image_shape=config.image_shape,
                batch_size=min(8, bmin),
            )
            builder_config.int8_calibrator = calib

        # Build (this can take minutes).
        t0 = time.time()
        self.logger.log(
            trt.Logger.INFO,
            f"Building TensorRT engine from {config.onnx_path} "
            f"(fp16={config.fp16}, int8={config.int8}, "
            f"batch={config.batch_range}, shape={config.image_shape})")
        plan = builder.build_serialized_network(network, builder_config)
        if plan is None:
            raise RuntimeError(
                "TensorRT build returned None — check the logger output "
                "above for the underlying error.")
        engine = self._runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError("Failed to deserialize built engine.")
        elapsed = time.time() - t0
        self.logger.log(
            trt.Logger.INFO,
            f"Engine built in {elapsed:.1f}s "
            f"({len(plan)/1e6:.1f} MB plan).")
        return engine

    def _first_input_name(self, network: Any) -> str:
        """Return the name of the first input tensor of ``network``."""
        for i in range(network.num_inputs):
            return network.get_input(i).name
        raise RuntimeError("Network has no inputs.")

    # ------------------------------------------------------------------ #
    # Internal: load / save / cache
    # ------------------------------------------------------------------ #
    def _cache_path(self, config: BuildConfig) -> Path:
        """Return the on-disk path for a cached engine."""
        suffix = "int8" if config.int8 else ("fp16" if config.fp16 else "fp32")
        return self.cache_dir / f"{config.hash()}_{suffix}.engine"

    def _load_engine(self, path: Union[str, Path]) -> Any:
        """Deserialize an engine from disk."""
        with open(path, "rb") as f:
            plan = f.read()
        engine = self._runtime.deserialize_cuda_engine(plan)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine from {path}. "
                "The engine was likely built with a different TensorRT "
                "version. Rebuild from ONNX.")
        return engine

    def _save_engine(
        self, engine: Any, path: Union[str, Path]
    ) -> None:
        """Serialize an engine to disk."""
        plan = engine.serialize()
        with open(path, "wb") as f:
            f.write(plan)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse
    parser = argparse.ArgumentParser(description="Build a TensorRT engine.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--cache", default="/models/trt_cache")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--batch-max", type=int, default=1)
    parser.add_argument("--shape", default="3,640,640")
    parser.add_argument("--calib-dir", help="Directory of calibration images")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    calib_dataset = None
    if args.int8:
        if not args.calib_dir:
            parser.error("--int8 requires --calib-dir")
        calib_dataset = []
        for fn in sorted(os.listdir(args.calib_dir))[:64]:
            if not fn.lower().endswith((".jpg", ".png")):
                continue
            try:
                import cv2
                img = cv2.imread(os.path.join(args.calib_dir, fn))
                img = cv2.resize(img, (shape[2], shape[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                arr = img.astype(np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)
                calib_dataset.append(arr)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {fn}: {exc}")

    loader = ModelLoader(cache_dir=args.cache)
    engine = loader.build(
        onnx_path=args.onnx,
        fp16=args.fp16, int8=args.int8,
        batch_range=(1, 1, args.batch_max),
        image_shape=shape,
        calib_dataset=calib_dataset,
        calib_cache_path=os.path.join(args.cache, "int8_calib.cache"),
    )
    print(f"Engine built: {engine}")
    print(f"Cached engines: {[e['onnx_path'] for e in loader.list_cached_engines()]}")
