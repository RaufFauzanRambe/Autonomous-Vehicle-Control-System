"""
Model Loader Module for Autonomous Vehicle AI.

Handles intelligent model loading with:
- Auto-detection of model format (PyTorch .pt/.pth, ONNX .onnx, TensorRT .engine)
- Lazy loading with deferred weight initialization
- Model caching with LRU eviction policy
- Version checking and compatibility validation
- Fallback loading with format conversion chain
- Memory tracking and GPU resource management
"""

import os
import time
import hashlib
import threading
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .inference_engine import ModelConfig


class ModelFormat(Enum):
    """Supported model file formats."""
    PYTORCH = "pytorch"       # .pt, .pth
    ONNX = "onnx"             # .onnx
    TENSORRT = "tensorrt"     # .engine, .plan
    OPENVINO = "openvino"     # .xml + .bin
    TORCHSCRIPT = "torchscript"  # .pt (scripted/traced)
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Metadata extracted from a model file."""
    format: ModelFormat = ModelFormat.UNKNOWN
    file_path: str = ""
    file_size_mb: float = 0.0
    hash_md5: str = ""
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    output_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    framework_version: str = ""
    opset_version: int = 0
    num_parameters: int = 0
    load_time_s: float = 0.0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheEntry:
    """Entry in the model cache."""
    model: Any
    metadata: ModelMetadata
    last_accessed: float = 0.0
    access_count: int = 0
    memory_mb: float = 0.0


class FormatDetector:
    """
    Auto-detects model file format from extension and content.

    Uses file extension as the primary signal, with content-based
    validation via magic bytes and header inspection.
    """

    # File extension to format mapping
    EXTENSION_MAP = {
        ".pt": ModelFormat.PYTORCH,
        ".pth": ModelFormat.PYTORCH,
        ".onnx": ModelFormat.ONNX,
        ".engine": ModelFormat.TENSORRT,
        ".plan": ModelFormat.TENSORRT,
        ".xml": ModelFormat.OPENVINO,
    }

    # Magic bytes for format identification
    MAGIC_BYTES = {
        b"\x08\x07": ModelFormat.ONNX,         # ONNX protobuf header
        b"PK": ModelFormat.PYTORCH,             # ZIP archive (PyTorch)
        b"\x89HDF": ModelFormat.PYTORCH,        # HDF5 format
    }

    @classmethod
    def detect(cls, file_path: str) -> ModelFormat:
        """
        Detect model format from file path.

        Args:
            file_path: Path to the model file.

        Returns:
            Detected model format.
        """
        if not os.path.exists(file_path):
            return ModelFormat.UNKNOWN

        # Extension-based detection
        _, ext = os.path.splitext(file_path.lower())
        if ext in cls.EXTENSION_MAP:
            format_candidate = cls.EXTENSION_MAP[ext]

            # Validate with content check
            if cls._validate_content(file_path, format_candidate):
                return format_candidate

        # Content-based detection as fallback
        return cls._detect_by_content(file_path)

    @classmethod
    def _validate_content(cls, file_path: str, expected: ModelFormat) -> bool:
        """Validate file content matches expected format."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            if expected == ModelFormat.ONNX:
                # ONNX files are protobuf, check for valid header
                return len(header) > 0
            elif expected == ModelFormat.PYTORCH:
                # PyTorch files are often ZIP archives
                return header[:2] == b"PK" or len(header) > 0
            elif expected == ModelFormat.TENSORRT:
                # TensorRT engine files have a specific header
                return len(header) > 0

            return True
        except Exception:
            return False

    @classmethod
    def _detect_by_content(cls, file_path: str) -> ModelFormat:
        """Detect format by reading file header."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            for magic, fmt in cls.MAGIC_BYTES.items():
                if header[:len(magic)] == magic:
                    return fmt

            # Check for OpenVINO XML
            if file_path.endswith(".xml"):
                with open(file_path, "r") as f:
                    content = f.read(200)
                if "<net" in content or "<Model" in content:
                    return ModelFormat.OPENVINO

        except Exception:
            pass

        return ModelFormat.UNKNOWN


class ModelCache:
    """
    LRU model cache with memory-based eviction.

    Caches loaded models in memory to avoid repeated disk I/O.
    Uses a least-recently-used eviction policy based on memory
    consumption and access patterns.
    """

    def __init__(self, max_memory_mb: float = 4096.0, max_entries: int = 10) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_memory_mb = max_memory_mb
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._current_memory_mb = 0.0

    def get(self, key: str) -> Optional[CacheEntry]:
        """
        Retrieve a model from cache.

        Args:
            key: Cache key (typically model path hash).

        Returns:
            Cache entry if found, None otherwise.
        """
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return entry
        return None

    def put(self, key: str, model: Any, metadata: ModelMetadata, memory_mb: float = 0.0) -> None:
        """
        Add a model to the cache.

        Args:
            key: Cache key.
            model: The loaded model object.
            metadata: Model metadata.
            memory_mb: Estimated memory usage in MB.
        """
        with self._lock:
            # Evict if necessary
            while (
                self._current_memory_mb + memory_mb > self._max_memory_mb
                or len(self._cache) >= self._max_entries
            ) and self._cache:
                self._evict_one()

            entry = CacheEntry(
                model=model,
                metadata=metadata,
                last_accessed=time.time(),
                access_count=1,
                memory_mb=memory_mb,
            )
            self._cache[key] = entry
            self._current_memory_mb += memory_mb

    def _evict_one(self) -> None:
        """Evict the least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            self._current_memory_mb -= entry.memory_mb
            # Attempt to free GPU memory
            self._free_model_memory(entry.model)

    @staticmethod
    def _free_model_memory(model: Any) -> None:
        """Try to free model memory."""
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except ImportError:
            del model

    def invalidate(self, key: str) -> None:
        """Remove a specific entry from the cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache.pop(key)
                self._current_memory_mb -= entry.memory_mb

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            for entry in self._cache.values():
                self._free_model_memory(entry.model)
            self._cache.clear()
            self._current_memory_mb = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "num_entries": len(self._cache),
            "max_entries": self._max_entries,
            "current_memory_mb": self._current_memory_mb,
            "max_memory_mb": self._max_memory_mb,
            "memory_utilization": self._current_memory_mb / max(self._max_memory_mb, 1),
        }


class ModelLoader:
    """
    Intelligent model loader with auto-detection and caching.

    Supports loading models from multiple formats with automatic format
    detection, version checking, lazy loading, and fallback conversion
    chains when the preferred format is unavailable.

    Example:
        >>> config = ModelConfig(model_path="detector.onnx", model_format="auto")
        >>> loader = ModelLoader(config)
        >>> model = loader.load()
        >>> result = loader.infer(np.random.randn(1, 3, 224, 224).astype(np.float32))
    """

    # Format fallback chain: try conversion in this order
    FORMAT_FALLBACK = {
        ModelFormat.PYTORCH: [ModelFormat.ONNX, ModelFormat.TENSORRT],
        ModelFormat.ONNX: [ModelFormat.TENSORRT],
        ModelFormat.TENSORRT: [],
        ModelFormat.OPENVINO: [ModelFormat.ONNX],
    }

    def __init__(
        self,
        config: ModelConfig,
        cache: Optional[ModelCache] = None,
        lazy: bool = False,
    ) -> None:
        self.config = config
        self._cache = cache or ModelCache()
        self._lazy = lazy
        self._model: Optional[Any] = None
        self._metadata: Optional[ModelMetadata] = None
        self._loaded = False
        self._lock = threading.Lock()

    def load(self) -> Any:
        """
        Load the model based on configuration.

        Detects format if set to 'auto', checks cache, validates version,
        and falls back to alternative formats if needed.

        Returns:
            Loaded model object (format depends on detected model type).

        Raises:
            FileNotFoundError: If no model file is found.
            RuntimeError: If model loading fails after all fallbacks.
        """
        if self._loaded and self._model is not None:
            return self._model

        model_path = self.config.model_path
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Check cache first
        cache_key = self._compute_cache_key(model_path)
        cached = self._cache.get(cache_key)
        if cached is not None:
            print(f"[ModelLoader] Cache hit for {model_path}")
            self._model = cached.model
            self._metadata = cached.metadata
            self._loaded = True
            return self._model

        # Detect format
        if self.config.model_format == "auto":
            model_format = FormatDetector.detect(model_path)
            print(f"[ModelLoader] Auto-detected format: {model_format.value}")
        else:
            format_map = {
                "pytorch": ModelFormat.PYTORCH,
                "onnx": ModelFormat.ONNX,
                "tensorrt": ModelFormat.TENSORRT,
                "openvino": ModelFormat.OPENVINO,
                "torchscript": ModelFormat.TORCHSCRIPT,
            }
            model_format = format_map.get(self.config.model_format, ModelFormat.UNKNOWN)

        # Load model with fallback chain
        model, metadata = self._load_with_fallback(model_path, model_format)

        if model is None:
            raise RuntimeError(f"Failed to load model: {model_path}")

        # Cache the loaded model
        memory_mb = metadata.file_size_mb  # Estimate from file size
        self._cache.put(cache_key, model, metadata, memory_mb)

        self._model = model
        self._metadata = metadata
        self._loaded = True

        print(
            f"[ModelLoader] Loaded {model_format.value} model: "
            f"{model_path} ({metadata.file_size_mb:.1f} MB, "
            f"{metadata.load_time_s:.2f}s)"
        )

        return self._model

    def _load_with_fallback(
        self, model_path: str, primary_format: ModelFormat
    ) -> Tuple[Optional[Any], ModelMetadata]:
        """Try loading with fallback format chain."""
        # Try primary format first
        model, metadata = self._load_format(model_path, primary_format)
        if model is not None:
            return model, metadata

        # Try fallback formats
        for fallback_format in self.FORMAT_FALLBACK.get(primary_format, []):
            fallback_path = self._find_fallback_file(model_path, fallback_format)
            if fallback_path:
                print(
                    f"[ModelLoader] Trying fallback: {fallback_format.value} "
                    f"at {fallback_path}"
                )
                model, metadata = self._load_format(fallback_path, fallback_format)
                if model is not None:
                    return model, metadata

        return None, ModelMetadata()

    def _load_format(
        self, model_path: str, model_format: ModelFormat
    ) -> Tuple[Optional[Any], ModelMetadata]:
        """Load a model of a specific format."""
        metadata = self._extract_metadata(model_path, model_format)
        start_time = time.time()

        try:
            if model_format == ModelFormat.PYTORCH:
                model = self._load_pytorch(model_path)
            elif model_format == ModelFormat.ONNX:
                model = self._load_onnx(model_path)
            elif model_format == ModelFormat.TENSORRT:
                model = self._load_tensorrt(model_path)
            elif model_format == ModelFormat.OPENVINO:
                model = self._load_openvino(model_path)
            elif model_format == ModelFormat.TORCHSCRIPT:
                model = self._load_torchscript(model_path)
            else:
                print(f"[ModelLoader] Unsupported format: {model_format}")
                return None, metadata

            metadata.load_time_s = time.time() - start_time
            return model, metadata

        except Exception as e:
            print(f"[ModelLoader] Failed to load {model_format.value}: {e}")
            return None, metadata

    def _load_pytorch(self, path: str) -> Any:
        """Load a PyTorch model."""
        import torch

        device = self._resolve_device()
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Check if it's a full model or state dict
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                return checkpoint["model"].to(device)
            elif "state_dict" in checkpoint:
                return checkpoint
            elif "model_state_dict" in checkpoint:
                return checkpoint["model_state_dict"]
            return checkpoint
        else:
            return checkpoint.to(device) if hasattr(checkpoint, 'to') else checkpoint

    def _load_onnx(self, path: str) -> Any:
        """Load an ONNX model as an inference session wrapper."""
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = self._get_onnx_providers()
        session = ort.InferenceSession(path, sess_options=session_options, providers=providers)

        return _ONNXModelWrapper(session)

    def _load_tensorrt(self, path: str) -> Any:
        """Load a TensorRT engine."""
        try:
            from .tensor_rt_integration import TensorRTEngine
            engine = TensorRTEngine()
            engine.load_engine(path)
            return engine
        except ImportError:
            # Fallback: try loading via ONNX Runtime with TRT provider
            import onnxruntime as ort
            session = ort.InferenceSession(
                path, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]
            )
            return _ONNXModelWrapper(session)

    def _load_openvino(self, path: str) -> Any:
        """Load an OpenVINO model."""
        from openvino.runtime import Core
        core = Core()
        model = core.read_model(model=path)
        compiled = core.compile_model(model, "AUTO")
        return _OpenVINOModelWrapper(compiled)

    def _load_torchscript(self, path: str) -> Any:
        """Load a TorchScript model."""
        import torch
        device = self._resolve_device()
        model = torch.jit.load(path, map_location=device)
        model.eval()
        return model

    def _resolve_device(self) -> str:
        """Resolve the target device string."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _get_onnx_providers(self) -> List[str]:
        """Get available ONNX Runtime providers."""
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            preferred = ["CUDAExecutionProvider", "TensorrtExecutionProvider", "CPUExecutionProvider"]
            return [p for p in preferred if p in available] + ["CPUExecutionProvider"]
        except ImportError:
            return ["CPUExecutionProvider"]

    def _find_fallback_file(self, original_path: str, target_format: ModelFormat) -> Optional[str]:
        """Find a fallback model file in the target format."""
        base_dir = os.path.dirname(original_path)
        base_name = os.path.splitext(os.path.basename(original_path))[0]

        extension_map = {
            ModelFormat.ONNX: ".onnx",
            ModelFormat.TENSORRT: ".engine",
            ModelFormat.OPENVINO: ".xml",
        }

        ext = extension_map.get(target_format)
        if ext is None:
            return None

        fallback_path = os.path.join(base_dir, base_name + ext)
        if os.path.exists(fallback_path):
            return fallback_path

        return None

    def _extract_metadata(self, path: str, model_format: ModelFormat) -> ModelMetadata:
        """Extract metadata from a model file."""
        metadata = ModelMetadata(
            format=model_format,
            file_path=path,
            file_size_mb=os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0,
        )

        # Compute hash for cache key
        metadata.hash_md5 = self._compute_file_hash(path)

        # Format-specific metadata extraction
        if model_format == ModelFormat.ONNX:
            self._extract_onnx_metadata(path, metadata)
        elif model_format == ModelFormat.PYTORCH:
            self._extract_pytorch_metadata(path, metadata)

        return metadata

    def _extract_onnx_metadata(self, path: str, metadata: ModelMetadata) -> None:
        """Extract metadata from ONNX model."""
        try:
            import onnx
            model = onnx.load(path)
            metadata.opset_version = model.opset_import[0].version if model.opset_import else 0

            for inp in model.graph.input:
                name = inp.name
                shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                metadata.input_names.append(name)
                metadata.input_shapes[name] = tuple(shape)

            for out in model.graph.output:
                name = out.name
                shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
                metadata.output_names.append(name)
                metadata.output_shapes[name] = tuple(shape)

        except Exception:
            pass

    def _extract_pytorch_metadata(self, path: str, metadata: ModelMetadata) -> None:
        """Extract metadata from PyTorch model."""
        try:
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    model = checkpoint["model"]
                    metadata.num_parameters = sum(
                        p.numel() for p in model.parameters()
                    )
                elif "epoch" in checkpoint:
                    metadata.custom_metadata["epoch"] = checkpoint.get("epoch")
                    metadata.custom_metadata["best_metric"] = checkpoint.get("best_metric")
        except Exception:
            pass

    @staticmethod
    def _compute_file_hash(path: str, chunk_size: int = 8192) -> str:
        """Compute MD5 hash of a file for version checking."""
        hasher = hashlib.md5()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""

    @staticmethod
    def _compute_cache_key(path: str) -> str:
        """Compute a cache key from the model path."""
        abs_path = os.path.abspath(path)
        return hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on the loaded model.

        Args:
            input_data: Input numpy array.

        Returns:
            Dictionary of output arrays.
        """
        if not self._loaded:
            self.load()

        if self._model is None:
            raise RuntimeError("Model not loaded")

        if hasattr(self._model, 'infer'):
            return self._model.infer(input_data)
        elif hasattr(self._model, 'run'):
            return self._model.run(input_data)
        elif callable(self._model):
            import torch
            with torch.no_grad():
                tensor = torch.from_numpy(input_data)
                result = self._model(tensor)
                if isinstance(result, torch.Tensor):
                    return {"output": result.cpu().numpy()}
                elif isinstance(result, (tuple, list)):
                    return {f"output_{i}": r.cpu().numpy() for i, r in enumerate(result)}
                return {"output": result}
        else:
            raise RuntimeError("Model does not support inference")

    def check_version(self, expected_version: Optional[str] = None) -> bool:
        """
        Check if the loaded model matches an expected version.

        Args:
            expected_version: Expected version string (MD5 hash or semantic version).

        Returns:
            True if the version matches or no version specified.
        """
        if expected_version is None or self._metadata is None:
            return True
        return self._metadata.hash_md5 == expected_version

    def unload(self) -> None:
        """Unload the model and free resources."""
        with self._lock:
            if self._metadata:
                cache_key = self._compute_cache_key(self._metadata.file_path)
                self._cache.invalidate(cache_key)
            self._model = None
            self._metadata = None
            self._loaded = False

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        """Get model metadata."""
        return self._metadata

    @property
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded and self._model is not None


class _ONNXModelWrapper:
    """Wrapper to provide a unified infer() interface for ONNX Runtime sessions."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._input_names = [inp.name for inp in session.get_inputs()]
        self._output_names = [out.name for out in session.get_outputs()]

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run ONNX Runtime inference."""
        feed = {self._input_names[0]: input_data.astype(np.float32)}
        outputs = self._session.run(self._output_names, feed)
        return dict(zip(self._output_names, outputs))


class _OpenVINOModelWrapper:
    """Wrapper to provide a unified infer() interface for OpenVINO compiled models."""

    def __init__(self, compiled_model: Any) -> None:
        self._model = compiled_model
        self._infer_request = compiled_model.create_infer_request()
        self._input_names = [inp.get_any_name() for inp in compiled_model.inputs]
        self._output_names = [out.get_any_name() for out in compiled_model.outputs]

    def infer(self, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Run OpenVINO inference."""
        input_tensor = {0: input_data.astype(np.float32)}
        self._infer_request.infer(input_tensor)
        results = {}
        for i, name in enumerate(self._output_names):
            results[name] = self._infer_request.get_output_tensor(i).data.copy()
        return results
