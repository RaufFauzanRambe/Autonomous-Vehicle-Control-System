"""
Edge Inference Module for Autonomous Vehicle AI.

Provides optimized inference for edge deployment scenarios:
- Model quantization (INT8/FP16) via PyTorch quantization APIs
- ONNX Runtime inference with hardware acceleration
- OpenVINO integration for Intel-based edge platforms
- Model size optimization and calibration dataset management
- Runtime adaptive precision selection based on device capability

Typical edge targets: NVIDIA Jetson (Xavier/Orin), Intel NUC,
Raspberry Pi 5, Qualcomm RB5, and automotive-grade SoCs.
"""

import os
import time
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class PrecisionMode(Enum):
    """Inference precision mode for edge deployment."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"
    MIXED = "mixed"  # FP16 + INT8 per-layer selection


class EdgeDevice(Enum):
    """Supported edge device types."""
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    JETSON_NANO = "jetson_nano"
    INTEL_NUC = "intel_nuc"
    RASPBERRY_PI5 = "rpi5"
    QUALCOMM_RB5 = "qualcomm_rb5"
    GENERIC_GPU = "generic_gpu"
    GENERIC_CPU = "generic_cpu"


@dataclass
class CalibrationConfig:
    """Configuration for INT8 calibration."""
    dataset_path: str = ""
    num_samples: int = 500
    batch_size: int = 1
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    calibration_method: str = "minmax"  # minmax, percentile, entropy
    percentile_value: float = 99.9  # For percentile method
    per_channel: bool = True
    use_mse: bool = False  # MSE-based calibration for better accuracy


@dataclass
class EdgeConfig:
    """Configuration for edge inference deployment."""
    device: EdgeDevice = EdgeDevice.GENERIC_GPU
    precision: PrecisionMode = PrecisionMode.FP16
    model_path: str = ""
    onnx_path: str = ""
    openvino_xml_path: str = ""
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    num_threads: int = 4
    gpu_memory_fraction: float = 0.7
    enable_caching: bool = True
    cache_dir: str = "/tmp/av_edge_cache"
    warmup_iterations: int = 10
    max_batch_size: int = 4
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])


@dataclass
class QuantizationResult:
    """Result from a model quantization operation."""
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    quantization_time_s: float = 0.0
    precision: PrecisionMode = PrecisionMode.FP16
    success: bool = False
    error: Optional[str] = None
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)


class CalibrationDataset:
    """
    Calibration dataset manager for INT8 quantization.

    Handles loading, preprocessing, and iteration over calibration
    data used to determine optimal quantization ranges per-tensor.
    Supports lazy loading to reduce memory footprint on edge devices.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self._samples: List[np.ndarray] = []
        self._index = 0
        self._loaded = False

    def load(self) -> None:
        """Load calibration samples from the configured dataset path."""
        if self._loaded:
            return

        self._samples = []
        dataset_path = self.config.dataset_path

        if not dataset_path or not os.path.exists(dataset_path):
            # Generate synthetic calibration data when no dataset is available
            print("[Calibration] No dataset found, generating synthetic calibration data")
            for _ in range(self.config.num_samples):
                sample = np.random.randn(*self.config.input_shape).astype(np.float32)
                # Apply ImageNet-like normalization
                sample = (sample - sample.min()) / (sample.max() - sample.min() + 1e-8)
                sample = (sample - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                self._samples.append(sample.astype(np.float32))
            self._loaded = True
            return

        # Load from directory of .npy files
        if os.path.isdir(dataset_path):
            npy_files = sorted([
                os.path.join(dataset_path, f)
                for f in os.listdir(dataset_path)
                if f.endswith((".npy", ".bin"))
            ])
            for fpath in npy_files[:self.config.num_samples]:
                try:
                    if fpath.endswith(".npy"):
                        arr = np.load(fpath)
                    else:
                        arr = np.fromfile(fpath, dtype=np.float32).reshape(
                            self.config.input_shape
                        )
                    if arr.shape == self.config.input_shape:
                        self._samples.append(arr.astype(np.float32))
                except Exception as e:
                    print(f"[Calibration] Failed to load {fpath}: {e}")

        # Load from a single .npz file
        elif dataset_path.endswith(".npz"):
            try:
                data = np.load(dataset_path, allow_pickle=True)
                keys = list(data.keys())
                for i in range(min(self.config.num_samples, len(keys))):
                    arr = data[keys[i]]
                    if arr.shape == self.config.input_shape:
                        self._samples.append(arr.astype(np.float32))
            except Exception as e:
                print(f"[Calibration] Failed to load npz: {e}")

        # Fallback to synthetic data
        if not self._samples:
            print("[Calibration] No valid samples loaded, using synthetic data")
            for _ in range(self.config.num_samples):
                self._samples.append(
                    np.random.randn(*self.config.input_shape).astype(np.float32)
                )

        self._loaded = True
        print(f"[Calibration] Loaded {len(self._samples)} calibration samples")

    def __iter__(self) -> "CalibrationDataset":
        self._index = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._index >= len(self._samples):
            raise StopIteration
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def __len__(self) -> int:
        return len(self._samples)

    def get_batch(self, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Get a batch of calibration data.

        Args:
            batch_size: Number of samples in the batch. Defaults to config batch size.

        Returns:
            Batched numpy array.
        """
        bs = batch_size or self.config.batch_size
        if not self._samples:
            self.load()
        end = min(self._index + bs, len(self._samples))
        batch = np.stack(self._samples[self._index:end])
        self._index = end
        return batch

    def reset(self) -> None:
        """Reset the iteration index."""
        self._index = 0


class PyTorchQuantizer:
    """
    PyTorch-based model quantization for edge deployment.

    Supports post-training static quantization (PTQ), post-training
    dynamic quantization, and quantization-aware training (QAT) flow
    preparation. Uses PyTorch's torch.quantization module.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._calibration_dataset = CalibrationDataset(config.calibration)

    def quantize_dynamic(self, model: Any) -> Any:
        """
        Apply post-training dynamic quantization.

        Dynamic quantization converts weight tensors to INT8 but computes
        activations in FP32 during inference. Best for RNN/Transformer models.

        Args:
            model: PyTorch model to quantize.

        Returns:
            Dynamically quantized model.
        """
        try:
            import torch
            import torch.quantization as quant

            model.eval()
            quantized_model = quant.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8,
            )
            return quantized_model

        except ImportError:
            print("[Quantizer] PyTorch not available for dynamic quantization")
            return model

    def quantize_static(
        self,
        model: Any,
        calibration_data: Optional[CalibrationDataset] = None,
    ) -> Tuple[Any, QuantizationResult]:
        """
        Apply post-training static quantization (PTQ).

        Static quantization quantizes both weights and activations using
        calibration data to determine activation scale factors.

        Args:
            model: PyTorch model to quantize.
            calibration_data: Calibration dataset. Uses internal one if None.

        Returns:
            Tuple of (quantized_model, quantization_result).
        """
        result = QuantizationResult(precision=PrecisionMode.INT8)
        start_time = time.time()

        try:
            import torch
            import torch.quantization as quant

            model.eval()

            # Set quantization config
            model.qconfig = quant.get_default_qconfig("qnnpack")

            # Prepare for quantization
            prepared_model = quant.prepare(model)

            # Calibrate
            if calibration_data is None:
                self._calibration_dataset.load()
                calibration_data = self._calibration_dataset

            with torch.no_grad():
                for sample in calibration_data:
                    tensor_input = torch.from_numpy(sample)
                    prepared_model(tensor_input)

            # Convert to quantized model
            quantized_model = quant.convert(prepared_model)

            result.quantization_time_s = time.time() - start_time
            result.success = True

            # Compute size metrics
            original_size = self._compute_model_size_mb(model)
            quantized_size = self._compute_model_size_mb(quantized_model)
            result.original_size_mb = original_size
            result.quantized_size_mb = quantized_size
            result.compression_ratio = original_size / max(quantized_size, 0.01)

            print(
                f"[Quantizer] Static INT8 quantization: "
                f"{original_size:.1f} MB -> {quantized_size:.1f} MB "
                f"({result.compression_ratio:.1f}x)"
            )

            return quantized_model, result

        except ImportError:
            result.error = "PyTorch not available for static quantization"
            return model, result
        except Exception as e:
            result.error = str(e)
            print(f"[Quantizer] Static quantization failed: {e}")
            return model, result

    def quantize_fp16(self, model: Any) -> Tuple[Any, QuantizationResult]:
        """
        Convert model to FP16 (half precision).

        FP16 quantization reduces model size by ~2x with minimal accuracy
        loss. Widely supported on modern GPUs and edge accelerators.

        Args:
            model: PyTorch model to convert.

        Returns:
            Tuple of (fp16_model, quantization_result).
        """
        result = QuantizationResult(precision=PrecisionMode.FP16)
        start_time = time.time()

        try:
            import torch

            model.eval()
            original_size = self._compute_model_size_mb(model)
            fp16_model = model.half()
            quantized_size = self._compute_model_size_mb(fp16_model)

            result.original_size_mb = original_size
            result.quantized_size_mb = quantized_size
            result.compression_ratio = original_size / max(quantized_size, 0.01)
            result.quantization_time_s = time.time() - start_time
            result.success = True

            print(
                f"[Quantizer] FP16 conversion: "
                f"{original_size:.1f} MB -> {quantized_size:.1f} MB"
            )

            return fp16_model, result

        except ImportError:
            result.error = "PyTorch not available for FP16 quantization"
            return model, result
        except Exception as e:
            result.error = str(e)
            return model, result

    @staticmethod
    def _compute_model_size_mb(model: Any) -> float:
        """Compute model size in megabytes from parameters."""
        try:
            import torch
            total_bytes = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            )
            return total_bytes / (1024 * 1024)
        except Exception:
            return 0.0


class ONNXRuntimeInferencer:
    """
    ONNX Runtime-based inference for edge devices.

    Provides hardware-accelerated inference using ONNX Runtime with
    support for CPU, CUDA, TensorRT, and OpenVINO execution providers.
    Includes session optimization and I/O binding for minimal latency.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._session: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._warm = False

    def initialize(self) -> None:
        """Initialize the ONNX Runtime inference session."""
        try:
            import onnxruntime as ort

            # Select execution providers based on device
            providers = self._get_providers()
            session_options = ort.SessionOptions()

            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = self.config.num_threads
            session_options.inter_op_num_threads = max(1, self.config.num_threads // 2)

            # Enable memory optimization for edge devices
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True

            # Disable fallback for strict edge deployment
            session_options.session_log_severity_level = 3

            # Set cache dir for pre-compiled kernels
            if self.config.enable_caching:
                os.makedirs(self.config.cache_dir, exist_ok=True)
                session_options.optimized_model_filepath = os.path.join(
                    self.config.cache_dir, "optimized_model.onnx"
                )

            self._session = ort.InferenceSession(
                self.config.onnx_path,
                sess_options=session_options,
                providers=providers,
            )

            # Cache I/O metadata
            for inp in self._session.get_inputs():
                self._input_names.append(inp.name)
                self._input_shapes[inp.name] = inp.shape
            for out in self._session.get_outputs():
                self._output_names.append(out.name)

            print(
                f"[ONNXRuntime] Session created with providers: "
                f"{self._session.get_providers()}"
            )

        except ImportError:
            print("[ONNXRuntime] onnxruntime not available")
        except Exception as e:
            print(f"[ONNXRuntime] Initialization failed: {e}")

    def _get_providers(self) -> List[str]:
        """Determine the best execution providers for the target device."""
        device_provider_map = {
            EdgeDevice.JETSON_XAVIER: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.JETSON_ORIN: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.JETSON_NANO: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.INTEL_NUC: ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.RASPBERRY_PI5: ["CPUExecutionProvider"],
            EdgeDevice.QUALCOMM_RB5: ["QNNExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.GENERIC_GPU: ["CUDAExecutionProvider", "CPUExecutionProvider"],
            EdgeDevice.GENERIC_CPU: ["CPUExecutionProvider"],
        }
        return device_provider_map.get(self.config.device, ["CPUExecutionProvider"])

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using ONNX Runtime.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        if self._session is None:
            self.initialize()

        # Prepare feed dict
        feed = {}
        for name in self._input_names:
            if name in inputs:
                arr = inputs[name]
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                feed[name] = arr
            else:
                # Use first available input with default shape
                shape = self._input_shapes.get(name, (1, 3, 224, 224))
                # Replace dynamic dims with 1
                concrete_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in shape)
                feed[name] = np.random.randn(*concrete_shape).astype(np.float32)

        # Run inference
        outputs = self._session.run(self._output_names, feed)

        return dict(zip(self._output_names, outputs))

    def warmup(self, iterations: int = 10) -> None:
        """Warm up the inference session with dummy inputs."""
        if self._warm or self._session is None:
            return

        dummy_inputs = {}
        for name in self._input_names:
            shape = self._input_shapes.get(name, (1, 3, 224, 224))
            concrete_shape = tuple(d if isinstance(d, int) and d > 0 else 1 for d in shape)
            dummy_inputs[name] = np.random.randn(*concrete_shape).astype(np.float32)

        for _ in range(iterations):
            self._session.run(self._output_names, dummy_inputs)

        self._warm = True
        print(f"[ONNXRuntime] Warmed up with {iterations} iterations")

    def get_session_info(self) -> Dict[str, Any]:
        """Get ONNX Runtime session information."""
        if self._session is None:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "providers": self._session.get_providers(),
            "input_names": self._input_names,
            "output_names": self._output_names,
            "input_shapes": self._input_shapes,
        }


class OpenVINOInferencer:
    """
    OpenVINO-based inference for Intel edge platforms.

    Leverages Intel's OpenVINO toolkit for optimized inference on
    Intel CPUs, integrated GPUs, and VPU (Myriad X) accelerators.
    Supports model caching and automatic device selection.
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._model: Optional[Any] = None
        self._compiled_model: Optional[Any] = None
        self._infer_request: Optional[Any] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def initialize(self) -> None:
        """Initialize the OpenVINO inference pipeline."""
        try:
            from openvino.runtime import Core

            core = Core()

            # Enable model caching
            if self.config.enable_caching:
                os.makedirs(self.config.cache_dir, exist_ok=True)
                core.set_property({"CACHE_DIR": self.config.cache_dir})

            # Select device
            device = self._select_device(core)

            # Load model
            xml_path = self.config.openvino_xml_path
            if xml_path and os.path.exists(xml_path):
                self._model = core.read_model(model=xml_path)
            elif self.config.onnx_path and os.path.exists(self.config.onnx_path):
                self._model = core.read_model(model=self.config.onnx_path)
            else:
                print("[OpenVINO] No valid model path provided")
                return

            # Configure precision hints
            if self.config.precision == PrecisionMode.INT8:
                core.set_property(device, {"PERFORMANCE_HINT": "THROUGHPUT"})
            elif self.config.precision == PrecisionMode.FP16:
                core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})

            # Set thread count for CPU
            if device == "CPU":
                core.set_property(device, {
                    "INFERENCE_NUM_THREADS": str(self.config.num_threads),
                    "AFFINITY": "CORE",
                })

            # Compile model
            self._compiled_model = core.compile_model(self._model, device)
            self._infer_request = self._compiled_model.create_infer_request()

            # Cache I/O names
            for inp in self._compiled_model.inputs:
                self._input_names.append(inp.get_any_name())
            for out in self._compiled_model.outputs:
                self._output_names.append(out.get_any_name())

            print(f"[OpenVINO] Initialized on {device} with {len(self._input_names)} inputs")

        except ImportError:
            print("[OpenVINO] openvino package not available")
        except Exception as e:
            print(f"[OpenVINO] Initialization failed: {e}")

    def _select_device(self, core: Any) -> str:
        """Auto-select the best available OpenVINO device."""
        available = core.available_devices

        # Priority order for edge deployment
        priority = ["GPU", "NPU", "MYRIAD", "CPU"]
        for dev in priority:
            if dev in available:
                return dev
        return available[0] if available else "CPU"

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using OpenVINO.

        Args:
            inputs: Dictionary mapping input names to numpy arrays.

        Returns:
            Dictionary mapping output names to numpy arrays.
        """
        if self._infer_request is None:
            self.initialize()
            if self._infer_request is None:
                return {}

        # Prepare input tensor map
        input_tensors = {}
        for i, name in enumerate(self._input_names):
            if name in inputs:
                input_tensors[i] = inputs[name].astype(np.float32)
            else:
                # Use model's input shape
                input_shape = self._compiled_model.inputs[i].get_shape()
                concrete = tuple(d if d > 0 else 1 for d in input_shape)
                input_tensors[i] = np.random.randn(*concrete).astype(np.float32)

        # Run inference
        self._infer_request.infer(input_tensors)

        # Collect outputs
        results = {}
        for i, name in enumerate(self._output_names):
            results[name] = self._infer_request.get_output_tensor(i).data.copy()

        return results

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available OpenVINO devices."""
        try:
            from openvino.runtime import Core
            core = Core()
            info = {"available_devices": core.available_devices}
            for device in core.available_devices:
                info[f"{device}_full_name"] = core.get_property(device, "FULL_DEVICE_NAME")
            return info
        except ImportError:
            return {"error": "OpenVINO not available"}


class EdgeInferenceManager:
    """
    Unified edge inference manager.

    Coordinates quantization, model conversion, and runtime inference
    across multiple backends (ONNX Runtime, OpenVINO) with automatic
    fallback and device-specific optimization.

    Example:
        >>> config = EdgeConfig(device=EdgeDevice.JETSON_ORIN, precision=PrecisionMode.FP16)
        >>> manager = EdgeInferenceManager(config)
        >>> manager.initialize()
        >>> result = manager.infer({"input": image_array})
    """

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config
        self._onnx_inferencer: Optional[ONNXRuntimeInferencer] = None
        self._openvino_inferencer: Optional[OpenVINOInferencer] = None
        self._quantizer: Optional[PyTorchQuantizer] = None
        self._active_backend: str = "none"
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the edge inference manager with the best available backend."""
        # Try OpenVINO first for Intel devices
        if self.config.device in (EdgeDevice.INTEL_NUC,):
            self._openvino_inferencer = OpenVINOInferencer(self.config)
            try:
                self._openvino_inferencer.initialize()
                if self._openvino_inferencer._compiled_model is not None:
                    self._active_backend = "openvino"
                    self._initialized = True
                    return
            except Exception:
                pass

        # Try ONNX Runtime for NVIDIA / general devices
        if self.config.onnx_path and os.path.exists(self.config.onnx_path):
            self._onnx_inferencer = ONNXRuntimeInferencer(self.config)
            try:
                self._onnx_inferencer.initialize()
                self._onnx_inferencer.warmup(self.config.warmup_iterations)
                self._active_backend = "onnxruntime"
                self._initialized = True
                return
            except Exception:
                pass

        # Fallback: try OpenVINO with ONNX model
        if self.config.onnx_path and os.path.exists(self.config.onnx_path):
            self._openvino_inferencer = OpenVINOInferencer(self.config)
            try:
                self._openvino_inferencer.initialize()
                if self._openvino_inferencer._compiled_model is not None:
                    self._active_backend = "openvino_fallback"
                    self._initialized = True
                    return
            except Exception:
                pass

        print("[EdgeManager] No inference backend could be initialized")

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference using the active backend.

        Args:
            inputs: Input data dictionary.

        Returns:
            Output data dictionary.

        Raises:
            RuntimeError: If no inference backend is available.
        """
        if not self._initialized:
            self.initialize()

        if self._active_backend in ("onnxruntime",) and self._onnx_inferencer:
            return self._onnx_inferencer.infer(inputs)
        elif self._active_backend in ("openvino", "openvino_fallback") and self._openvino_inferencer:
            return self._openvino_inferencer.infer(inputs)
        else:
            raise RuntimeError("No active inference backend available")

    def quantize_model(
        self,
        model: Any,
        method: str = "fp16",
        calibration_data: Optional[CalibrationDataset] = None,
    ) -> Tuple[Any, QuantizationResult]:
        """
        Quantize a PyTorch model for edge deployment.

        Args:
            model: PyTorch model to quantize.
            method: Quantization method ("fp16", "int8_dynamic", "int8_static").
            calibration_data: Calibration dataset for static quantization.

        Returns:
            Tuple of (quantized_model, quantization_result).
        """
        self._quantizer = PyTorchQuantizer(self.config)

        if method == "fp16":
            return self._quantizer.quantize_fp16(model)
        elif method == "int8_dynamic":
            quantized = self._quantizer.quantize_dynamic(model)
            return quantized, QuantizationResult(
                precision=PrecisionMode.INT8, success=True
            )
        elif method == "int8_static":
            return self._quantizer.quantize_static(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def get_deployment_info(self) -> Dict[str, Any]:
        """Get current deployment information."""
        return {
            "device": self.config.device.value,
            "precision": self.config.precision.value,
            "active_backend": self._active_backend,
            "initialized": self._initialized,
            "onnx_available": self._onnx_inferencer is not None
            and self._onnx_inferencer._session is not None,
            "openvino_available": self._openvino_inferencer is not None
            and self._openvino_inferencer._compiled_model is not None,
        }
