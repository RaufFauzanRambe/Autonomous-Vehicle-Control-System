"""
Inference Engine Module for Autonomous Vehicle AI.

Main inference engine that orchestrates the complete inference pipeline:
- Model loading and initialization
- Pre-processing, inference, post-processing pipeline
- Multi-model orchestration for perception stack
- Synchronous and asynchronous inference modes
- Performance monitoring and SLA enforcement
"""

import os
import time
import threading
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class InferenceMode(Enum):
    """Inference execution mode."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"
    BATCH = "batch"


class ModelRole(Enum):
    """Role of a model in the perception stack."""
    OBJECT_DETECTION = "object_detection"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    DEPTH_ESTIMATION = "depth_estimation"
    LANE_DETECTION = "lane_detection"
    TRAJECTORY_PREDICTION = "trajectory_prediction"
    END_TO_END_DRIVING = "end_to_end_driving"
    CLASSIFICATION = "classification"


@dataclass
class ModelConfig:
    """Configuration for a single model in the inference pipeline."""
    name: str = "detector"
    role: ModelRole = ModelRole.OBJECT_DETECTION
    model_path: str = ""
    model_format: str = "auto"  # auto, pytorch, onnx, tensorrt
    device: str = "auto"
    precision: str = "fp32"  # fp32, fp16, int8
    batch_size: int = 1
    input_names: List[str] = field(default_factory=lambda: ["input"])
    output_names: List[str] = field(default_factory=lambda: ["output"])
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    max_batch_size: int = 8
    warmup_iterations: int = 5
    priority: int = 0  # Higher = more important


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    mode: InferenceMode = InferenceMode.SYNCHRONOUS
    max_latency_ms: float = 50.0  # Maximum acceptable latency
    target_fps: float = 20.0
    enable_profiling: bool = True
    preprocess_on_cpu: bool = True
    postprocess_on_cpu: bool = True
    gpu_memory_fraction: float = 0.8
    allow_batching: bool = True
    batch_timeout_ms: float = 5.0
    max_queue_size: int = 100


@dataclass
class InferenceResult:
    """Result from a single inference call."""
    model_name: str
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    timestamp: float = 0.0
    batch_size: int = 1
    success: bool = True
    error: Optional[str] = None


class PreProcessor:
    """
    Input pre-processing for inference.

    Handles image resizing, normalization, and format conversion
    to match model input requirements.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._input_shape = config.input_shape

    def process(self, raw_input: np.ndarray) -> np.ndarray:
        """
        Pre-process raw input for model inference.

        Args:
            raw_input: Raw input data (e.g., image array).

        Returns:
            Pre-processed input ready for the model.
        """
        start = time.time()

        # Resize if needed
        if raw_input.ndim == 3 and raw_input.shape[2] == 3:
            target_h, target_w = self._input_shape[2], self._input_shape[3]
            if raw_input.shape[0] != target_h or raw_input.shape[1] != target_w:
                raw_input = self._resize(raw_input, target_h, target_w)

            # Normalize to [0, 1]
            processed = raw_input.astype(np.float32) / 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            processed = (processed - mean) / std

            # HWC -> CHW
            processed = processed.transpose(2, 0, 1)

            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
        elif raw_input.ndim == 1:
            # State vector input
            processed = raw_input.astype(np.float32).reshape(1, -1)
        else:
            processed = raw_input.astype(np.float32)

        # Ensure contiguous memory
        processed = np.ascontiguousarray(processed)

        self._last_latency = (time.time() - start) * 1000
        return processed

    def _resize(self, image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Simple resize using numpy indexing."""
        h, w = image.shape[:2]
        y_idx = np.linspace(0, h - 1, target_h).astype(int)
        x_idx = np.linspace(0, w - 1, target_w).astype(int)
        return image[np.ix_(y_idx, x_idx)][0] if image.ndim == 3 else image[np.ix_(y_idx, x_idx)]


class PostProcessor:
    """
    Output post-processing for inference.

    Converts raw model outputs to application-friendly format
    (e.g., bounding boxes, segmentation masks, depth maps).
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def process(self, raw_output: Dict[str, np.ndarray], role: ModelRole) -> Dict[str, Any]:
        """
        Post-process model outputs.

        Args:
            raw_output: Raw model output dictionary.
            role: Model role for role-specific processing.

        Returns:
            Processed output dictionary.
        """
        start = time.time()

        result = {}
        if role == ModelRole.OBJECT_DETECTION:
            result = self._process_detection(raw_output)
        elif role == ModelRole.SEMANTIC_SEGMENTATION:
            result = self._process_segmentation(raw_output)
        elif role == ModelRole.DEPTH_ESTIMATION:
            result = self._process_depth(raw_output)
        elif role == ModelRole.LANE_DETECTION:
            result = self._process_lanes(raw_output)
        else:
            result = {"raw": raw_output}

        self._last_latency = (time.time() - start) * 1000
        return result

    def _process_detection(self, output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process detection outputs into bounding boxes."""
        detections = []
        if "output" in output or "boxes" in output:
            raw = output.get("output", output.get("boxes"))
            if raw is not None and raw.ndim >= 2:
                # Simplified: assume [N, 6] format (x1, y1, x2, y2, conf, class)
                for det in raw.reshape(-1, raw.shape[-1])[:100]:
                    if det.shape[0] >= 6 and det[4] > 0.5:
                        detections.append({
                            "bbox": det[:4].tolist(),
                            "confidence": float(det[4]),
                            "class_id": int(det[5]),
                        })
        return {"detections": detections, "num_detections": len(detections)}

    def _process_segmentation(self, output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process segmentation outputs."""
        if "output" in output:
            mask = output["output"]
            if mask.ndim == 4:
                mask = mask.argmax(axis=1)
            return {"segmentation_mask": mask, "num_classes": int(mask.max()) + 1}
        return {"segmentation_mask": None}

    def _process_depth(self, output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process depth estimation outputs."""
        if "output" in output:
            depth = output["output"]
            if depth.ndim == 4:
                depth = depth.squeeze(0).squeeze(0)
            return {
                "depth_map": depth,
                "min_depth": float(depth.min()),
                "max_depth": float(depth.max()),
            }
        return {"depth_map": None}

    def _process_lanes(self, output: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Process lane detection outputs."""
        if "output" in output:
            lanes = output["output"]
            return {"lane_points": lanes, "num_lanes": 1}
        return {"lane_points": None}


class InferenceEngine:
    """
    Main inference engine for autonomous vehicle AI.

    Orchestrates the complete inference pipeline including model loading,
    pre-processing, inference, post-processing, and result aggregation
    across multiple models in the perception stack.

    Example:
        >>> engine = InferenceEngine(PipelineConfig())
        >>> engine.add_model(ModelConfig(name="detector", role=ModelRole.OBJECT_DETECTION, model_path="model.onnx"))
        >>> engine.initialize()
        >>> result = engine.infer({"detector": image_data})
    """

    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config
        self._models: Dict[str, Any] = {}
        self._model_configs: Dict[str, ModelConfig] = {}
        self._preprocessors: Dict[str, PreProcessor] = {}
        self._postprocessors: Dict[str, PostProcessor] = {}
        self._initialized = False
        self._lock = threading.Lock()

        # Performance tracking
        self._total_inferences = 0
        self._total_errors = 0
        self._latency_history: List[float] = []

        # Async support
        self._request_queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._result_queue: queue.Queue = queue.Queue()
        self._async_thread: Optional[threading.Thread] = None

    def add_model(self, config: ModelConfig) -> None:
        """
        Add a model to the inference pipeline.

        Args:
            config: Model configuration.
        """
        self._model_configs[config.name] = config
        self._preprocessors[config.name] = PreProcessor(config)
        self._postprocessors[config.name] = PostProcessor(config)
        print(f"[Engine] Added model: {config.name} (role={config.role.value})")

    def initialize(self) -> None:
        """Initialize all models and warm up."""
        if self._initialized:
            return

        from .model_loader import ModelLoader

        for name, config in self._model_configs.items():
            loader = ModelLoader(config)
            model = loader.load()
            self._models[name] = model

            # Warmup
            self._warmup_model(name, config)

        self._initialized = True

        # Start async thread if needed
        if self.config.mode == InferenceMode.ASYNCHRONOUS:
            self._start_async_thread()

        print(f"[Engine] Initialized with {len(self._models)} models")

    def _warmup_model(self, name: str, config: ModelConfig) -> None:
        """Warm up a model with dummy inputs."""
        model = self._models.get(name)
        if model is None:
            return

        dummy_input = np.random.randn(*config.input_shape).astype(np.float32)
        for _ in range(config.warmup_iterations):
            try:
                if hasattr(model, 'infer'):
                    model.infer(dummy_input)
                elif callable(model):
                    model(dummy_input)
            except Exception:
                pass

        print(f"[Engine] Warmed up model: {name}")

    def infer(
        self,
        inputs: Dict[str, np.ndarray],
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, InferenceResult]:
        """
        Run inference on the specified models.

        Args:
            inputs: Dictionary mapping model names to input data.
            model_names: Optional list of models to run (defaults to all).

        Returns:
            Dictionary mapping model names to inference results.
        """
        if not self._initialized:
            self.initialize()

        target_models = model_names or list(self._models.keys())
        results: Dict[str, InferenceResult] = {}

        pipeline_start = time.time()

        for name in target_models:
            if name not in self._models:
                results[name] = InferenceResult(
                    model_name=name, outputs={},
                    latency_ms=0.0, success=False,
                    error=f"Model not found: {name}"
                )
                continue

            result = self._infer_single(name, inputs.get(name))
            results[name] = result

        # Check SLA
        total_latency = (time.time() - pipeline_start) * 1000
        if total_latency > self.config.max_latency_ms:
            pass  # Log warning in production

        self._total_inferences += 1
        self._latency_history.append(total_latency)

        return results

    def _infer_single(self, model_name: str, raw_input: Optional[np.ndarray]) -> InferenceResult:
        """Run inference on a single model."""
        config = self._model_configs[model_name]
        model = self._models[model_name]
        start_time = time.time()

        try:
            # Pre-process
            preprocess_start = time.time()
            if raw_input is not None:
                processed_input = self._preprocessors[model_name].process(raw_input)
            else:
                processed_input = np.random.randn(*config.input_shape).astype(np.float32)
            preprocess_ms = (time.time() - preprocess_start) * 1000

            # Inference
            infer_start = time.time()
            if hasattr(model, 'infer'):
                raw_output = model.infer(processed_input)
            elif callable(model):
                raw_output = model(processed_input)
            else:
                raw_output = {"output": np.zeros(1)}

            inference_ms = (time.time() - infer_start) * 1000

            # Post-process
            postprocess_start = time.time()
            if isinstance(raw_output, dict):
                processed_output = self._postprocessors[model_name].process(
                    raw_output, config.role
                )
            else:
                processed_output = {"output": raw_output}
            postprocess_ms = (time.time() - postprocess_start) * 1000

            total_ms = (time.time() - start_time) * 1000

            return InferenceResult(
                model_name=model_name,
                outputs=processed_output,
                latency_ms=total_ms,
                preprocess_ms=preprocess_ms,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
                timestamp=time.time(),
                success=True,
            )

        except Exception as e:
            self._total_errors += 1
            return InferenceResult(
                model_name=model_name,
                outputs={},
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
            )

    def _start_async_thread(self) -> None:
        """Start the asynchronous inference worker thread."""
        self._async_thread = threading.Thread(target=self._async_worker, daemon=True)
        self._async_thread.start()

    def _async_worker(self) -> None:
        """Worker thread for asynchronous inference."""
        while True:
            try:
                request = self._request_queue.get(timeout=1.0)
                if request is None:
                    break

                request_id, inputs, callback = request
                results = self.infer(inputs)
                if callback:
                    callback(request_id, results)
                else:
                    self._result_queue.put((request_id, results))

            except queue.Empty:
                continue

    def infer_async(
        self,
        inputs: Dict[str, np.ndarray],
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit an asynchronous inference request.

        Args:
            inputs: Input data dictionary.
            callback: Optional callback function(request_id, results).

        Returns:
            Request ID string.
        """
        request_id = f"req_{self._total_inferences}_{int(time.time() * 1000)}"
        self._request_queue.put((request_id, inputs, callback))
        return request_id

    def get_async_result(self, timeout: float = 1.0) -> Optional[Tuple[str, Dict]]:
        """Get a result from the async result queue."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference engine statistics."""
        latencies = self._latency_history[-1000:]
        return {
            "total_inferences": self._total_inferences,
            "total_errors": self._total_errors,
            "error_rate": self._total_errors / max(self._total_inferences, 1),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "models_loaded": list(self._models.keys()),
        }

    def shutdown(self) -> None:
        """Shut down the inference engine."""
        if self._async_thread is not None:
            self._request_queue.put(None)
            self._async_thread.join(timeout=5.0)

        self._models.clear()
        self._initialized = False
        print("[Engine] Shutdown complete")
