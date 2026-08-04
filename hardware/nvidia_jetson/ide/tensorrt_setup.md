# TensorRT Setup & Engine Building

> Companion document to `python/model_loader.py` and `cpp/tensor_rt.{hpp,cpp}`.
> Tested with TensorRT 8.5 / 8.6 (JetPack 5.x) and TensorRT 10.x (JetPack 6.x).

TensorRT is NVIDIA's high-performance inference optimizer. It takes an
ONNX model (or a trained framework checkpoint via the TF/PyTorch parsers),
performs layer fusion, kernel auto-tuning, precision calibration, and emits
a serialized *engine* that can be reloaded without rebuilding.

The full workflow is:

```
   model.onnx  ──[builder + optimizer]──▶  model.engine
                                               │
                                               ▼
                                  InferenceEngine (Python / C++)
```

---

## 1. Prerequisites

- JetPack 5.1.2+ with TensorRT 8.5+ (verify with `python3 -c "import tensorrt; print(tensorrt.__version__)"`).
- An ONNX model. For YOLOv5/v8, use `models/export.py` from the Ultralytics
  repo and then `python3 -m onnxsim model.onnx model.sim.onnx` to simplify
  the graph (required for clean TRT parsing).
- The `trtexec` CLI, located at `/usr/src/tensorrt/bin/trtexec`.

Set up the environment once:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH
```

---

## 2. Building an Engine with `trtexec`

The fastest way to verify an ONNX model parses and to benchmark it:

```bash
# FP32 baseline
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.fp32.engine --workspace=4096

# FP16 (recommended default — 2× speed-up, ~0.5 mAP loss)
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.fp16.engine \
        --workspace=4096 --fp16

# INT8 with random calibration (use --calib= for real data)
trtexec --onnx=yolov5s.onnx --saveEngine=yolov5s.int8.engine \
        --workspace=4096 --int8 --calib=calib.cache
```

Inspect a built engine:
```bash
trtexec --loadEngine=yolov5s.fp16.engine
```

---

## 3. Building Engines in Python (`python/model_loader.py`)

The `ModelLoader` class wraps the builder to support:

- FP32 / FP16 / INT8 selection
- Dynamic batch & image dimensions via optimization profiles
- Engine caching keyed on ONNX SHA-256 + build flags
- INT8 calibration with a user-supplied dataset

```python
from python.model_loader import ModelLoader

loader = ModelLoader(cache_dir="/models/trt_cache")

engine = loader.build(
    onnx_path="/models/yolov5s.onnx",
    fp16=True,
    int8=False,
    batch_range=(1, 8, 16),        # min, optimal, max
    image_shape=(3, 640, 640),
    workspace_gb=4,
    calib_dataset=None,            # list of np.ndarray for INT8
)
# engine is a tensorrt.ICudaEngine, ready to pass to InferenceEngine
```

The loader stores the engine at
`/models/trt_cache/<sha256>_<fp16|int8|fp32>_<shape>.engine` and skips
rebuilding if it already exists.

---

## 4. INT8 Calibration

INT8 quantization requires a *calibrator* that feeds representative input
samples to the network so TensorRT can collect activation statistics and
compute scale factors. For YOLO-style detection networks, **200–500
calibration images** is sufficient.

### 4.1 Implementing a calibrator

`python/model_loader.py` ships an `Int8Calibrator` subclass of
`trt.IInt8EntropyCalibrator2` that:

1. Pre-loads calibration images into pinned host memory.
2. Returns them one-by-one in `get_batch()` as a list of `int` device pointers.
3. Caches the resulting scale table to `calib.cache` so subsequent builds
   don't need to re-read the dataset.

```python
calib = loader.build_int8_calibrator(
    image_dir="/data/calib/coco_2017",
    cache_file="/models/calib/yolov5s_int8.cache",
    image_shape=(3, 640, 640),
    batch_size=8,
    max_batches=64,
)
```

### 4.2 Choosing the right calibrator

| Calibrator | When to use |
|------------|-------------|
| `IInt8EntropyCalibrator2` (default) | Dense / regression tasks (YOLO, SSD) |
| `IInt8MinMaxCalibrator` | NLP / transformer activations |
| `IInt8LegacyCalibrator` | Backwards compat with TRT 5/6 caches |

### 4.3 Verifying INT8 accuracy

Always run the evaluation harness on a holdout set after building an INT8
engine. Compare mAP / pixel-IoU against the FP16 baseline. A drop greater
than 1 mAP point indicates the calibration set was not representative.

---

## 5. Dynamic Shapes (Optimization Profiles)

Most detection networks need to support a *range* of input resolutions
(for example, to run the same engine at 640×640 for exploration and
1280×1280 for high-resolution mode). TensorRT handles this via
**optimization profiles**:

```cpp
// cpp/tensor_rt.cpp — excerpt
nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
profile->setDimensions(network->getInput(0)->getName(),
    nvinfer1::OptProfileSelector::kMIN,  nvinfer1::Dims4{1, 3, 320, 320});
profile->setDimensions(network->getInput(0)->getName(),
    nvinfer1::OptProfileSelector::kOPT,  nvinfer1::Dims4{1, 3, 640, 640});
profile->setDimensions(network->getInput(0)->getName(),
    nvinfer1::OptProfileSelector::kMAX,  nvinfer1::Dims4{1, 3, 1280, 1280});
config->addOptimizationProfile(profile);
```

In Python the same thing is one call on `builder.create_optimization_profile()`
in `python/model_loader.py::ModelLoader.build()`.

**Selecting a profile at inference time**:
```python
context.set_input_shape("images", (1, 3, 1280, 1280))
# or in the older API:
context.active_optimization_profile = 0
context.set_binding_shape(0, (1, 3, 1280, 1280))
```

---

## 6. Custom Plugins

Some ONNX operators (e.g. YOLOv5 `Focus`, EfficientNMS) require a custom
plugin. Register plugins before deserializing the engine:

```cpp
// cpp/tensor_rt.cpp
#include "NvInferPlugin.h"
initLibNvInferPlugins(sample_logger, "");
```

```python
# python
import tensorrt as trt
trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.INFO), "")
```

For third-party plugin libraries (e.g. TensorRT plugins from
`tensorrt-plugins`):
```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8:$LD_PRELOAD
```

---

## 7. Async Inference & CUDA Streams

To overlap H↔D memory copies with compute, use multiple CUDA streams and
`enqueueV2` (TRT 8.x) or `enqueueV3` (TRT 10.x):

```python
# python/inference.py
stream1 = cuda.Stream()
stream2 = cuda.Stream()

# Submit batch 1 on stream1, batch 2 on stream2 — they will execute
# concurrently on different SMs if the GPU has spare capacity.
context.execute_async_v2(buffers=[d_input, d_output], stream_handle=stream1.handle())
context.execute_async_v2(buffers=[d_input2, d_output2], stream_handle=stream2.handle())
```

For maximum throughput, double-buffer the host input:

```
  Frame N  → H2D copy on stream A
  Frame N-1 → inference on stream B
  Frame N-2 → D2H copy on stream C
```

See `cpp/inference_engine.cpp::InferenceEngine::infer_async()` for a C++
implementation using `cudaMemcpyAsync` with pinned host buffers.

---

## 8. Common Issues

### 8.1 "Could not find layer … for ONNX node …"
The ONNX graph contains an operator that the TRT parser does not support.
Fixes:
- Simplify the graph: `python3 -m onnxsim model.onnx model.sim.onnx`.
- Install missing plugins: `trt.init_libnvinfer_plugins(...)`.
- Re-export the ONNX with `opset_version=11` (some operators became
  supported only in opset 11+).

### 8.2 "CUDA out of memory" while building
The builder workspace has a hard cap on Jetson (shared with the OS):
```cpp
config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1 GB
```
Or via `trtexec --workspace=1024`. Close all other GPU processes
(`sudo systemctl stop gdm3`) before building large engines.

### 8.3 "Engine was built with different version of TensorRT"
Serialized engines are NOT portable across TensorRT versions, even minor
ones. Rebuild from ONNX after every JetPack upgrade.

### 8.4 FP16 → FP32 numerical drift
- Verify that you are not comparing against a model trained with
  `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False`.
- For post-processing that uses `expit()` / log-probabilities, keep FP32
  for those layers using `config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS)`
  and `layer->setPrecision(nvinfer1::DataType::kFLOAT)`.

### 8.5 DLA offload
To offload layers to the Deep Learning Accelerator (Xavier/Orin only):
```cpp
config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
config->setDLACore(0);
config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);  // for unsupported layers
```
DLA supports FP16 only; INT8 support is limited to a subset of layers.

---

## 9. Performance Checklist

Before declaring an engine "production ready":

- [ ] `trtexec --loadEngine=... --iterations=200` reports the expected median latency.
- [ ] `nvpmodel -m 0 && jetson_clocks` for max-power benchmark.
- [ ] Profile with `nsys profile` and confirm there are no large gaps
      (H2D/D2H or host-side pre/post processing).
- [ ] Compare INT8 vs FP16 — if speed-up < 1.3×, calibrator likely failed.
- [ ] Run the accuracy harness on a held-out test set.
- [ ] Rebuild the engine after every JetPack upgrade.
- [ ] Verify the engine can be reloaded from disk (cache hit path in
      `python/model_loader.py`).
