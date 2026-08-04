# CUDA Toolkit Setup on Jetson

> Companion document to `cuda/*.cu`, `cuda/*.cuh`, and `cpp/cuda_memory.cpp`.
> Applies to JetPack 4.6 (CUDA 10.2), 5.x (CUDA 11.4), and 6.x (CUDA 12.x).

Jetson modules ship with the full CUDA Toolkit pre-installed. This guide
covers environment configuration, the supported compute capabilities,
profiling with Nsight Systems / Nsight Compute, and building the CUDA
samples shipped in this directory.

---

## 1. Environment

Add the following to `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export OPENBLAS_CORETYPE=ARMV8        # silence OpenBLAS warnings on Nano
ulimit -n 65535                        # raise file descriptor limit for ROS2
```

Verify:
```bash
nvcc --version
# Cuda compilation tools, release 11.4, V11.4.315
```

---

## 2. Compute Capability by Module

The **compute capability (sm_xx)** of the GPU dictates which CUDA features
are available. Always compile with the right `-arch=sm_XX` for your
target module.

| Module | Compute Capability | `-arch=` | Volta Tensor Cores? | Ampere Tensor Cores? | BF16? |
|--------|:------------------:|:--------:|:-------------------:|:--------------------:|:-----:|
| Jetson Nano (Tegra X1, Maxwell) | 5.3 | `sm_53` | ✗ | ✗ | ✗ |
| Jetson TX2 (Pascal) | 6.2 | `sm_62` | ✗ | ✗ | ✗ |
| Jetson Xavier NX / AGX Xavier (Volta) | 7.2 | `sm_72` | ✓ (FP16) | ✗ | ✗ |
| Jetson Orin Nano/NX/AGX (Ampere) | 8.7 | `sm_87` | ✓ (FP16/BF16/TF32) | ✓ | ✓ |

When distributing a binary across modules, use `-gencode` for each:
```bash
nvcc -gencode arch=compute_53,code=sm_53 \
     -gencode arch=compute_72,code=sm_72 \
     -gencode arch=compute_87,code=sm_87 \
     -O3 -std=c++17 kernel.cu -o kernel
```

For PTX (forward-compatible JIT), add:
```bash
-gencode arch=compute_87,code=compute_87
```

### Per-module notable features
- **sm_53 (Nano):** No `__shfl_sync` with full mask, no warp-aggregated
  atomics, no cooperative groups `reduce`. Use the warp-reduce pattern with
  `__shfl_down_sync(0xffffffff, val, delta)`.
- **sm_72 (Xavier):** First Jetson with Tensor Cores — `wmma::mma_sync`
  and `mma_async` available. Use `nvcuda::wmma` for FP16 GEMM.
- **sm_87 (Orin):** Async copy (`cp.async`) and TMA are available, and
  the L2 cache supports residency control.

---

## 3. Compiling the CUDA Kernels in This Repo

All kernels in `cuda/` are designed to compile with a single `nvcc`
invocation. A minimal `Makefile`:

```makefile
CUDA_ARCH ?= sm_72
NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++17 --use_fast_math \
              -Xcompiler -fPIC -Xcompiler -Wall \
              -lineinfo \
              -gencode arch=compute_$(subst sm,,$(CUDA_ARCH)),code=$(CUDA_ARCH)

TARGETS := kernel image_processing tensor_operations

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(TARGETS)
```

Build & run:
```bash
make CUDA_ARCH=sm_72
./kernel
./image_processing
./tensor_operations
```

> `--use_fast_math` enables `__sinf`/`__cosf` and reciprocal approximations.
> Disable it when numerical reproducibility matters (e.g. unit tests).

---

## 4. Nsight Systems — Timeline Profiling

Nsight Systems (`nsys`) produces a timeline showing kernel launches, memory
copies, CUDA API calls, and even ROS2 / Python function calls when run
with the Python tracing module.

```bash
# Capture a 30-second timeline of an inference benchmark
nsys profile --trace=cuda,nvtx,osrt \
             --output=/tmp/inference \
             --force-overwrite=true \
             --duration=30 \
             python3 python/benchmark.py --engine /models/yolov5s.engine --iters 200

# Open the .qdrep / .nsys-rep file in Nsight Systems GUI on a host PC
nsys stats /tmp/inference.nsys-rep     # CLI summary tables
```

Typical findings on Jetson:
- H2D copies take 20–40 % of inference time if buffers are not pinned.
  Fix: use `cudaMallocHost` / `cudaHostAlloc` (see `cpp/cuda_memory.cpp::PinnedBuffer`).
- `cudaDeviceSynchronize` is called too often. Use CUDA streams and
  `cudaEventSynchronize` on the last operation only.
- Argus capture and inference are serialized. Use `nvbufsurface` zero-copy
  to hand the captured NV12 buffer straight to the pre-proc kernel.

---

## 5. Nsight Compute — Per-Kernel Profiling

`ncu` (Nsight Compute) gives per-instruction throughput, register pressure,
shared-memory bank conflicts, and warp divergence.

```bash
sudo /usr/local/cuda/bin/ncu --set full \
    --target-processes all \
    --kernel-name regex:softmax \
    -o /tmp/softmax_profile \
    ./kernel
```

Watch for:
- **Achieved occupancy < 50 %** → register pressure. Add `__launch_bounds__(256, 2)`
  to the kernel declaration.
- **Shared memory bank conflicts** → re-pad arrays to stride `33` instead of `32`.
- **Warp execution efficiency < 80 %** → divergent branches; restructure
  data layout (SoA instead of AoS) or warp-ballot primitives
  (`__ballot_sync`, `__popc`).

---

## 6. CUDA Samples

JetPack ships the official CUDA samples under `/usr/local/cuda/samples/`.
Useful ones to build and run on Jetson:

```bash
cd /usr/local/cuda/samples
sudo make -j$(nproc) TARGET_ARCH=armv7l SMS="53 72 87"

# Run a few:
cd bin/aarch64/linux/release
./0_Simple/vectorAdd          # smoke test
./1_Utilities/deviceQuery      # prints compute capability, memory
./2_Graphics/simpleGL         # OpenGL interop (headless: use EGL)
./5_Simulations/particles     # CUDA-GL interop
./6_Advanced/reduction        # warp-shuffle reduction pattern
```

`deviceQuery` is the canonical way to verify that the toolkit sees the GPU:
```
Device 0: "NVIDIA Tegra X1"
  CUDA Capability Major/Minor version number:    5.3
  Total amount of global memory:                  4044 MBytes
  (1) Multiprocessors, (128) CUDA Cores/MP:       128 CUDA Cores
  GPU Max Clock rate:                             922 MHz
```

---

## 7. Mixed-Precision Programming

### FP16
- Use `__half` and `__half2` types from `<cuda_fp16.h>`.
- Hardware conversion is free: `__float2half`, `__half2float`.
- Vectorized loads: `__half2` (32-bit) read/writes coalesce cleanly.

```cuda
#include <cuda_fp16.h>
__global__ void scale_fp16(const __half* __restrict__ in,
                           __half* __restrict__ out,
                           float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 + 1 < n) {
        __half2 v = reinterpret_cast<const __half2*>(in)[idx];
        __half2 a = __float2half2_rn(alpha);
        v = __hmul2(v, a);
        reinterpret_cast<__half2*>(out)[idx] = v;
    }
}
```

### BF16 (Orin only, sm_87+)
```cuda
#include <cuda_bf16.h>
__nv_bfloat16 x = __float2bfloat16(1.0f);
```

### Tensor Cores (Volta+ via `wmma`)
See `cuda/tensor_operations.cu::matmul_wmma` for a worked example using
`nvcuda::wmma::mma_sync` to compute a 16×16×16 FP16 matrix multiply.

---

## 8. Memory Hierarchies on Jetson

Jetson uses a **unified memory architecture** — the CPU and GPU share the
same physical LPDDR4/LPDDR5 chips. This has important consequences:

- `cudaMalloc` and `cudaMallocManaged` both end up in the same DRAM, but
  `cudaMalloc` provides stronger coherency guarantees for streaming workloads.
- `cudaMemcpyAsync` to a `cudaMallocHost`-pinned buffer is the fastest path
  for H↔D copies (≈ 12 GB/s on Xavier NX).
- `cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)`
  can pin a managed buffer to the CPU; useful for ROS2 message objects.

For the InferenceEngine buffers we use:
1. `cudaMalloc` for device tensors (inputs/outputs).
2. `cudaMallocHost` for pinned host staging buffers.
3. `cudaStreamSynchronize` only after a batch of work is queued.

See `cpp/cuda_memory.cpp` for the RAII wrappers `DeviceBuffer`,
`PinnedBuffer`, `UnifiedBuffer`.

---

## 9. Power & Thermal Considerations

CUDA kernels on Jetson are bounded by **thermal** limits long before they
hit compute limits. The GPU frequency drops automatically when the SoC
hits 80 °C. To stay in steady state:

```bash
# Fix the GPU clock and monitor thermal throttling
sudo jetson_clocks --fan
sudo nvpmodel -m 2          # 15 W mode (Xavier NX) — sustainable
watch -n 1 "cat /sys/devices/virtual/thermal/thermal_zone*/temp"
```

In your application, use `jetson-stats` (`python/jtop` API) to read:
- `gpu.power` — instantaneous GPU power draw (W)
- `gpu.temp` — GPU temperature (°C)
- `gpu.val` — current GPU frequency (kHz)
- `thermal.throttling` — `True` when the SoC has throttled

See `python/benchmark.py::BenchmarkSuite.measure_power()` for an example
of sampling these at 10 Hz during a benchmark run.

---

## 10. Debugging Checklist

| Symptom | Likely cause | Tool |
|---------|-------------|------|
| Kernel returns `cudaErrorInvalidConfiguration` | grid/block dim mismatch | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| `cudaErrorLaunchTimeout` | Kernel ran > 2 s (Windows TDR); on Jetson usually a watchdog | Disable X server / use `nvpmodel 0` |
| Random NaNs after long runs | Uninitialized memory or FP overflow | `cuda-memcheck`, `compute-sanitizer` |
| Slow launch latency (µs) | Synchronous mallocs in hot loop | Pre-allocate buffers; use pool allocator |
| Bank conflicts in shared mem | Stride == 32 | `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` |

### compute-sanitizer (replaces cuda-memcheck)
```bash
sudo compute-sanitizer --tool memcheck ./kernel
sudo compute-sanitizer --tool racecheck ./kernel
sudo compute-sanitizer --tool initcheck ./kernel
sudo compute-sanitizer --tool synccheck ./kernel
```
