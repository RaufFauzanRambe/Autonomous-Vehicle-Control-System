// =============================================================================
// File: cuda/cuda_utils.cuh
// Brief: Shared CUDA utilities for the AV Jetson stack — error-checking
//        macros, timing helpers, warp primitives, and a small set of math
//        intrinsics used across kernels.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
//
// This header is included by every .cu file in the cuda/ directory.
// It is header-only (no .cpp companion) to keep the build graph flat.
// -------------------------------------------------------------------------

#ifndef AV_JETSON_CUDA_CUDA_UTILS_CUH
#define AV_JETSON_CUDA_CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

// =============================================================================
// Error-checking macros
// =============================================================================

/// CHECK_CUDA — wraps a CUDA Runtime call and aborts with a useful message
/// on failure. Used in host code.
#define CHECK_CUDA(expr)                                                      \
  do {                                                                        \
    cudaError_t __err = (expr);                                               \
    if (__err != cudaSuccess) {                                               \
      std::fprintf(stderr,                                                    \
                   "CUDA error %s:%d: %s (%s)\n",                             \
                   __FILE__, __LINE__,                                        \
                   cudaGetErrorString(__err),                                 \
                   cudaGetErrorName(__err));                                  \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

/// CHECK_CUDA_THROW — like CHECK_CUDA but throws std::runtime_error.
/// Prefer this in library code that must be exception-safe.
#define CHECK_CUDA_THROW(expr)                                                \
  do {                                                                        \
    cudaError_t __err = (expr);                                               \
    if (__err != cudaSuccess) {                                               \
      throw std::runtime_error(                                               \
          std::string("CUDA error at ") + __FILE__ + ":" +                    \
          std::to_string(__LINE__) + " — " +                                  \
          cudaGetErrorString(__err) + " (" +                                  \
          cudaGetErrorName(__err) + ")");                                     \
    }                                                                         \
  } while (0)

/// CUDA_CHECK — alias used by inference_engine.cpp. Same as CHECK_CUDA.
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) CHECK_CUDA(expr)
#endif

/// CHECK_TRT — wraps a TensorRT call that returns bool.
#define CHECK_TRT(expr)                                                       \
  do {                                                                        \
    bool __ok = (expr);                                                       \
    if (!__ok) {                                                              \
      std::fprintf(stderr,                                                    \
                   "TensorRT error %s:%d: %s returned false\n",               \
                   __FILE__, __LINE__, #expr);                                \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

/// CHECK_LAST — check the last CUDA error (for kernel launches that don't
/// return an error code directly).
#define CHECK_LAST() CHECK_CUDA(cudaGetLastError())

// =============================================================================
// Timing helpers
// =============================================================================

/// CudaTimer — RAII wrapper around cudaEvent_t for measuring kernel time.
/// Use start() before the kernel and stop_ms() after.
class CudaTimer {
 public:
  CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer&) = delete;
  CudaTimer& operator=(const CudaTimer&) = delete;

  void start(cudaStream_t stream = 0) const {
    cudaEventRecord(start_, stream);
  }

  /// Record stop event on the given stream, synchronize, and return the
  /// elapsed time in milliseconds.
  float stop_ms(cudaStream_t stream = 0) const {
    cudaEventRecord(stop_, stream);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    return ms;
  }

 private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

/// ScopedTimer — RAII timer that prints the elapsed time when destroyed.
class ScopedTimer {
 public:
  explicit ScopedTimer(const char* label, cudaStream_t stream = 0)
      : label_(label), stream_(stream) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_, stream_);
  }
  ~ScopedTimer() {
    cudaEventRecord(stop_, stream_);
    cudaEventSynchronize(stop_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_, stop_);
    std::printf("[timer] %s: %.3f ms\n", label_, ms);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  const char* label_;
  cudaStream_t stream_;
  cudaEvent_t start_, stop_;
};

// =============================================================================
// Warp primitives (sm_60+). These are device-only.
// =============================================================================

/// Warp-wide sum reduction using __shfl_down_sync. Works on any sm_60+.
__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

/// Warp-wide max reduction.
__device__ __forceinline__ float warp_reduce_max(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, val, offset);
    val = (other > val) ? other : val;
  }
  return val;
}

/// Warp-wide min reduction.
__device__ __forceinline__ float warp_reduce_min(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffff, val, offset);
    val = (other < val) ? other : val;
  }
  return val;
}

/// Warp-wide all predicate (true iff every lane in the warp returns pred).
__device__ __forceinline__ bool warp_all(bool pred) {
  return __all_sync(0xffffffff, pred);
}

/// Warp-wide any predicate.
__device__ __forceinline__ bool warp_any(bool pred) {
  return __any_sync(0xffffffff, pred);
}

/// Warp-wide ballot — returns a 32-bit mask of lanes where pred is true.
__device__ __forceinline__ unsigned int warp_ballot(bool pred) {
  return __ballot_sync(0xffffffff, pred);
}

// =============================================================================
// Block-level reduction helpers (sm_60+). Use shared memory + warp reduce.
// =============================================================================

/// Block-wide sum reduction. ``val`` is each thread's contribution.
/// After the call, thread 0 of the block holds the result. ``shared`` must
/// be at least ``blockDim.x * sizeof(float)`` bytes.
__device__ __forceinline__ void block_reduce_sum(float* shared, float val) {
  int tid = threadIdx.x;
  int bdim = blockDim.x;

  // Reduce within each warp.
  val = warp_reduce_sum(val);
  if (tid % 32 == 0) shared[tid / 32] = val;
  __syncthreads();

  // Reduce the warp partial sums.
  int num_warps = bdim / 32;
  if (tid < num_warps) {
    val = shared[tid];
    val = warp_reduce_sum(val);
    if (tid == 0) shared[0] = val;
  }
  __syncthreads();
}

// =============================================================================
// Half-precision helpers
// =============================================================================

/// Convert a float to __half (alias for __float2half — defined for clarity).
__device__ __forceinline__ __half to_half(float x) {
  return __float2half(x);
}

/// Convert a __half to float.
__device__ __forceinline__ float to_float(__half x) {
  return __half2float(x);
}

/// Vectorized half add (two halfs at once).
__device__ __forceinline__ __half2 hadd2(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

// =============================================================================
// Miscellaneous math helpers
// =============================================================================

/// Fast approximate expf — uses the CUDA __expf intrinsic when available.
__device__ __forceinline__ float fast_expf(float x) {
  return __expf(x);
}

/// Fast approximate 1/sqrtf(x).
__device__ __forceinline__ float fast_rsqrtf(float x) {
  return __frsqrt_rn(x);
}

/// Clamp a value to [lo, hi].
template <typename T>
__device__ __forceinline__ T clamp_val(T x, T lo, T hi) {
  return (x < lo) ? lo : ((x > hi) ? hi : x);
}

/// Divide a / b safely (returns 0 when b == 0).
__device__ __forceinline__ float safe_div(float a, float b) {
  return (b != 0.0f) ? (a / b) : 0.0f;
}

// =============================================================================
// Grid-stride loop helper — preferred over manual index math for kernels
// that process arbitrary-length 1D arrays.
// =============================================================================

/// Run ``body`` for every index in ``[0, n)``. The body is a lambda that
/// takes a single ``int`` argument. Each thread strides by blockDim.x *
/// gridDim.x so coalescing is preserved.
template <typename Body>
__device__ __forceinline__ void grid_stride_loop(int n, Body body) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    body(i);
  }
}

// =============================================================================
// Alignment helpers
// =============================================================================

/// Round up to the nearest multiple of ``alignment``.
__host__ __device__ __forceinline__ size_t align_up(size_t v,
                                                     size_t alignment) {
  return (v + alignment - 1) & ~(alignment - 1);
}

/// Check whether ``v`` is a multiple of ``alignment``.
__host__ __device__ __forceinline__ bool is_aligned(size_t v,
                                                     size_t alignment) {
  return (v & (alignment - 1)) == 0;
}

#endif  // AV_JETSON_CUDA_CUDA_UTILS_CUH
