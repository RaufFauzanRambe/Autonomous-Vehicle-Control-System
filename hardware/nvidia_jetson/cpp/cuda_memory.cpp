// =============================================================================
// File: cpp/cuda_memory.cpp
// Brief: RAII wrappers for CUDA memory — DeviceBuffer (cudaMalloc),
//        PinnedBuffer (cudaMallocHost), UnifiedBuffer (cudaMallocManaged).
//        The header cuda_memory.hpp is included inline at the top of this
//        file so the whole module is self-contained.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
//
// Design goals
// ------------
//   * No raw cudaMalloc / cudaFree calls outside this file.
//   * Exception-safe — destructor never throws.
//   * Movable but not copyable (a CUDA allocation has unique ownership).
//   * Minimal overhead — inlined getters.
//
// Usage
// -----
//   DeviceBuffer buf(1024 * sizeof(float));
//   CUDA_CHECK(cudaMemcpy(buf.get(), host, buf.bytes(),
//                          cudaMemcpyHostToDevice));
//   // ... use buf.get() as a device pointer ...
//   // freed automatically when buf goes out of scope
//
// For asynchronous copies with a CUDA stream, use PinnedBuffer on the host:
//
//   PinnedBuffer host_buf(1024 * sizeof(float));
//   DeviceBuffer dev_buf(1024 * sizeof(float));
//   CUDA_CHECK(cudaMemcpyAsync(dev_buf.get(), host_buf.get(),
//                              dev_buf.bytes(),
//                              cudaMemcpyHostToDevice, stream));
// -------------------------------------------------------------------------

#ifndef AV_JETSON_CPP_CUDA_MEMORY_HPP
#define AV_JETSON_CPP_CUDA_MEMORY_HPP

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace av_jetson {

// ---------------------------------------------------------------------------
// Error-checking macro — used by all wrappers. Throws on CUDA errors.
// ---------------------------------------------------------------------------
#define AV_CUDA_CHECK(expr)                                                 \
  do {                                                                      \
    cudaError_t err__ = (expr);                                             \
    if (err__ != cudaSuccess) {                                             \
      throw std::runtime_error(                                             \
          std::string("CUDA error at ") + __FILE__ + ":" +                  \
          std::to_string(__LINE__) + " — " +                                \
          cudaGetErrorString(err__) + " (" +                                \
          cudaGetErrorName(err__) + ")");                                   \
    }                                                                       \
  } while (0)

// ===========================================================================
// DeviceBuffer — owns cudaMalloc / cudaFree memory on the GPU.
// ===========================================================================
class DeviceBuffer {
 public:
  /// Allocate ``bytes`` bytes of device memory. Throws on failure.
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes), ptr_(nullptr) {
    if (bytes_ == 0) return;
    AV_CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
  }

  ~DeviceBuffer() { release(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : bytes_(other.bytes_), ptr_(other.ptr_) {
    other.bytes_ = 0;
    other.ptr_ = nullptr;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      bytes_ = other.bytes_;
      ptr_ = other.ptr_;
      other.bytes_ = 0;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  /// Resize — frees the old allocation and creates a new one.
  void resize(std::size_t bytes) {
    release();
    bytes_ = bytes;
    if (bytes_ == 0) return;
    AV_CUDA_CHECK(cudaMalloc(&ptr_, bytes_));
  }

  /// Zero the buffer on a stream.
  void zero(cudaStream_t stream = 0) const {
    if (ptr_ && bytes_ > 0) {
      AV_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes_, stream));
    }
  }

  [[nodiscard]] void* get() noexcept { return ptr_; }
  [[nodiscard]] const void* get() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr; }

 private:
  void release() noexcept {
    if (ptr_) {
      cudaFree(ptr_);  // best-effort; ignore errors
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  std::size_t bytes_ = 0;
  void* ptr_ = nullptr;
};

// ===========================================================================
// PinnedBuffer — owns cudaMallocHost / cudaFreeHost page-locked host memory.
// ===========================================================================
class PinnedBuffer {
 public:
  explicit PinnedBuffer(std::size_t bytes) : bytes_(bytes), ptr_(nullptr) {
    if (bytes_ == 0) return;
    AV_CUDA_CHECK(cudaMallocHost(&ptr_, bytes_));
  }

  ~PinnedBuffer() { release(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept
      : bytes_(other.bytes_), ptr_(other.ptr_) {
    other.bytes_ = 0;
    other.ptr_ = nullptr;
  }

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      release();
      bytes_ = other.bytes_;
      ptr_ = other.ptr_;
      other.bytes_ = 0;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  void resize(std::size_t bytes) {
    release();
    bytes_ = bytes;
    if (bytes_ == 0) return;
    AV_CUDA_CHECK(cudaMallocHost(&ptr_, bytes_));
  }

  void zero(cudaStream_t stream = 0) const {
    if (ptr_ && bytes_ > 0) {
      AV_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes_, stream));
    }
  }

  /// Copy from a host range into this pinned buffer (host-to-host).
  void copy_from_host(const void* src, std::size_t bytes) {
    if (bytes > bytes_) {
      throw std::runtime_error("PinnedBuffer::copy_from_host overflow");
    }
    std::memcpy(ptr_, src, bytes);
  }

  [[nodiscard]] void* get() noexcept { return ptr_; }
  [[nodiscard]] const void* get() const noexcept { return ptr_; }
  template <typename T>
  [[nodiscard]] T* as() noexcept {
    return static_cast<T*>(ptr_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr; }

 private:
  void release() noexcept {
    if (ptr_) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  std::size_t bytes_ = 0;
  void* ptr_ = nullptr;
};

// ===========================================================================
// UnifiedBuffer — owns cudaMallocManaged / cudaFree unified memory.
// Unified memory is convenient on Jetson because the CPU and GPU share
// physical DRAM — the pointer is valid from both sides with zero copy.
// ===========================================================================
class UnifiedBuffer {
 public:
  explicit UnifiedBuffer(std::size_t bytes,
                          unsigned int flags = cudaMemAttachGlobal)
      : bytes_(bytes), ptr_(nullptr) {
    if (bytes_ == 0) return;
    AV_CUDA_CHECK(cudaMallocManaged(&ptr_, bytes_, flags));
  }

  ~UnifiedBuffer() { release(); }

  UnifiedBuffer(const UnifiedBuffer&) = delete;
  UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;
  UnifiedBuffer(UnifiedBuffer&&) noexcept = default;
  UnifiedBuffer& operator=(UnifiedBuffer&&) noexcept = default;

  /// Hint the runtime that this buffer is primarily read by the device.
  void advise_device_preferred(int device_id) const {
    if (ptr_) {
      cudaMemAdvise(ptr_, bytes_, cudaMemAdviseSetPreferredLocation,
                    device_id);
      cudaMemAdvise(ptr_, bytes_, cudaMemAdviseSetAccessedBy, device_id);
    }
  }

  /// Prefetch the buffer to the device for the next kernel launch.
  void prefetch(int device_id, cudaStream_t stream = 0) const {
    if (ptr_) {
      AV_CUDA_CHECK(cudaMemPrefetchAsync(ptr_, bytes_, device_id, stream));
    }
  }

  void zero(cudaStream_t stream = 0) const {
    if (ptr_ && bytes_ > 0) {
      AV_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, bytes_, stream));
    }
  }

  [[nodiscard]] void* get() noexcept { return ptr_; }
  [[nodiscard]] const void* get() const noexcept { return ptr_; }
  template <typename T>
  [[nodiscard]] T* as() noexcept {
    return static_cast<T*>(ptr_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return ptr_ == nullptr; }

 private:
  void release() noexcept {
    if (ptr_) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  std::size_t bytes_ = 0;
  void* ptr_ = nullptr;
};

// ===========================================================================
// CudaStream — RAII wrapper around cudaStream_t.
// ===========================================================================
class CudaStream {
 public:
  /// Construct a non-blocking stream by default (recommended for inference).
  explicit CudaStream(unsigned int flags = cudaStreamNonBlocking)
      : stream_(nullptr) {
    AV_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
  }
  ~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
  }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }
  CudaStream& operator=(CudaStream&& other) noexcept {
    if (this != &other) {
      if (stream_) cudaStreamDestroy(stream_);
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  void sync() const { AV_CUDA_CHECK(cudaStreamSynchronize(stream_)); }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] bool is_ready() const {
    return cudaStreamQuery(stream_) == cudaSuccess;
  }

 private:
  cudaStream_t stream_;
};

// ===========================================================================
// CudaEvent — RAII wrapper around cudaEvent_t.
// ===========================================================================
class CudaEvent {
 public:
  explicit CudaEvent(unsigned int flags = cudaEventDefault) {
    AV_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
  }
  ~CudaEvent() {
    if (event_) cudaEventDestroy(event_);
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  CudaEvent(CudaEvent&&) noexcept = default;
  CudaEvent& operator=(CudaEvent&&) noexcept = default;

  void record(cudaStream_t stream = 0) const {
    AV_CUDA_CHECK(cudaEventRecord(event_, stream));
  }
  void sync() const { AV_CUDA_CHECK(cudaEventSynchronize(event_)); }
  [[nodiscard]] float elapsed_ms_since(const CudaEvent& start) const {
    float ms = 0.0f;
    AV_CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
    return ms;
  }
  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

}  // namespace av_jetson

#endif  // AV_JETSON_CPP_CUDA_MEMORY_HPP

// ===========================================================================
// Implementation-only utilities (kept here so the whole module is one TU).
// ===========================================================================
// The header above is the public interface. Below are helpers that compile
// only when this .cpp is built.

#include <cstdio>
#include <string>

namespace av_jetson {

/// Pretty-print the current GPU's memory usage.
/// Useful for debugging "out of memory" errors during engine building.
inline std::string gpu_memory_report() {
  std::size_t free = 0, total = 0;
  if (cudaMemGetInfo(&free, &total) != cudaSuccess) {
    return "cudaMemGetInfo failed";
  }
  char buf[128];
  std::snprintf(buf, sizeof(buf),
                "GPU memory: used=%.1f MB, free=%.1f MB, total=%.1f MB",
                (total - free) / 1048576.0,
                free / 1048576.0,
                total / 1048576.0);
  return std::string(buf);
}

}  // namespace av_jetson
