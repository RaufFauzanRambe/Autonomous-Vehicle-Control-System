// =============================================================================
// File: cpp/inference_engine.hpp
// Brief: Declaration of InferenceEngine — a RAII wrapper around the
//        TensorRT C++ API for async inference on Jetson. Manages CUDA
//        streams, device buffers, and execution contexts with smart
//        pointers so there are no manual cudaFree / nvinfer1 deletes.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
#ifndef AV_JETSON_CPP_INFERENCE_ENGINE_HPP
#define AV_JETSON_CPP_INFERENCE_ENGINE_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_memory.hpp"  // DeviceBuffer / PinnedBuffer

namespace av_jetson {

// ---------------------------------------------------------------------------
// A simple TRT logger that funnels messages through std::cerr (or a custom
// callback). Implementing nvinfer1::ILogger is required for every TRT
// runtime object.
// ---------------------------------------------------------------------------
class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}

  void log(Severity severity, const char* msg) noexcept override;

  void set_verbosity(Severity v) noexcept { verbosity_ = v; }

 private:
  Severity verbosity_;
};

// ---------------------------------------------------------------------------
// Describes a single I/O tensor binding of the engine.
// ---------------------------------------------------------------------------
struct BindingInfo {
  std::string name;
  nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  nvinfer1::Dims dims{};
  bool is_input = false;
  std::size_t volume = 0;       // number of elements (product of dims)
  std::size_t bytes = 0;        // volume * sizeof(dtype)
};

// ---------------------------------------------------------------------------
// InferenceEngine — async TensorRT inference, RAII-managed.
//
//   InferenceEngine eng("/models/yolov5s.engine");
//   eng.set_input("images", host_input.data());
//   eng.enqueue();
//   eng.sync();
//   eng.copy_output("output0", host_output.data());
//
// All device memory is owned by the engine and freed on destruction.
// Multiple InferenceEngine instances can coexist on different CUDA streams.
// ---------------------------------------------------------------------------
class InferenceEngine {
 public:
  /// Build an engine from a serialized plan file.
  /// @throws std::runtime_error if the file cannot be opened or the plan
  ///         cannot be deserialized (e.g. TensorRT version mismatch).
  explicit InferenceEngine(const std::string& engine_path,
                           int device_id = 0,
                           TrtLogger::Severity verbosity =
                               TrtLogger::Severity::kWARNING);

  ~InferenceEngine();

  InferenceEngine(const InferenceEngine&) = delete;
  InferenceEngine& operator=(const InferenceEngine&) = delete;
  InferenceEngine(InferenceEngine&&) noexcept = default;
  InferenceEngine& operator=(InferenceEngine&&) noexcept = default;

  // --- Inspection -----------------------------------------------------------
  /// Names of all input bindings.
  [[nodiscard]] std::vector<std::string> input_names() const;

  /// Names of all output bindings.
  [[nodiscard]] std::vector<std::string> output_names() const;

  /// Metadata for a binding by name. Throws std::out_of_range if missing.
  [[nodiscard]] const BindingInfo& binding(const std::string& name) const;

  /// Total device memory held by this engine (bytes).
  [[nodiscard]] std::size_t device_memory_bytes() const noexcept {
    return total_device_bytes_;
  }

  /// Short hash of the serialized plan file (first 16 hex chars).
  [[nodiscard]] std::string engine_hash() const noexcept {
    return engine_hash_;
  }

  // --- Inference ------------------------------------------------------------
  /// Copy host data into a named input binding. The host buffer must be at
  /// least ``binding(name).bytes`` bytes long.
  /// @throws std::runtime_error on CUDA errors.
  void set_input(const std::string& name, const void* host_data);

  /// Overload that takes a device pointer directly (zero-copy).
  void set_input_device(const std::string& name, const void* device_data);

  /// Update a dynamic input shape. Must be called before enqueue() if the
  /// engine was built with dynamic dimensions.
  void set_input_shape(const std::string& name,
                       const nvinfer1::Dims& shape);

  /// Enqueue an inference on the internal CUDA stream. Returns immediately.
  /// @throws std::runtime_error on TRT or CUDA errors.
  void enqueue();

  /// Block until the most recent enqueue() completes.
  void sync();

  /// Copy a named output binding back to the host.
  /// Must be called after sync().
  void copy_output(const std::string& name, void* host_data);

  /// Convenience: run set_input + enqueue + sync + copy_output for a
  /// single input/output pair (synchronous path).
  void infer(const std::string& input_name, const void* host_in,
              const std::string& output_name, void* host_out);

  // --- Benchmarking hooks ---------------------------------------------------
  /// Return the GPU time elapsed between two CUDA events recorded around
  /// the last enqueue()/sync() pair (in milliseconds).
  [[nodiscard]] float last_inference_time_ms() const noexcept {
    return last_time_ms_;
  }

 private:
  // Custom deleters for unique_ptr to TRT opaque types.
  struct TrtDeleter {
    void operator()(nvinfer1::IRuntime* p) const { if (p) p->destroy(); }
    void operator()(nvinfer1::ICudaEngine* p) const { if (p) p->destroy(); }
    void operator()(nvinfer1::IExecutionContext* p) const {
      if (p) p->destroy();
    }
  };
  using TrtRuntimePtr = std::unique_ptr<nvinfer1::IRuntime, TrtDeleter>;
  using TrtEnginePtr = std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter>;
  using TrtContextPtr =
      std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter>;

  void load_engine_(const std::string& path);
  void analyze_bindings_();
  void allocate_buffers_();
  static std::size_t dtype_size_(nvinfer1::DataType dt) noexcept;

  // Members.
  int device_id_ = 0;
  TrtLogger logger_;
  TrtRuntimePtr runtime_{nullptr};
  TrtEnginePtr engine_{nullptr};
  TrtContextPtr context_{nullptr};

  std::vector<BindingInfo> bindings_;
  std::unordered_map<std::string, std::size_t> binding_index_;

  // One DeviceBuffer per binding; inputs and outputs interleaved in
  // binding order to match the TRT buffer array layout.
  std::vector<std::unique_ptr<DeviceBuffer>> device_buffers_;
  std::vector<void*> device_ptrs_;  // raw view used by enqueueV2
  std::size_t total_device_bytes_ = 0;

  // CUDA stream + events for async / timing.
  cudaStream_t stream_ = nullptr;
  cudaEvent_t start_event_ = nullptr;
  cudaEvent_t end_event_ = nullptr;
  float last_time_ms_ = 0.0f;

  std::string engine_hash_;

  // True when at least one binding has a dynamic dimension.
  bool has_dynamic_shapes_ = false;
};

}  // namespace av_jetson

#endif  // AV_JETSON_CPP_INFERENCE_ENGINE_HPP
