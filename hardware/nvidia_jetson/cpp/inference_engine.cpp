// =============================================================================
// File: cpp/inference_engine.cpp
// Brief: Implementation of InferenceEngine — async TensorRT inference on
//        Jetson using RAII smart pointers and CUDA streams.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
#include "inference_engine.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace av_jetson {

namespace {

// ---------------------------------------------------------------------------
// Custom deleter for cudaStream / cudaEvent — std::unique_ptr friendly.
// ---------------------------------------------------------------------------
struct CudaStreamDeleter {
  void operator()(cudaStream_t s) const {
    if (s) cudaStreamDestroy(s);
  }
};
struct CudaEventDeleter {
  void operator()(cudaEvent_t e) const {
    if (e) cudaEventDestroy(e);
  }
};

// ---------------------------------------------------------------------------
// SHA-256 of a file using a minimal inline implementation. We avoid
// pulling in OpenSSL here so the build is portable; for production use
// prefer a real crypto library. Returns the first 16 hex chars.
// ---------------------------------------------------------------------------
std::string short_file_hash(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "0000000000000000";
  std::uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
  std::uint64_t prime = 0x100000001b3ULL;
  char buf[1 << 16];
  while (f) {
    f.read(buf, sizeof(buf));
    auto g = f.gcount();
    for (std::streamsize i = 0; i < g; ++i) {
      h ^= static_cast<std::uint8_t>(buf[i]);
      h *= prime;
    }
  }
  std::ostringstream os;
  os << std::hex << std::setw(16) << std::setfill('0') << h;
  return os.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// TrtLogger
// ---------------------------------------------------------------------------
void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity > verbosity_) return;
  const char* tag = "INFO";
  switch (severity) {
    case Severity::kINTERNAL_ERROR: tag = "INTERNAL_ERROR"; break;
    case Severity::kERROR:           tag = "ERROR";          break;
    case Severity::kWARNING:         tag = "WARNING";        break;
    case Severity::kINFO:            tag = "INFO";           break;
    case Severity::kVERBOSE:         tag = "VERBOSE";        break;
  }
  std::fprintf(stderr, "[TRT][%s] %s\n", tag, msg);
}

// ---------------------------------------------------------------------------
// InferenceEngine — construction
// ---------------------------------------------------------------------------
InferenceEngine::InferenceEngine(const std::string& engine_path,
                                   int device_id,
                                   TrtLogger::Severity verbosity)
    : device_id_(device_id), logger_(verbosity) {
  CUDA_CHECK(cudaSetDevice(device_id_));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(
      &start_event_, cudaEventBlockingSync));
  CUDA_CHECK(cudaEventCreateWithFlags(
      &end_event_, cudaEventBlockingSync));

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    throw std::runtime_error("createInferRuntime returned nullptr");
  }
  load_engine_(engine_path);
  analyze_bindings_();
  allocate_buffers_();
}

InferenceEngine::~InferenceEngine() {
  // Release TRT objects first (they reference CUDA memory indirectly),
  // then free device buffers, then destroy the stream.
  context_.reset();
  engine_.reset();
  runtime_.reset();
  device_buffers_.clear();
  if (stream_) cudaStreamDestroy(stream_);
  if (start_event_) cudaEventDestroy(start_event_);
  if (end_event_) cudaEventDestroy(end_event_);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
void InferenceEngine::load_engine_(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Cannot open engine file: " + path);
  }
  auto size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<std::size_t>(size));
  if (!f.read(data.data(), size)) {
    throw std::runtime_error("Failed to read engine file: " + path);
  }
  engine_hash_ = short_file_hash(path);

  nvinfer1::ICudaEngine* raw = runtime_->deserializeCudaEngine(
      data.data(), data.size());
  if (!raw) {
    throw std::runtime_error(
        "Failed to deserialize engine. Possible TensorRT version mismatch "
        "or corrupted plan file: " + path);
  }
  engine_.reset(raw);

  nvinfer1::IExecutionContext* ctx = engine_->createExecutionContext();
  if (!ctx) {
    throw std::runtime_error("createExecutionContext returned nullptr");
  }
  context_.reset(ctx);
}

// ---------------------------------------------------------------------------
// Binding analysis
// ---------------------------------------------------------------------------
void InferenceEngine::analyze_bindings_() {
  // Support both the TRT 8.x (num_bindings) and TRT 10.x
  // (num_io_tensors + getTensorName) APIs.
  int n = 0;
  if (engine_->getNbIOTensors) {
    // TRT 10+ path.
    n = engine_->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
      const char* name = engine_->getIOTensorName(i);
      BindingInfo b;
      b.name = name;
      b.is_input = (engine_->getTensorIOMode(name) ==
                    nvinfer1::TensorIOMode::kINPUT);
      b.dtype = engine_->getTensorDataType(name);
      b.dims = engine_->getTensorShape(name);
      b.volume = 1;
      for (int d = 0; d < b.dims.nbDims; ++d) {
        int v = b.dims.d[d];
        if (v == -1) v = 1;  // dynamic — pre-allocate size 1
        b.volume *= static_cast<std::size_t>(v);
      }
      b.bytes = b.volume * dtype_size_(b.dtype);
      if (b.is_input) {
        for (int d = 0; d < b.dims.nbDims; ++d) {
          if (b.dims.d[d] == -1) has_dynamic_shapes_ = true;
        }
      }
      binding_index_[b.name] = bindings_.size();
      bindings_.push_back(std::move(b));
    }
  } else {
    // TRT 8.x path.
    n = engine_->getNbBindings();
    for (int i = 0; i < n; ++i) {
      BindingInfo b;
      b.name = engine_->getBindingName(i);
      b.is_input = engine_->bindingIsInput(i);
      b.dtype = engine_->getBindingDataType(i);
      b.dims = engine_->getBindingDimensions(i);
      b.volume = 1;
      for (int d = 0; d < b.dims.nbDims; ++d) {
        int v = b.dims.d[d];
        if (v == -1) v = 1;
        b.volume *= static_cast<std::size_t>(v);
      }
      b.bytes = b.volume * dtype_size_(b.dtype);
      if (b.is_input) {
        for (int d = 0; d < b.dims.nbDims; ++d) {
          if (b.dims.d[d] == -1) has_dynamic_shapes_ = true;
        }
      }
      binding_index_[b.name] = bindings_.size();
      bindings_.push_back(std::move(b));
    }
  }
}

void InferenceEngine::allocate_buffers_() {
  device_buffers_.clear();
  device_ptrs_.clear();
  total_device_bytes_ = 0;
  device_buffers_.reserve(bindings_.size());
  device_ptrs_.reserve(bindings_.size());
  for (auto& b : bindings_) {
    auto buf = std::make_unique<DeviceBuffer>(b.bytes);
    device_ptrs_.push_back(buf->get());
    total_device_bytes_ += b.bytes;
    device_buffers_.push_back(std::move(buf));
  }
}

// ---------------------------------------------------------------------------
// Inspection
// ---------------------------------------------------------------------------
std::vector<std::string> InferenceEngine::input_names() const {
  std::vector<std::string> out;
  for (const auto& b : bindings_) {
    if (b.is_input) out.push_back(b.name);
  }
  return out;
}

std::vector<std::string> InferenceEngine::output_names() const {
  std::vector<std::string> out;
  for (const auto& b : bindings_) {
    if (!b.is_input) out.push_back(b.name);
  }
  return out;
}

const BindingInfo& InferenceEngine::binding(const std::string& name) const {
  auto it = binding_index_.find(name);
  if (it == binding_index_.end()) {
    throw std::out_of_range("Unknown binding: " + name);
  }
  return bindings_[it->second];
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
void InferenceEngine::set_input(const std::string& name,
                                  const void* host_data) {
  const auto& b = binding(name);
  if (!b.is_input) {
    throw std::invalid_argument(name + " is not an input binding");
  }
  auto& buf = device_buffers_[binding_index_[name]];
  CUDA_CHECK(cudaMemcpyAsync(buf->get(), host_data, b.bytes,
                              cudaMemcpyHostToDevice, stream_));
}

void InferenceEngine::set_input_device(const std::string& name,
                                         const void* device_data) {
  const auto& b = binding(name);
  if (!b.is_input) {
    throw std::invalid_argument(name + " is not an input binding");
  }
  auto& buf = device_buffers_[binding_index_[name]];
  CUDA_CHECK(cudaMemcpyAsync(buf->get(), device_data, b.bytes,
                              cudaMemcpyDeviceToDevice, stream_));
}

void InferenceEngine::set_input_shape(const std::string& name,
                                        const nvinfer1::Dims& shape) {
  auto it = binding_index_.find(name);
  if (it == binding_index_.end()) {
    throw std::out_of_range("Unknown binding: " + name);
  }
  if (!bindings_[it->second].is_input) {
    throw std::invalid_argument(name + " is not an input binding");
  }
#if NV_TENSORRT_MAJOR >= 10
  context_->setInputShape(name.c_str(), shape);
#else
  context_->setBindingShape(static_cast<int>(it->second), shape);
#endif
  // Update cached volume / bytes so set_input copies the right amount.
  auto& b = bindings_[it->second];
  b.dims = shape;
  b.volume = 1;
  for (int d = 0; d < b.dims.nbDims; ++d) {
    b.volume *= static_cast<std::size_t>(std::max(1, b.dims.d[d]));
  }
  b.bytes = b.volume * dtype_size_(b.dtype);
  // Reallocate device buffer to fit.
  device_buffers_[it->second] = std::make_unique<DeviceBuffer>(b.bytes);
  device_ptrs_[it->second] = device_buffers_[it->second]->get();
}

void InferenceEngine::enqueue() {
  CUDA_CHECK(cudaEventRecord(start_event_, stream_));
  bool ok = false;
#if NV_TENSORRT_MAJOR >= 10
  // TRT 10+: set tensor addresses, then call executeAsyncV3.
  for (std::size_t i = 0; i < bindings_.size(); ++i) {
    context_->setTensorAddress(bindings_[i].name.c_str(),
                                device_ptrs_[i]);
  }
  ok = context_->enqueueV3(stream_);
#else
  // TRT 8.x: pass a void** array.
  ok = context_->enqueueV2(device_ptrs_.data(), stream_, nullptr);
#endif
  if (!ok) {
    throw std::runtime_error("enqueueV2/V3 returned false");
  }
  CUDA_CHECK(cudaEventRecord(end_event_, stream_));
}

void InferenceEngine::sync() {
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  CUDA_CHECK(cudaEventElapsedTime(&last_time_ms_,
                                    start_event_, end_event_));
}

void InferenceEngine::copy_output(const std::string& name, void* host_data) {
  const auto& b = binding(name);
  if (b.is_input) {
    throw std::invalid_argument(name + " is not an output binding");
  }
  auto& buf = device_buffers_[binding_index_[name]];
  CUDA_CHECK(cudaMemcpyAsync(host_data, buf->get(), b.bytes,
                              cudaMemcpyDeviceToHost, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void InferenceEngine::infer(const std::string& input_name,
                              const void* host_in,
                              const std::string& output_name,
                              void* host_out) {
  set_input(input_name, host_in);
  enqueue();
  sync();
  copy_output(output_name, host_out);
}

// ---------------------------------------------------------------------------
// dtype_size_
// ---------------------------------------------------------------------------
std::size_t InferenceEngine::dtype_size_(nvinfer1::DataType dt) noexcept {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    case nvinfer1::DataType::kUINT8: return 1;
#ifdef NV_TENSORRT_MAJOR
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kFP8:   return 1;
#endif
#endif
    default:                         return 4;
  }
}

}  // namespace av_jetson
