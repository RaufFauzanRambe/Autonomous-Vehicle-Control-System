// =============================================================================
// File: cpp/tensor_rt.cpp
// Brief: Implementation of TensorRTBuilder — ONNX → TRT engine with
//        dynamic shape profiles, FP16/INT8 precision, and DLA offload.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
#include "tensor_rt.hpp"

#include "inference_engine.hpp"  // TrtLogger

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace av_jetson {

// ---------------------------------------------------------------------------
// Custom deleter for nvinfer1 objects — they all expose destroy().
// ---------------------------------------------------------------------------
namespace {
struct TrtDeleter {
  template <typename T>
  void operator()(T* p) const {
    if (p) p->destroy();
  }
};
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

// Minimal IInt8Calibrator that returns random batches — replace with a
// real dataset for production. Implemented as a header-only class to keep
// the build simple.
class DummyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  DummyInt8Calibrator(int batch_size, std::size_t input_bytes,
                      const std::string& cache_file)
      : batch_size_(batch_size),
        input_bytes_(input_bytes),
        cache_file_(cache_file) {
    CUDA_CHECK(cudaMalloc(&device_ptr_, input_bytes_));
    host_buffer_.resize(input_bytes_);
    std::fill(host_buffer_.begin(), host_buffer_.end(), 0);
  }
  ~DummyInt8Calibrator() override {
    if (device_ptr_) cudaFree(device_ptr_);
  }

  int getBatchSize() const noexcept override { return batch_size_; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override {
    if (call_count_ >= max_batches_) return false;
    // Fill with a deterministic pattern so the cache is reproducible.
    for (std::size_t i = 0; i < host_buffer_.size(); ++i) {
      host_buffer_[i] = static_cast<std::uint8_t>((i + call_count_) & 0xFF);
    }
    CUDA_CHECK(cudaMemcpy(device_ptr_, host_buffer_.data(),
                           input_bytes_, cudaMemcpyHostToDevice));
    bindings[0] = device_ptr_;
    ++call_count_;
    return true;
  }

  const void* readCalibrationCache(std::size_t& length) noexcept override {
    if (cache_data_.empty()) {
      std::ifstream f(cache_file_, std::ios::binary);
      if (f) {
        f.seekg(0, std::ios::end);
        length = static_cast<std::size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        cache_data_.resize(length);
        f.read(reinterpret_cast<char*>(cache_data_.data()), length);
      } else {
        length = 0;
        return nullptr;
      }
    } else {
      length = cache_data_.size();
    }
    return cache_data_.data();
  }

  void writeCalibrationCache(const void* ptr,
                              std::size_t length) noexcept override {
    std::ofstream f(cache_file_, std::ios::binary);
    if (f) f.write(reinterpret_cast<const char*>(ptr), length);
    cache_data_.assign(static_cast<const std::uint8_t*>(ptr),
                       static_cast<const std::uint8_t*>(ptr) + length);
  }

 private:
  int batch_size_;
  std::size_t input_bytes_;
  std::string cache_file_;
  void* device_ptr_ = nullptr;
  std::vector<std::uint8_t> host_buffer_;
  std::vector<std::uint8_t> cache_data_;
  int call_count_ = 0;
  static constexpr int max_batches_ = 64;
};
}  // namespace

// ---------------------------------------------------------------------------
// TensorRTBuilder::Impl — pimpl to hide TRT headers from the public API.
// ---------------------------------------------------------------------------
struct TensorRTBuilder::Impl {
  TrtLogger logger;
  TrtUniquePtr<nvinfer1::IBuilder> builder;
  TrtUniquePtr<nvinfer1::INetworkDefinition> network;
  TrtUniquePtr<nvinfer1::IBuilderConfig> config;
  TrtUniquePtr<nvonnxparser::IParser> parser;

  explicit Impl(TrtLogger::Severity v) : logger(v) {}
};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
TensorRTBuilder::TensorRTBuilder(TrtLogger::Severity verbosity)
    : pimpl_(std::make_unique<Impl>(verbosity)) {}

TensorRTBuilder::~TensorRTBuilder() = default;

// ---------------------------------------------------------------------------
// build / build_serialized
// ---------------------------------------------------------------------------
void TensorRTBuilder::build(const BuildOptions& opts) {
  auto plan = build_serialized(opts);
  save(plan, opts.output_engine_path);
}

std::vector<std::uint8_t> TensorRTBuilder::build_serialized(
    const BuildOptions& opts) {
  pimpl_->builder.reset(nvinfer1::createInferBuilder(pimpl_->logger));
  if (!pimpl_->builder) {
    throw std::runtime_error("createInferBuilder failed");
  }
  pimpl_->network.reset(pimpl_->builder->createNetworkV2(
      1U << static_cast<unsigned int>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
  if (!pimpl_->network) {
    throw std::runtime_error("createNetworkV2 failed");
  }
  pimpl_->parser.reset(nvonnxparser::createParser(
      *pimpl_->network, pimpl_->logger));
  if (!pimpl_->parser) {
    throw std::runtime_error("createParser failed");
  }
  pimpl_->config.reset(pimpl_->builder->createBuilderConfig());
  if (!pimpl_->config) {
    throw std::runtime_error("createBuilderConfig failed");
  }

  // Read and parse the ONNX file.
  std::ifstream f(opts.onnx_path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Cannot open ONNX file: " + opts.onnx_path);
  }
  auto size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<char> onnx_buf(static_cast<std::size_t>(size));
  if (!f.read(onnx_buf.data(), size)) {
    throw std::runtime_error("Failed to read ONNX file: " + opts.onnx_path);
  }
  if (!pimpl_->parser->parse(onnx_buf.data(), onnx_buf.size())) {
    for (int i = 0; i < pimpl_->parser->getNbErrors(); ++i) {
      auto* err = pimpl_->parser->getError(i);
      pimpl_->logger.log(
          nvinfer1::ILogger::Severity::kERROR, err->desc());
    }
    throw std::runtime_error("ONNX parse failed for " + opts.onnx_path);
  }

  // Workspace.
  pimpl_->config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE,
      opts.workspace_gb * (1ULL << 30));

  // Precision flags.
  if (opts.fp16) {
    pimpl_->config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (opts.int8) {
    pimpl_->config->setFlag(nvinfer1::BuilderFlag::kINT8);
    // Attach a calibrator (dummy if no cache file is provided).
    // For production, replace with a real calibrator that streams a
    // representative dataset.
    auto first_input = pimpl_->network->getInput(0);
    std::size_t input_bytes = 4;  // sizeof(float)
    for (int d = 0; d < first_input->getDimensions().nbDims; ++d) {
      int v = first_input->getDimensions().d[d];
      if (v == -1) v = 1;
      input_bytes *= static_cast<std::size_t>(v);
    }
    pimpl_->config->setInt8Calibrator(new DummyInt8Calibrator(
        1, input_bytes, opts.int8_calib_cache));
  }
  if (opts.strict_types) {
    pimpl_->config->setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
  }

  // DLA offload.
  if (opts.use_dla) {
    pimpl_->config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
    pimpl_->config->setDLACore(opts.dla_core);
    if (opts.gpu_fallback) {
      pimpl_->config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
    }
  }

  // Optimization profiles for dynamic shapes.
  if (!opts.profiles.empty()) {
    for (const auto& [tensor_name, range] : opts.profiles) {
      auto* profile = pimpl_->builder->createOptimizationProfile();
      nvinfer1::Dims min_d, opt_d, max_d;
      min_d.nbDims = static_cast<int>(range.min.size());
      opt_d.nbDims = static_cast<int>(range.opt.size());
      max_d.nbDims = static_cast<int>(range.max.size());
      for (size_t i = 0; i < range.min.size(); ++i) {
        min_d.d[i] = range.min[i];
        opt_d.d[i] = range.opt[i];
        max_d.d[i] = range.max[i];
      }
      profile->setDimensions(tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kMIN, min_d);
      profile->setDimensions(tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kOPT, opt_d);
      profile->setDimensions(tensor_name.c_str(),
                              nvinfer1::OptProfileSelector::kMAX, max_d);
      pimpl_->config->addOptimizationProfile(profile);
    }
  }

  // Build (this is the slow step).
  auto* plan = pimpl_->builder->buildSerializedNetwork(
      *pimpl_->network, *pimpl_->config);
  if (!plan) {
    throw std::runtime_error(
        "buildSerializedNetwork returned nullptr — check the TRT log.");
  }
  std::vector<std::uint8_t> out(
      static_cast<const std::uint8_t*>(plan->data()),
      static_cast<const std::uint8_t*>(plan->data()) + plan->size());
  plan->destroy();
  return out;
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------
void TensorRTBuilder::save(const std::vector<std::uint8_t>& plan,
                            const std::string& path) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Cannot open output engine file: " + path);
  }
  f.write(reinterpret_cast<const char*>(plan.data()),
          static_cast<std::streamsize>(plan.size()));
}

std::vector<std::uint8_t> TensorRTBuilder::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    throw std::runtime_error("Cannot open engine file: " + path);
  }
  auto size = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> out(static_cast<std::size_t>(size));
  f.read(reinterpret_cast<char*>(out.data()), size);
  return out;
}

}  // namespace av_jetson
