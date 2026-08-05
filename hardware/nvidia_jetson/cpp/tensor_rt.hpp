// =============================================================================
// File: cpp/tensor_rt.hpp
// Brief: TensorRT wrapper — logger, builder, network definition, and an
//        ONNX parser front-end. Provides a single TensorRTBuilder class
//        that hides the nvinfer1::IBuilder / IBuilderConfig / INetworkDefinition
//        lifecycle behind RAII smart pointers.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
#ifndef AV_JETSON_CPP_TENSOR_RT_HPP
#define AV_JETSON_CPP_TENSOR_RT_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace av_jetson {

// Reuse the logger from inference_engine.hpp.
class TrtLogger;

// ---------------------------------------------------------------------------
// BuildOptions — fully configurable engine build parameters.
// ---------------------------------------------------------------------------
struct BuildOptions {
  std::string onnx_path;
  std::string output_engine_path;
  bool fp16 = true;
  bool int8 = false;
  bool strict_types = false;
  std::size_t workspace_gb = 4;

  // Dynamic shape profile: per-input {min, opt, max}.
  struct ShapeRange {
    std::vector<int> min;
    std::vector<int> opt;
    std::vector<int> max;
  };
  std::unordered_map<std::string, ShapeRange> profiles;

  // INT8 calibration cache path (binary file).
  std::string int8_calib_cache;

  // DLA offload (Xavier/Orin only).
  bool use_dla = false;
  int dla_core = 0;
  bool gpu_fallback = true;
};

// ---------------------------------------------------------------------------
// TensorRTBuilder — builds and serializes a TensorRT engine from ONNX.
//
//   BuildOptions opts;
//   opts.onnx_path = "/models/yolov5s.onnx";
//   opts.output_engine_path = "/models/yolov5s.engine";
//   opts.fp16 = true;
//   TensorRTBuilder builder;
//   builder.build(opts);
//
// All TRT objects (builder, config, network, parser) are owned by the
// builder and released on destruction. The serialized plan file is the
// only state that escapes the build.
// ---------------------------------------------------------------------------
class TensorRTBuilder {
 public:
  explicit TensorRTBuilder(TrtLogger::Severity verbosity =
                               TrtLogger::Severity::kWARNING);
  ~TensorRTBuilder();

  TensorRTBuilder(const TensorRTBuilder&) = delete;
  TensorRTBuilder& operator=(const TensorRTBuilder&) = delete;

  /// Build the engine described by ``opts`` and serialize it to
  /// ``opts.output_engine_path``. Throws std::runtime_error on failure.
  void build(const BuildOptions& opts);

  /// Convenience: parse ONNX, configure a profile, and return a
  /// serialized plan in memory (does not write to disk).
  std::vector<std::uint8_t> build_serialized(const BuildOptions& opts);

  /// Save a serialized plan to disk.
  static void save(const std::vector<std::uint8_t>& plan,
                   const std::string& path);

  /// Load a serialized plan from disk.
  static std::vector<std::uint8_t> load(const std::string& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace av_jetson

#endif  // AV_JETSON_CPP_TENSOR_RT_HPP
