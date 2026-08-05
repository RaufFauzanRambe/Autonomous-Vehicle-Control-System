// =============================================================================
// File: cuda/image_processing.cu
// Brief: Image-processing kernels for the Jetson AV stack — RGB→GRAY,
//        bilinear resize, mean/std normalization, NCHW↔NHWC layout
//        conversion, and Gaussian blur. Uses cooperative groups for
//        clean warp/block iteration.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
//
// Build:
//   nvcc -O3 -std=c++17 --use_fast_math \
//        -gencode arch=compute_72,code=sm_72 \
//        -gencode arch=compute_87,code=sm_87 \
//        image_processing.cu -o image_processing
// -------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy.h>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <iostream>

#include "cuda_utils.cuh"

namespace cg = cooperative_groups;

// =============================================================================
// 1. RGB → GRAY conversion (NTSC weights: 0.299 R + 0.587 G + 0.114 B).
//    Each thread processes one pixel; memory access is fully coalesced.
// =============================================================================
__global__ void rgb_to_gray_kernel(const uint8_t* __restrict__ rgb,
                                   uint8_t* __restrict__ gray,
                                   int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;
    float r = rgb[idx + 0];
    float g = rgb[idx + 1];
    float b = rgb[idx + 2];
    float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
    gray[y * width + x] = static_cast<uint8_t>(luminance + 0.5f);
}

void rgb_to_gray(const uint8_t* d_rgb, uint8_t* d_gray,
                 int width, int height, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    rgb_to_gray_kernel<<<grid, block, 0, stream>>>(
        d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 2. Bilinear resize — works for both RGB (channels=3) and grayscale.
//    Each thread writes one output pixel by sampling the input bilinearly.
// =============================================================================
__global__ void resize_bilinear_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       int src_w, int src_h,
                                       int dst_w, int dst_h,
                                       int channels) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;
    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = sx - x0;
    float fy = sy - y0;

    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    for (int c = 0; c < channels; ++c) {
        float v00 = src[(y0 * src_w + x0) * channels + c];
        float v01 = src[(y0 * src_w + x1) * channels + c];
        float v10 = src[(y1 * src_w + x0) * channels + c];
        float v11 = src[(y1 * src_w + x1) * channels + c];
        float v0 = v00 + (v01 - v00) * fx;
        float v1 = v10 + (v11 - v10) * fx;
        float v = v0 + (v1 - v0) * fy;
        dst[(dy * dst_w + dx) * channels + c] =
            static_cast<uint8_t>(v + 0.5f);
    }
}

void resize_bilinear(const uint8_t* d_src, uint8_t* d_dst,
                     int src_w, int src_h, int dst_w, int dst_h,
                     int channels, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);
    resize_bilinear_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, src_w, src_h, dst_w, dst_h, channels);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 3. Normalize — apply (x / 255.0 - mean) / std per channel.
//    Input: uint8 NHWC, Output: float32 NHWC. Channels typically 3.
// =============================================================================
__global__ void normalize_kernel(const uint8_t* __restrict__ src,
                                 float* __restrict__ dst,
                                 int width, int height, int channels,
                                 float3 mean, float3 std) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = (y * width + x) * channels;
    if (channels == 3) {
        dst[idx + 0] = (src[idx + 0] / 255.0f - mean.x) / std.x;
        dst[idx + 1] = (src[idx + 1] / 255.0f - mean.y) / std.y;
        dst[idx + 2] = (src[idx + 2] / 255.0f - mean.z) / std.z;
    } else {
        for (int c = 0; c < channels; ++c) {
            dst[idx + c] = src[idx + c] / 255.0f;
        }
    }
}

void normalize(const uint8_t* d_src, float* d_dst,
               int width, int height, int channels,
               float mean_r, float mean_g, float mean_b,
               float std_r, float std_g, float std_b,
               cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    normalize_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, width, height, channels,
        make_float3(mean_r, mean_g, mean_b),
        make_float3(std_r, std_g, std_b));
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 4. NCHW ↔ NHWC layout conversion. Used to bridge OpenCV (NHWC) and
//    most deep learning frameworks (NCHW).
// =============================================================================
__global__ void nhwc_to_nchw_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int n, int c, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || ch >= c) return;
    int batch = 0;  // single-batch path for simplicity
    int src_idx = batch * (c * h * w) + (y * w + x) * c + ch;
    int dst_idx = batch * (c * h * w) + (ch * h * w) + (y * w + x);
    dst[dst_idx] = src[src_idx];
}

void nhwc_to_nchw(const float* d_src, float* d_dst,
                  int n, int c, int h, int w, cudaStream_t stream = 0) {
    dim3 block(8, 8, 4);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y,
              (c + block.z - 1) / block.z);
    nhwc_to_nchw_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, n, c, h, w);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void nchw_to_nhwc_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int n, int c, int h, int w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int ch = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= w || y >= h || ch >= c) return;
    int batch = 0;
    int src_idx = batch * (c * h * w) + (ch * h * w) + (y * w + x);
    int dst_idx = batch * (c * h * w) + (y * w + x) * c + ch;
    dst[dst_idx] = src[src_idx];
}

void nchw_to_nhwc(const float* d_src, float* d_dst,
                  int n, int c, int h, int w, cudaStream_t stream = 0) {
    dim3 block(8, 8, 4);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y,
              (c + block.z - 1) / block.z);
    nchw_to_nhwc_kernel<<<grid, block, 0, stream>>>(
        d_src, d_dst, n, c, h, w);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 5. Gaussian blur — separable implementation (horizontal + vertical pass).
//    Uses shared memory to cache the input tile plus the halo region.
// =============================================================================
constexpr int K_RADIUS = 4;  // 9-tap kernel
constexpr float K_SIGMA = 2.0f;

__device__ __constant__ float d_gauss_kernel[2 * K_RADIUS + 1];

void upload_gauss_kernel(float sigma) {
    float h_kernel[2 * K_RADIUS + 1];
    float sum = 0.0f;
    for (int i = -K_RADIUS; i <= K_RADIUS; ++i) {
        float v = expf(-(i * i) / (2.0f * sigma * sigma));
        h_kernel[i + K_RADIUS] = v;
        sum += v;
    }
    for (int i = 0; i < 2 * K_RADIUS + 1; ++i) {
        h_kernel[i] /= sum;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_gauss_kernel, h_kernel,
                                   sizeof(h_kernel)));
}

__global__ void gaussian_blur_h_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       int width, int height) {
    extern __shared__ uint8_t sdata[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // Load row into shared memory (with halo).
    for (int i = tid; i < bdim + 2 * K_RADIUS; i += bdim) {
        int sx = blockIdx.x * blockDim.x + i - K_RADIUS;
        sx = max(0, min(sx, width - 1));
        if (y < height) sdata[i] = src[y * width + sx];
    }
    __syncthreads();

    if (x >= width || y >= height) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
        acc += sdata[tid + K_RADIUS + k] * d_gauss_kernel[k + K_RADIUS];
    }
    dst[y * width + x] = static_cast<uint8_t>(acc + 0.5f);
}

__global__ void gaussian_blur_v_kernel(const uint8_t* __restrict__ src,
                                       uint8_t* __restrict__ dst,
                                       int width, int height) {
    extern __shared__ uint8_t sdata[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y;
    int bdim = blockDim.y;

    for (int i = tid; i < bdim + 2 * K_RADIUS; i += bdim) {
        int sy = blockIdx.y * blockDim.y + i - K_RADIUS;
        sy = max(0, min(sy, height - 1));
        if (x < width) sdata[i] = src[sy * width + x];
    }
    __syncthreads();

    if (x >= width || y >= height) return;
    float acc = 0.0f;
    #pragma unroll
    for (int k = -K_RADIUS; k <= K_RADIUS; ++k) {
        acc += sdata[tid + K_RADIUS + k] * d_gauss_kernel[k + K_RADIUS];
    }
    dst[y * width + x] = static_cast<uint8_t>(acc + 0.5f);
}

void gaussian_blur(const uint8_t* d_src, uint8_t* d_dst,
                   uint8_t* d_tmp, int width, int height,
                   cudaStream_t stream = 0) {
    upload_gauss_kernel(K_SIGMA);
    dim3 block(16, 16);
    dim3 grid_h((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);
    dim3 grid_v((width + block.x - 1) / block.x,
                (height + block.y - 1) / block.y);
    size_t smem_h = (block.x + 2 * K_RADIUS) * sizeof(uint8_t);
    size_t smem_v = (block.y + 2 * K_RADIUS) * sizeof(uint8_t);
    gaussian_blur_h_kernel<<<grid_h, block, smem_h, stream>>>(
        d_src, d_tmp, width, height);
    gaussian_blur_v_kernel<<<grid_v, block, smem_v, stream>>>(
        d_tmp, d_dst, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Host smoke-test
// =============================================================================
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    constexpr int W = 1280, H = 720;
    std::vector<uint8_t> h_rgb(W * H * 3, 128);
    std::vector<uint8_t> h_gray(W * H, 0);

    uint8_t *d_rgb, *d_gray, *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_rgb, W * H * 3 * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_gray, W * H * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_tmp, W * H * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), W * H * 3,
                          cudaMemcpyHostToDevice));

    CudaTimer timer;

    timer.start();
    rgb_to_gray(d_rgb, d_gray, W, H);
    std::printf("rgb_to_gray: %.3f ms\n", timer.stop_ms());

    timer.start();
    gaussian_blur(d_gray, d_gray, d_tmp, W, H);
    std::printf("gaussian_blur: %.3f ms\n", timer.stop_ms());

    // Resize 1280x720 → 640x360
    uint8_t* d_resized;
    CUDA_CHECK(cudaMalloc(&d_resized, 640 * 360 * sizeof(uint8_t)));
    timer.start();
    resize_bilinear(d_gray, d_resized, W, H, 640, 360, 1);
    std::printf("resize_bilinear: %.3f ms\n", timer.stop_ms());

    // Normalize
    float* d_norm;
    CUDA_CHECK(cudaMalloc(&d_norm, W * H * 3 * sizeof(float)));
    timer.start();
    normalize(d_rgb, d_norm, W, H, 3,
              0.485f, 0.456f, 0.406f,
              0.229f, 0.224f, 0.225f);
    std::printf("normalize: %.3f ms\n", timer.stop_ms());

    // NCHW <-> NHWC
    float *d_nchw, *d_nhwc;
    CUDA_CHECK(cudaMalloc(&d_nchw, 3 * H * W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nhwc, 3 * H * W * sizeof(float)));
    timer.start();
    nhwc_to_nchw(d_norm, d_nchw, 1, 3, H, W);
    std::printf("nhwc_to_nchw: %.3f ms\n", timer.stop_ms());
    timer.start();
    nchw_to_nhwc(d_nchw, d_nhwc, 1, 3, H, W);
    std::printf("nchw_to_nhwc: %.3f ms\n", timer.stop_ms());

    cudaFree(d_rgb); cudaFree(d_gray); cudaFree(d_tmp);
    cudaFree(d_resized); cudaFree(d_norm); cudaFree(d_nchw); cudaFree(d_nhwc);
    return 0;
}
