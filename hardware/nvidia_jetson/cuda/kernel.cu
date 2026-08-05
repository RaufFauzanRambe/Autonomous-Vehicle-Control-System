// =============================================================================
// File: cuda/kernel.cu
// Brief: Foundational CUDA kernels — vectorAdd, elementwise ops, reductions,
//        and softmax. Demonstrates __global__/__device__ separation, shared
//        memory usage, and warp-shuffle primitives for the AV Jetson stack.
// Author: AV Control System Team
// Date: 2025
// License: MIT
// =============================================================================
//
// Build:
//   nvcc -O3 -std=c++17 --use_fast_math \
//        -gencode arch=compute_72,code=sm_72 \
//        -gencode arch=compute_87,code=sm_87 \
//        kernel.cu -o kernel
//
// Run:
//   ./kernel
// -------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <iostream>

#include "cuda_utils.cuh"

namespace cg = cooperative_groups;

// =============================================================================
// 1. vectorAdd — the canonical "hello CUDA" kernel. Coalesced 1D access.
// =============================================================================
__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/// Host wrapper for vector_add_kernel.
void vector_add(const float* a, const float* b, float* c, int n,
                cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 2. elementwise — generic elementwise op on float arrays. Supports add,
//    sub, mul, div, scale (alpha*x), and sigmoid via an enum dispatch.
// =============================================================================
enum class ElementwiseOp : int {
    kAdd = 0,
    kSub = 1,
    kMul = 2,
    kDiv = 3,
    kScale = 4,
    kSigmoid = 5,
    kRelu = 6,
};

template <ElementwiseOp op>
__device__ __forceinline__ float apply_op(float x, float y, float alpha) {
    if constexpr (op == ElementwiseOp::kAdd)     return x + y;
    if constexpr (op == ElementwiseOp::kSub)     return x - y;
    if constexpr (op == ElementwiseOp::kMul)     return x * y;
    if constexpr (op == ElementwiseOp::kDiv)     return x / (y + 1e-12f);
    if constexpr (op == ElementwiseOp::kScale)   return alpha * x;
    if constexpr (op == ElementwiseOp::kSigmoid) return 1.0f / (1.0f + __expf(-x));
    if constexpr (op == ElementwiseOp::kRelu)    return fmaxf(0.0f, x);
    return x;
}

template <ElementwiseOp op>
__global__ void elementwise_kernel(const float* __restrict__ x,
                                   const float* __restrict__ y,
                                   float* __restrict__ out,
                                   float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xv = x[idx];
        float yv = y ? y[idx] : 0.0f;
        out[idx] = apply_op<op>(xv, yv, alpha);
    }
}

/// Dispatch helper — picks the templated kernel based on the op enum.
void elementwise(const float* x, const float* y, float* out, int n,
                 ElementwiseOp op, float alpha = 1.0f,
                 cudaStream_t stream = 0) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    switch (op) {
        case ElementwiseOp::kAdd:
            elementwise_kernel<ElementwiseOp::kAdd>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kSub:
            elementwise_kernel<ElementwiseOp::kSub>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kMul:
            elementwise_kernel<ElementwiseOp::kMul>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kDiv:
            elementwise_kernel<ElementwiseOp::kDiv>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kScale:
            elementwise_kernel<ElementwiseOp::kScale>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kSigmoid:
            elementwise_kernel<ElementwiseOp::kSigmoid>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
        case ElementwiseOp::kRelu:
            elementwise_kernel<ElementwiseOp::kRelu>
                <<<blocks, threads, 0, stream>>>(x, y, out, alpha, n); break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// 3. Reduction — sum of a float array using shared memory + warp shuffle.
//    Two-stage reduction: (a) each block reduces into shared memory and
//    writes its partial sum to a temp buffer; (b) the final block reduces
//    the partial sums. Uses cooperative_groups::reduce for the warp step.
// =============================================================================
__global__ void reduce_sum_kernel(const float* __restrict__ in,
                                  float* __restrict__ partial,
                                  int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;

    // Load two elements per thread (grid-stride loop unrolled once).
    float sum = 0.0f;
    if (i < n) sum = in[i];
    if (i + blockDim.x < n) sum += in[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Final warp reduction via shuffle (no __syncthreads needed).
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) partial[blockIdx.x] = v;
    }
}

__global__ void reduce_sum_final_kernel(const float* __restrict__ partial,
                                        float* __restrict__ out,
                                        int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += partial[i];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) *out = v;
    }
}

float reduce_sum(const float* d_in, int n, cudaStream_t stream = 0) {
    const int block = 256;
    int grid = (n + block * 2 - 1) / (block * 2);
    float* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, sizeof(float) * grid));
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    reduce_sum_kernel<<<grid, block, sizeof(float) * block, stream>>>(
        d_in, d_partial, n);
    reduce_sum_final_kernel<<<1, block, sizeof(float) * block, stream>>>(
        d_partial, d_out, grid);

    float result = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_out, sizeof(float),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(d_partial);
    cudaFree(d_out);
    return result;
}

// =============================================================================
// 4. Softmax — numerically stable, online (one-pass) softmax over the last
//    dimension. Layout: [rows, cols], row-major. Each block handles one row.
// =============================================================================
__global__ void softmax_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int blockDim_x = blockDim.x;

    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    // 1. Find the row max for numerical stability (warp reduction).
    float local_max = -INFINITY;
    for (int i = tid; i < cols; i += blockDim_x) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    // Reduce within block.
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim_x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
        }
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    float row_max = sdata[0];

    // 2. Compute exp(x - max) and accumulate sum.
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim_x) {
        float e = __expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim_x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, offset);
        }
        if (tid == 0) sdata[0] = v;
    }
    __syncthreads();
    float row_sum = sdata[0];

    // 3. Normalize.
    float inv_sum = 1.0f / (row_sum + 1e-12f);
    for (int i = tid; i < cols; i += blockDim_x) {
        row_out[i] *= inv_sum;
    }
}

void softmax(const float* d_in, float* d_out, int rows, int cols,
             cudaStream_t stream = 0) {
    int threads = (cols < 1024) ? cols : 1024;
    // Round up to nearest multiple of 32.
    threads = ((threads + 31) / 32) * 32;
    softmax_kernel<<<rows, threads, sizeof(float) * threads, stream>>>(
        d_in, d_out, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Host smoke-test
// =============================================================================
int main() {
    CUDA_CHECK(cudaSetDevice(0));

    constexpr int N = 1 << 20;  // 1M elements
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i) * 0.001f;
        h_b[i] = static_cast<float>(i) * 0.002f;
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));

    CudaTimer timer;

    // 1. vectorAdd
    timer.start();
    vector_add(d_a, d_b, d_c, N);
    float ms = timer.stop_ms();
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::printf("vector_add: %.3f ms, c[0]=%.4f c[N-1]=%.4f\n",
                ms, h_c[0], h_c[N - 1]);

    // 2. elementwise sigmoid
    timer.start();
    elementwise(d_a, nullptr, d_c, N, ElementwiseOp::kSigmoid);
    ms = timer.stop_ms();
    std::printf("elementwise(sigmoid): %.3f ms\n", ms);

    // 3. Reduction
    timer.start();
    float total = reduce_sum(d_a, N);
    ms = timer.stop_ms();
    float expected = 0.0f;
    for (int i = 0; i < N; ++i) expected += h_a[i];
    std::printf("reduce_sum: %.3f ms, got=%.2f expected=%.2f\n",
                ms, total, expected);

    // 4. Softmax over 16 rows of 1024 cols.
    constexpr int ROWS = 16, COLS = 1024;
    std::vector<float> h_in(ROWS * COLS), h_out(ROWS * COLS);
    for (int i = 0; i < ROWS * COLS; ++i) {
        h_in[i] = static_cast<float>(std::rand() % 100) * 0.01f;
    }
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, ROWS * COLS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), ROWS * COLS * sizeof(float),
                          cudaMemcpyHostToDevice));
    timer.start();
    softmax(d_in, d_out, ROWS, COLS);
    ms = timer.stop_ms();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ROWS * COLS * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float row_sum = 0.0f;
    for (int j = 0; j < COLS; ++j) row_sum += h_out[j];
    std::printf("softmax: %.3f ms, row0 sum=%.4f (should be ~1.0)\n",
                ms, row_sum);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
