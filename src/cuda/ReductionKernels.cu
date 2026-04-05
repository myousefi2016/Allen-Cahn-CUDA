#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"

#include <cfloat>
#include <algorithm>

namespace ac::cuda {

// ── Block-level max reduction ──────────────────────────────────────────────

__device__ void warp_reduce_max(volatile double* sdata, int tid) {
    if (tid < 32) {
        if (sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
        if (sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
        if (sdata[tid] < sdata[tid +  8]) sdata[tid] = sdata[tid +  8];
        if (sdata[tid] < sdata[tid +  4]) sdata[tid] = sdata[tid +  4];
        if (sdata[tid] < sdata[tid +  2]) sdata[tid] = sdata[tid +  2];
        if (sdata[tid] < sdata[tid +  1]) sdata[tid] = sdata[tid +  1];
    }
}

/// Compute max(|a[i] - b[i]|) across all elements via block reduction.
__global__ void max_abs_diff_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ block_results,
    int N)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double thread_max = 0.0;
    if (i < static_cast<unsigned>(N))
        thread_max = fabs(a[i] - b[i]);
    if (i + blockDim.x < static_cast<unsigned>(N))
        thread_max = fmax(thread_max, fabs(a[i + blockDim.x] - b[i + blockDim.x]));

    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    warp_reduce_max(sdata, tid);

    if (tid == 0) {
        block_results[blockIdx.x] = sdata[0];
    }
}

/// Second-pass reduction: find max across block results.
__global__ void final_max_kernel(
    const double* __restrict__ block_results,
    double* __restrict__ result,
    int num_blocks)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    double thread_max = 0.0;

    for (unsigned int i = tid; i < static_cast<unsigned>(num_blocks); i += blockDim.x) {
        thread_max = fmax(thread_max, block_results[i]);
    }

    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    warp_reduce_max(sdata, tid);

    if (tid == 0) {
        result[0] = sdata[0];
    }
}

// ── Launch wrapper ─────────────────────────────────────────────────────────

void launch_max_abs_diff(
    const double* a, const double* b,
    double* d_result, std::size_t N,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::max(num_blocks, 1);

    // Temporary storage for per-block results
    DeviceField<double> block_results(static_cast<std::size_t>(num_blocks));

    max_abs_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        a, b, block_results.data(), static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    // Final reduction pass
    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        block_results.data(), d_result, num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

/// Max absolute value reduction: max(|a[i]|)
__global__ void max_abs_kernel(
    const double* __restrict__ data,
    double* __restrict__ block_results,
    int N)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double thread_max = 0.0;
    if (i < static_cast<unsigned>(N))
        thread_max = fabs(data[i]);
    if (i + blockDim.x < static_cast<unsigned>(N))
        thread_max = fmax(thread_max, fabs(data[i + blockDim.x]));

    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    warp_reduce_max(sdata, tid);

    if (tid == 0) block_results[blockIdx.x] = sdata[0];
}

void launch_max_abs_reduction(
    const double* field, double* result, std::size_t N,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::max(num_blocks, 1);

    DeviceField<double> block_results(static_cast<std::size_t>(num_blocks));

    max_abs_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        field, block_results.data(), static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        block_results.data(), result, num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
