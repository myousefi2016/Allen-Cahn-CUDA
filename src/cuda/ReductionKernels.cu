#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"

#include <algorithm>
#include <cfloat>

namespace ac::cuda {

// ── Block-level max reduction ──────────────────────────────────────────────

/// Warp-level max reduction using __shfl_down_sync (correct on Volta+ with ITS).
__device__ double warp_reduce_max_val(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/// Compute max(|a[i] - b[i]|) across all elements via block reduction.
/// Uses grid-stride loop so that fewer blocks can still cover all N elements.
__global__ void max_abs_diff_kernel(const double* __restrict__ a, const double* __restrict__ b,
                                    double* __restrict__ block_results, int N) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int grid_stride = gridDim.x * blockDim.x * 2;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double thread_max = 0.0;
    for (; i < static_cast<unsigned>(N); i += grid_stride) {
        thread_max = fmax(thread_max, fabs(a[i] - b[i]));
        if (i + blockDim.x < static_cast<unsigned>(N))
            thread_max = fmax(thread_max, fabs(a[i + blockDim.x] - b[i + blockDim.x]));
    }

    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Final warp reduction using shuffle
    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = fmax(val, sdata[tid + 32]);
        val = warp_reduce_max_val(val);
        if (tid == 0)
            block_results[blockIdx.x] = val;
    }
}

/// Second-pass reduction: find max across block results.
__global__ void final_max_kernel(const double* __restrict__ block_results,
                                 double* __restrict__ result, int num_blocks) {
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

    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = fmax(val, sdata[tid + 32]);
        val = warp_reduce_max_val(val);
        if (tid == 0)
            result[0] = val;
    }
}

// ── Launch wrapper ─────────────────────────────────────────────────────────

void launch_max_abs_diff(const double* a, const double* b, double* d_result, std::size_t N,
                         cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::max(num_blocks, 1);

    // Temporary storage for per-block results
    DeviceField<double> block_results(static_cast<std::size_t>(num_blocks));

    max_abs_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        a, b, block_results.data(), static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    // Final reduction pass
    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(block_results.data(),
                                                                             d_result, num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

/// Max absolute value reduction: max(|a[i]|)
/// Uses grid-stride loop so that fewer blocks can still cover all N elements.
__global__ void max_abs_kernel(const double* __restrict__ data, double* __restrict__ block_results,
                               int N) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int grid_stride = gridDim.x * blockDim.x * 2;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double thread_max = 0.0;
    for (; i < static_cast<unsigned>(N); i += grid_stride) {
        thread_max = fmax(thread_max, fabs(data[i]));
        if (i + blockDim.x < static_cast<unsigned>(N))
            thread_max = fmax(thread_max, fabs(data[i + blockDim.x]));
    }

    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = fmax(val, sdata[tid + 32]);
        val = warp_reduce_max_val(val);
        if (tid == 0)
            block_results[blockIdx.x] = val;
    }
}

void launch_max_abs_reduction(const double* field, double* result, std::size_t N,
                              cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::max(num_blocks, 1);

    DeviceField<double> block_results(static_cast<std::size_t>(num_blocks));

    max_abs_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        field, block_results.data(), static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(block_results.data(),
                                                                             result, num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

// ── Overloads with pre-allocated scratch ───────────────────────────────────

void launch_max_abs_diff(const double* a, const double* b, double* d_result, std::size_t N,
                         double* scratch, int scratch_size, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::min(std::max(num_blocks, 1), scratch_size);

    max_abs_diff_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        a, b, scratch, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(scratch, d_result,
                                                                             num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

void launch_max_abs_reduction(const double* field, double* result, std::size_t N, double* scratch,
                              int scratch_size, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::min(std::max(num_blocks, 1), scratch_size);

    max_abs_kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(
        field, scratch, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    final_max_kernel<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(double), stream>>>(scratch, result,
                                                                             num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
