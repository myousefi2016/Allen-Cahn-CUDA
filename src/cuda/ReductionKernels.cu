#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"

#include <algorithm>
#include <cfloat>
#include <stdexcept>

namespace ac::cuda {

// ── Block-level max reduction ──────────────────────────────────────────────

/// Warp-level max reduction using __shfl_down_sync (correct on Volta+ with ITS).
__device__ double warp_reduce_max_val(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = nan_max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
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
        thread_max = nan_max(thread_max, fabs(a[i] - b[i]));
        if (i + blockDim.x < static_cast<unsigned>(N))
            thread_max = nan_max(thread_max, fabs(a[i + blockDim.x] - b[i + blockDim.x]));
    }

    sdata[tid] = thread_max;
    __syncthreads();

    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = nan_max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Final warp reduction using shuffle
    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = nan_max(val, sdata[tid + 32]);
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
        thread_max = nan_max(thread_max, block_results[i]);
    }

    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = nan_max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = nan_max(val, sdata[tid + 32]);
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
        thread_max = nan_max(thread_max, fabs(data[i]));
        if (i + blockDim.x < static_cast<unsigned>(N))
            thread_max = nan_max(thread_max, fabs(data[i + blockDim.x]));
    }

    sdata[tid] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = nan_max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = nan_max(val, sdata[tid + 32]);
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

/// |x| as an integer whose unsigned order is the order of |x| for every
/// double, with NaN above +inf: with the sign bit cleared, the exponent and
/// mantissa bits of an IEEE double order like the magnitude, and a NaN has
/// the largest exponent and a nonzero mantissa. A max over these is a
/// NaN-propagating max of magnitudes in integer instructions; nan_max's
/// FP64 compares made the fused reduction below compute bound on GPUs with a
/// low FP64 rate (169 us against 78 us for max_abs_diff on 160^3, RTX 4090).
__device__ __forceinline__ unsigned long long magnitude_bits(double x) {
    return static_cast<unsigned long long>(__double_as_longlong(x)) & 0x7FFFFFFFFFFFFFFFULL;
}

__device__ __forceinline__ unsigned long long warp_reduce_max_bits(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = max(v, __shfl_down_sync(0xFFFFFFFF, v, offset));
    return v;
}

/// Block-wide max of two values per thread (blockDim.x a power of two,
/// >= 64, with 2 * blockDim.x entries of shared memory); thread 0 returns
/// them in d and m.
__device__ void block_reduce_max_bits2(unsigned long long& d, unsigned long long& m) {
    extern __shared__ unsigned long long sbits[];
    unsigned long long* sd = sbits;
    unsigned long long* sm = sbits + blockDim.x;
    const unsigned int tid = threadIdx.x;
    sd[tid] = d;
    sm[tid] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sd[tid] = max(sd[tid], sd[tid + s]);
            sm[tid] = max(sm[tid], sm[tid + s]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        d = warp_reduce_max_bits(max(sd[tid], sd[tid + 32]));
        m = warp_reduce_max_bits(max(sm[tid], sm[tid + 32]));
    }
}

/// max|a[i] - b[i]| and max(|a[i]|, |b[i]|) in one pass: block k writes its
/// two maxima to block_results[k] and block_results[gridDim.x + k].
__global__ void max_abs_diff_and_max_abs_kernel(const double* __restrict__ a,
                                                const double* __restrict__ b,
                                                double* __restrict__ block_results, int N) {
    const unsigned int grid_stride = gridDim.x * blockDim.x * 2;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    unsigned long long d = 0, m = 0;
    for (; i < static_cast<unsigned>(N); i += grid_stride) {
        d = max(d, magnitude_bits(a[i] - b[i]));
        m = max(m, max(magnitude_bits(a[i]), magnitude_bits(b[i])));
        if (i + blockDim.x < static_cast<unsigned>(N)) {
            const unsigned int j = i + blockDim.x;
            d = max(d, magnitude_bits(a[j] - b[j]));
            m = max(m, max(magnitude_bits(a[j]), magnitude_bits(b[j])));
        }
    }
    block_reduce_max_bits2(d, m);
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = __longlong_as_double(static_cast<long long>(d));
        block_results[gridDim.x + blockIdx.x] = __longlong_as_double(static_cast<long long>(m));
    }
}

/// Second pass: result[0] = max of block_results[0, n), result[1] = max of
/// block_results[n, 2n) (all non-negative or NaN).
__global__ void final_max2_kernel(const double* __restrict__ block_results,
                                  double* __restrict__ result, int num_blocks) {
    unsigned long long d = 0, m = 0;
    for (unsigned int i = threadIdx.x; i < static_cast<unsigned>(num_blocks); i += blockDim.x) {
        d = max(d, magnitude_bits(block_results[i]));
        m = max(m, magnitude_bits(block_results[num_blocks + i]));
    }
    block_reduce_max_bits2(d, m);
    if (threadIdx.x == 0) {
        result[0] = __longlong_as_double(static_cast<long long>(d));
        result[1] = __longlong_as_double(static_cast<long long>(m));
    }
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

void launch_max_abs_diff_and_max_abs(const double* a, const double* b, double* d_result,
                                     std::size_t N, double* scratch, int scratch_size,
                                     cudaStream_t stream) {
    if (scratch_size < 2)
        throw std::invalid_argument("launch_max_abs_diff_and_max_abs: scratch_size must be >= 2");
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
    num_blocks = std::min(std::max(num_blocks, 1), scratch_size / 2);

    constexpr std::size_t shared = 2 * BLOCK_SIZE * sizeof(unsigned long long);
    max_abs_diff_and_max_abs_kernel<<<num_blocks, BLOCK_SIZE, shared, stream>>>(
        a, b, scratch, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());
    final_max2_kernel<<<1, BLOCK_SIZE, shared, stream>>>(scratch, d_result, num_blocks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
