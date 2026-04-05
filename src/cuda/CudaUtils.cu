#include "cuda/CudaUtils.cuh"

#include <spdlog/spdlog.h>

namespace ac::cuda {

ScopedTimer::ScopedTimer(const char* label, cudaStream_t stream)
    : stream_(stream), label_(label)
{
    start_.record(stream_);
}

ScopedTimer::~ScopedTimer()
{
    stop_.record(stream_);
    stop_.synchronize();
    float ms = stop_.elapsed_ms(start_);
    spdlog::debug("{}: {:.3f} ms", label_, ms);
}

} // namespace ac::cuda
