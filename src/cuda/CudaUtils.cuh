#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace ac::cuda {

// ── Error checking ─────────────────────────────────────────────────────────

/// Throws std::runtime_error on CUDA failure.
inline void check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(err) + " (" + cudaGetErrorName(err) + ")");
    }
}

#define CUDA_CHECK(call) ::ac::cuda::check((call), __FILE__, __LINE__)

// ── RAII CUDA Stream ───────────────────────────────────────────────────────

class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }

    explicit Stream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    ~Stream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }  // NOLINT

    void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

// ── RAII CUDA Event ────────────────────────────────────────────────────────

class Event {
public:
    explicit Event(unsigned int flags = cudaEventDefault) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }

    ~Event() {
        if (event_) cudaEventDestroy(event_);
    }

    Event(Event&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    Event& operator=(Event&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() { CUDA_CHECK(cudaEventSynchronize(event_)); }

    /// Returns elapsed time in milliseconds between start and this event.
    [[nodiscard]] float elapsed_ms(const Event& start) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

private:
    cudaEvent_t event_ = nullptr;
};

// ── Scoped GPU timer ───────────────────────────────────────────────────────

class ScopedTimer {
public:
    ScopedTimer(const char* label, cudaStream_t stream = nullptr);
    ~ScopedTimer();

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    Event start_{cudaEventDefault};
    Event stop_{cudaEventDefault};
    cudaStream_t stream_;
    const char* label_;
};

// ── Kernel launch configuration ────────────────────────────────────────────

struct LaunchConfig {
    dim3 grid;
    dim3 block;
    std::size_t shared_mem = 0;
    cudaStream_t stream = nullptr;

    /// 1D launch config for total_threads work items.
    static LaunchConfig for_1d(std::size_t total_threads, int block_size = 256) {
        LaunchConfig cfg;
        cfg.block = dim3(static_cast<unsigned>(block_size));
        cfg.grid = dim3(static_cast<unsigned>(
            (total_threads + block_size - 1) / block_size));
        return cfg;
    }

    /// 3D launch config for structured grid.
    static LaunchConfig for_3d(int Nx, int Ny, int Nz,
                                dim3 block = {8, 8, 8}) {
        LaunchConfig cfg;
        cfg.block = block;
        cfg.grid = dim3(
            (static_cast<unsigned>(Nx) + block.x - 1) / block.x,
            (static_cast<unsigned>(Ny) + block.y - 1) / block.y,
            (static_cast<unsigned>(Nz) + block.z - 1) / block.z);
        return cfg;
    }
};

} // namespace ac::cuda
