#pragma once

#include "cuda/CudaUtils.cuh"

#include <cstddef>
#include <utility>

namespace ac::cuda {

/// RAII wrapper for a flat GPU device allocation.
/// Move-only. Automatically calls cudaFree on destruction.
template <typename T>
class DeviceField {
public:
    DeviceField() = default;

    explicit DeviceField(std::size_t count)
        : count_(count)
    {
        if (count_ > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
        }
    }

    ~DeviceField() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceField(DeviceField&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_)
    {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    DeviceField& operator=(DeviceField&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    DeviceField(const DeviceField&) = delete;
    DeviceField& operator=(const DeviceField&) = delete;

    [[nodiscard]] T* data() noexcept { return ptr_; }
    [[nodiscard]] const T* data() const noexcept { return ptr_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }

    /// Async copy from host to device.
    void copy_from_host(const T* host_data, cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, host_data, count_ * sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
    }

    /// Async copy from device to host.
    void copy_to_host(T* host_data, cudaStream_t stream = nullptr) const {
        CUDA_CHECK(cudaMemcpyAsync(host_data, ptr_, count_ * sizeof(T),
                                    cudaMemcpyDeviceToHost, stream));
    }

    /// Copy from another DeviceField.
    void copy_from(const DeviceField& other, cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(ptr_, other.ptr_, count_ * sizeof(T),
                                    cudaMemcpyDeviceToDevice, stream));
    }

    /// Zero all bytes asynchronously.
    void zero_async(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream));
    }

    /// Swap pointers with another DeviceField (O(1), no GPU work).
    friend void swap(DeviceField& a, DeviceField& b) noexcept {
        std::swap(a.ptr_, b.ptr_);
        std::swap(a.count_, b.count_);
    }

private:
    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

} // namespace ac::cuda
