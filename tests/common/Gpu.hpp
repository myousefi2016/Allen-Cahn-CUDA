#pragma once

#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>

namespace ac::test {

/// True if a CUDA device is usable; otherwise `why` says what failed
/// (driver/runtime error or no device).
inline bool gpu_available(std::string& why) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        why = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorName(err) + " (" +
              cudaGetErrorString(err) + ")";
        return false;
    }
    if (count == 0) {
        why = "no CUDA devices";
        return false;
    }
    return true;
}

/// AC_REQUIRE_GPU=1 turns "no usable GPU" into a test failure, so a job that
/// is meant to exercise the GPU cannot pass by skipping everything.
inline bool gpu_required() {
    const char* v = std::getenv("AC_REQUIRE_GPU");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

} // namespace ac::test

/// For SetUp() of GPU fixtures: skip without a usable GPU, or fail when
/// AC_REQUIRE_GPU=1 (a fatal failure in SetUp prevents the test body).
#define AC_GPU_TEST_SETUP()                                                                        \
    do {                                                                                           \
        std::string ac_gpu_why_;                                                                   \
        if (!::ac::test::gpu_available(ac_gpu_why_)) {                                             \
            if (::ac::test::gpu_required())                                                        \
                FAIL() << ac_gpu_why_ << " and AC_REQUIRE_GPU=1";                                  \
            GTEST_SKIP() << ac_gpu_why_;                                                           \
        }                                                                                          \
    } while (0)
