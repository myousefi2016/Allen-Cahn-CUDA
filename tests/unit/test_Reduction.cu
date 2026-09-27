#include "common/Gpu.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class ReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

TEST_F(ReductionTest, MaxAbsDiff) {
    const std::size_t N = 10000;

    std::vector<double> a(N), b(N);
    for (std::size_t i = 0; i < N; ++i) {
        a[i] = std::sin(i * 0.01);
        b[i] = std::cos(i * 0.01);
    }

    // CPU reference
    double expected = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        expected = std::max(expected, std::abs(a[i] - b[i]));
    }

    DeviceField<double> d_a(N), d_b(N), d_result(1);
    d_a.copy_from_host(a.data());
    d_b.copy_from_host(b.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_max_abs_diff(d_a.data(), d_b.data(), d_result.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_result.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(ReductionTest, MaxAbsDiffIdentical) {
    const std::size_t N = 1000;
    std::vector<double> data(N, 3.14);

    DeviceField<double> d_a(N), d_b(N), d_result(1);
    d_a.copy_from_host(data.data());
    d_b.copy_from_host(data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_max_abs_diff(d_a.data(), d_b.data(), d_result.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_result.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST_F(ReductionTest, MaxAbsDiffSingleDifference) {
    const std::size_t N = 5000;
    std::vector<double> a(N, 0.0), b(N, 0.0);
    a[2500] = 7.5; // The only difference

    DeviceField<double> d_a(N), d_b(N), d_result(1);
    d_a.copy_from_host(a.data());
    d_b.copy_from_host(b.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_max_abs_diff(d_a.data(), d_b.data(), d_result.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_result.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_DOUBLE_EQ(result, 7.5);
}

TEST_F(ReductionTest, MaxAbsReduction) {
    const std::size_t N = 8000;
    std::vector<double> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = std::sin(i * 0.005) * (i % 2 == 0 ? 1.0 : -1.0);
    }

    double expected = 0.0;
    for (auto v : data)
        expected = std::max(expected, std::abs(v));

    DeviceField<double> d_data(N), d_result(1);
    d_data.copy_from_host(data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_max_abs_reduction(d_data.data(), d_result.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_result.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_NEAR(result, expected, 1e-10);
}

TEST_F(ReductionTest, LargeArray) {
    // Test with > 1M elements to ensure multi-block reduction works
    const std::size_t N = 1024 * 1024 + 37; // Non-power-of-2

    std::vector<double> a(N, 1.0), b(N, 1.0);
    b[N / 2] = 100.0; // Max diff = 99.0

    DeviceField<double> d_a(N), d_b(N), d_result(1);
    d_a.copy_from_host(a.data());
    d_b.copy_from_host(b.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_max_abs_diff(d_a.data(), d_b.data(), d_result.data(), N);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_result.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_DOUBLE_EQ(result, 99.0);
}

// A diverged field must not reduce to a finite maximum: fmax drops a NaN
// operand, which would let adaptive dt and the saturation guard carry on
// with a blown-up solution. The NaN is placed in the first, a middle and the
// last element (last block, and the second half of a thread's element pair).
TEST_F(ReductionTest, MaxAbsDiffAndMaxAbsPropagateNaN) {
    const std::size_t N = 1024 * 1024 + 37;
    for (std::size_t at : {std::size_t{0}, N / 2 + 300, N - 1}) {
        SCOPED_TRACE("NaN at " + std::to_string(at));
        std::vector<double> a(N, 1.0), b(N, 2.0);
        a[at] = std::nan("");

        DeviceField<double> d_a(N), d_b(N), d_result(1);
        d_a.copy_from_host(a.data());
        d_b.copy_from_host(b.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        double result = 0.0;
        launch_max_abs_diff(d_a.data(), d_b.data(), d_result.data(), N);
        d_result.copy_to_host(&result);
        CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_TRUE(std::isnan(result)) << "max|a-b| = " << result;

        result = 0.0;
        launch_max_abs_reduction(d_a.data(), d_result.data(), N);
        d_result.copy_to_host(&result);
        CUDA_CHECK(cudaDeviceSynchronize());
        EXPECT_TRUE(std::isnan(result)) << "max|a| = " << result;
    }
}
