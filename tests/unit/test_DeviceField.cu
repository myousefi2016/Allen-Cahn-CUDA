#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

using namespace ac::cuda;

class DeviceFieldTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        ac::Logger::init(spdlog::level::off);
    }
};

TEST_F(DeviceFieldTest, Construction) {
    DeviceField<double> field(100);
    EXPECT_EQ(field.size(), 100u);
    EXPECT_EQ(field.bytes(), 100 * sizeof(double));
    EXPECT_NE(field.data(), nullptr);
    EXPECT_FALSE(field.empty());
}

TEST_F(DeviceFieldTest, DefaultConstruction) {
    DeviceField<double> field;
    EXPECT_EQ(field.size(), 0u);
    EXPECT_EQ(field.data(), nullptr);
    EXPECT_TRUE(field.empty());
}

TEST_F(DeviceFieldTest, MoveConstruction) {
    DeviceField<double> a(50);
    double* ptr = a.data();

    DeviceField<double> b(std::move(a));
    EXPECT_EQ(b.data(), ptr);
    EXPECT_EQ(b.size(), 50u);
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0u);
}

TEST_F(DeviceFieldTest, MoveAssignment) {
    DeviceField<double> a(50);
    DeviceField<double> b(100);

    b = std::move(a);
    EXPECT_EQ(b.size(), 50u);
    EXPECT_EQ(a.data(), nullptr);
}

TEST_F(DeviceFieldTest, CopyRoundTrip) {
    const std::size_t N = 256;
    DeviceField<double> field(N);

    // Upload
    std::vector<double> host_data(N);
    std::iota(host_data.begin(), host_data.end(), 0.0); // 0, 1, 2, ...
    field.copy_from_host(host_data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    std::vector<double> result(N, -1.0);
    field.copy_to_host(result.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], static_cast<double>(i));
    }
}

TEST_F(DeviceFieldTest, ZeroAsync) {
    const std::size_t N = 128;
    DeviceField<double> field(N);

    // Upload non-zero data
    std::vector<double> data(N, 42.0);
    field.copy_from_host(data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Zero it
    field.zero_async();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download and verify
    std::vector<double> result(N, -1.0);
    field.copy_to_host(result.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < N; ++i) {
        EXPECT_DOUBLE_EQ(result[i], 0.0);
    }
}

TEST_F(DeviceFieldTest, Swap) {
    DeviceField<double> a(10);
    DeviceField<double> b(20);

    std::vector<double> data_a(10, 1.0);
    std::vector<double> data_b(20, 2.0);
    a.copy_from_host(data_a.data());
    b.copy_from_host(data_b.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    swap(a, b);

    EXPECT_EQ(a.size(), 20u);
    EXPECT_EQ(b.size(), 10u);

    std::vector<double> result_a(20);
    std::vector<double> result_b(10);
    a.copy_to_host(result_a.data());
    b.copy_to_host(result_b.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto v : result_a)
        EXPECT_DOUBLE_EQ(v, 2.0);
    for (auto v : result_b)
        EXPECT_DOUBLE_EQ(v, 1.0);
}

TEST_F(DeviceFieldTest, CopyFrom) {
    const std::size_t N = 64;
    DeviceField<double> src(N);
    DeviceField<double> dst(N);

    std::vector<double> data(N, 99.0);
    src.copy_from_host(data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    dst.copy_from(src);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> result(N);
    dst.copy_to_host(result.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (auto v : result)
        EXPECT_DOUBLE_EQ(v, 99.0);
}

TEST_F(DeviceFieldTest, LargeAllocation) {
    // Allocate 1M doubles (~8 MB)
    const std::size_t N = 1024 * 1024;
    DeviceField<double> field(N);
    EXPECT_EQ(field.size(), N);
    EXPECT_NE(field.data(), nullptr);

    // Verify round-trip on subset
    std::vector<double> data(N, 3.14);
    field.copy_from_host(data.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> result(N, 0.0);
    field.copy_to_host(result.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_DOUBLE_EQ(result[0], 3.14);
    EXPECT_DOUBLE_EQ(result[N - 1], 3.14);
}
