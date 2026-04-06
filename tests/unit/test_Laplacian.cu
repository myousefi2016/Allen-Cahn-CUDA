#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class LaplacianTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }
};

/// Kernel that evaluates the Laplacian at every interior point.
__global__ void test_laplacian_kernel(
    const double* __restrict__ phi,
    double* __restrict__ lap,
    KernelParams p, int stencil_type)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.Nx * p.Ny * p.Nz;
    if (tid >= static_cast<unsigned>(total)) return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (x < 1 || x >= p.Nx-1 || y < 1 || y >= p.Ny-1 || z < 1 || z >= p.Nz-1) {
        lap[c] = 0.0;
        return;
    }

    p.stencil_type = stencil_type;
    lap[c] = laplacian(phi, x, y, z, p);
}

// Test: Laplacian of f(x,y,z) = x^2 + y^2 + z^2 should be 6.0 (exactly for 2nd order)
TEST_F(LaplacianTest, QuadraticField7pt)
{
    const int N = 32;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;
    p.stencil_type = 0;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    // f(x,y,z) = (x*dx)^2 + (y*dy)^2 + (z*dz)^2
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x*N*N + y*N + z] = (x*h)*(x*h) + (y*h)*(y*h) + (z*h)*(z*h);

    DeviceField<double> d_phi(total);
    DeviceField<double> d_lap(total);
    d_phi.copy_from_host(phi_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_laplacian_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_lap.data(), p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lap_host(total);
    d_lap.copy_to_host(lap_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check interior points: Laplacian of x^2+y^2+z^2 = 2+2+2 = 6
    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z) {
                double lap_val = lap_host[x*N*N + y*N + z];
                EXPECT_NEAR(lap_val, 6.0, 1e-10)
                    << "Failed at (" << x << "," << y << "," << z << ")";
            }
}

// Test: 27-point Laplacian on quadratic field (should also give 6.0, isotropic)
TEST_F(LaplacianTest, QuadraticField27pt)
{
    const int N = 32;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;
    p.stencil_type = 1;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x*N*N + y*N + z] = (x*h)*(x*h) + (y*h)*(y*h) + (z*h)*(z*h);

    DeviceField<double> d_phi(total);
    DeviceField<double> d_lap(total);
    d_phi.copy_from_host(phi_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_laplacian_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_lap.data(), p, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lap_host(total);
    d_lap.copy_to_host(lap_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 27-point stencil should also give 6.0 for quadratic functions
    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z) {
                double lap_val = lap_host[x*N*N + y*N + z];
                EXPECT_NEAR(lap_val, 6.0, 1e-8)
                    << "Failed at (" << x << "," << y << "," << z << ")";
            }
}

// Test: Laplacian of constant field should be 0
TEST_F(LaplacianTest, ConstantFieldIsZero)
{
    const int N = 16;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total, 42.0);  // constant

    DeviceField<double> d_phi(total);
    DeviceField<double> d_lap(total);
    d_phi.copy_from_host(phi_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_laplacian_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_lap.data(), p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lap_host(total);
    d_lap.copy_to_host(lap_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z) {
                EXPECT_NEAR(lap_host[x*N*N + y*N + z], 0.0, 1e-12);
            }
}

// Test: Laplacian of linear field should be 0
TEST_F(LaplacianTest, LinearFieldIsZero)
{
    const int N = 16;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x*N*N + y*N + z] = 3.0*x*h + 2.0*y*h - 1.0*z*h;

    DeviceField<double> d_phi(total);
    DeviceField<double> d_lap(total);
    d_phi.copy_from_host(phi_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_laplacian_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_lap.data(), p, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lap_host(total);
    d_lap.copy_to_host(lap_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z) {
                EXPECT_NEAR(lap_host[x*N*N + y*N + z], 0.0, 1e-12);
            }
}
