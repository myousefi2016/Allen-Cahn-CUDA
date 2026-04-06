#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ac;
using namespace ac::cuda;

// Test kernel: computes 2nd-order gradient at interior points
// direction: 0=x, 1=y, 2=z
__global__ void test_gradient_2nd_kernel(
    const double* __restrict__ phi,
    double* __restrict__ grad_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    int direction)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (tid >= static_cast<unsigned>(total)) return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), Ny, Nz, x, y, z);
    int c = idx3d(x, y, z, Ny, Nz);

    if (x < 1 || x >= Nx - 1 || y < 1 || y >= Ny - 1 || z < 1 || z >= Nz - 1) {
        grad_out[c] = 0.0;
        return;
    }

    if (direction == 0)
        grad_out[c] = gradient_x(phi, x, y, z, Ny, Nz, dx);
    else if (direction == 1)
        grad_out[c] = gradient_y(phi, x, y, z, Ny, Nz, dy);
    else
        grad_out[c] = gradient_z(phi, x, y, z, Ny, Nz, dz);
}

// Test kernel: computes 4th-order gradient at interior points (needs +-2 neighbors)
__global__ void test_gradient_4th_kernel(
    const double* __restrict__ phi,
    double* __restrict__ grad_out,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    int direction)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (tid >= static_cast<unsigned>(total)) return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), Ny, Nz, x, y, z);
    int c = idx3d(x, y, z, Ny, Nz);

    if (x < 2 || x >= Nx - 2 || y < 2 || y >= Ny - 2 || z < 2 || z >= Nz - 2) {
        grad_out[c] = 0.0;
        return;
    }

    if (direction == 0)
        grad_out[c] = gradient_x_4th(phi, x, y, z, Ny, Nz, dx);
    else if (direction == 1)
        grad_out[c] = gradient_y_4th(phi, x, y, z, Ny, Nz, dy);
    else
        grad_out[c] = gradient_z_4th(phi, x, y, z, Ny, Nz, dz);
}

class GradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    int idx(int x, int y, int z, int Ny, int Nz) const {
        return x * Ny * Nz + y * Nz + z;
    }
};

// phi = 3*x*dx -> gradient_x = 3.0
TEST_F(GradientTest, LinearFieldX_2nd)
{
    const int N = 16;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_h[idx(x, y, z, N, N)] = 3.0 * x * h;

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], 3.0, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi = 2*y*dy -> gradient_y = 2.0
TEST_F(GradientTest, LinearFieldY_2nd)
{
    const int N = 16;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_h[idx(x, y, z, N, N)] = 2.0 * y * h;

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], 2.0, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi = 5*z*dz -> gradient_z = 5.0
TEST_F(GradientTest, LinearFieldZ_2nd)
{
    const int N = 16;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_h[idx(x, y, z, N, N)] = 5.0 * z * h;

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], 5.0, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi = (x*dx)^2 -> gradient_x = 2*x*dx (central diff is exact for quadratic)
TEST_F(GradientTest, QuadraticFieldX_2nd)
{
    const int N = 16;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_h[idx(x, y, z, N, N)] = (x * h) * (x * h);

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2nd-order central diff of x^2: ((x+1)^2 - (x-1)^2) / (2*h) = 2x (exact)
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double expected = 2.0 * x * h;
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], expected, 1e-10)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi = 3*x*dx -> gradient_x_4th = 3.0 (exact for linear)
TEST_F(GradientTest, LinearFieldX_4th)
{
    const int N = 32;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_h[idx(x, y, z, N, N)] = 3.0 * x * h;

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_4th_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 2; x < N - 2; ++x)
        for (int y = 2; y < N - 2; ++y)
            for (int z = 2; z < N - 2; ++z) {
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], 3.0, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi = (x*dx)^3 -> exact derivative = 3*(x*dx)^2
// 4th-order stencil should be exact for cubic polynomials
TEST_F(GradientTest, CubicFieldX_4th)
{
    const int N = 32;
    const double h = 1.0;
    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> phi_h(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                double xv = x * h;
                phi_h[idx(x, y, z, N, N)] = xv * xv * xv;
            }

    DeviceField<double> d_phi(total), d_grad(total);
    d_phi.copy_from_host(phi_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_gradient_4th_kernel<<<cfg.grid, cfg.block>>>(
        d_phi.data(), d_grad.data(), N, N, N, h, h, h, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> grad_h(total);
    d_grad.copy_to_host(grad_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4th-order central difference is exact for polynomials up to degree 3
    // d/dx(x^3) = 3x^2
    for (int x = 2; x < N - 2; ++x)
        for (int y = 2; y < N - 2; ++y)
            for (int z = 2; z < N - 2; ++z) {
                double xv = x * h;
                double expected = 3.0 * xv * xv;
                EXPECT_NEAR(grad_h[idx(x, y, z, N, N)], expected, 1e-8)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}
