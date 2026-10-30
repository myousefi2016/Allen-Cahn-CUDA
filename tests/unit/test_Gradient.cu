#include "common/Gpu.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ac;
using namespace ac::cuda;

// Test kernel: computes 2nd-order gradient at interior points
// direction: 0=x, 1=y, 2=z
__global__ void test_gradient_2nd_kernel(const double* __restrict__ phi,
                                         double* __restrict__ grad_out, int Nx, int Ny, int Nz,
                                         double dx, double dy, double dz, int direction) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (tid >= static_cast<unsigned>(total))
        return;

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
__global__ void test_gradient_4th_kernel(const double* __restrict__ phi,
                                         double* __restrict__ grad_out, int Nx, int Ny, int Nz,
                                         double dx, double dy, double dz, int direction) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Nx * Ny * Nz;
    if (tid >= static_cast<unsigned>(total))
        return;

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

// Test kernel: the six helpers at every cell, out[k * total + c] for k = 0..5 =
// gradient_x, _y, _z of q and gradient_x_4th, _y_4th, _z_4th of c (0 where the
// helper's stencil does not fit).
__global__ void test_all_gradients_kernel(const double* __restrict__ q,
                                          const double* __restrict__ c, double* __restrict__ out,
                                          int Nx, int Ny, int Nz, double dx, double dy, double dz) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = Nx * Ny * Nz;
    if (tid >= static_cast<unsigned>(total))
        return;
    int x, y, z;
    linear_to_3d(static_cast<int>(tid), Ny, Nz, x, y, z);
    double g[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    if (x >= 1 && x < Nx - 1 && y >= 1 && y < Ny - 1 && z >= 1 && z < Nz - 1) {
        g[0] = gradient_x(q, x, y, z, Ny, Nz, dx);
        g[1] = gradient_y(q, x, y, z, Ny, Nz, dy);
        g[2] = gradient_z(q, x, y, z, Ny, Nz, dz);
    }
    if (x >= 2 && x < Nx - 2 && y >= 2 && y < Ny - 2 && z >= 2 && z < Nz - 2) {
        g[3] = gradient_x_4th(c, x, y, z, Ny, Nz, dx);
        g[4] = gradient_y_4th(c, x, y, z, Ny, Nz, dy);
        g[5] = gradient_z_4th(c, x, y, z, Ny, Nz, dz);
    }
    for (int k = 0; k < 6; ++k)
        out[static_cast<std::size_t>(k) * total + tid] = g[k];
}

class GradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }

    int idx(int x, int y, int z, int Ny, int Nz) const { return x * Ny * Nz + y * Nz + z; }
};

// phi = 3*x*dx -> gradient_x = 3.0
TEST_F(GradientTest, LinearFieldX_2nd) {
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
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      0);
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
TEST_F(GradientTest, LinearFieldY_2nd) {
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
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      1);
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
TEST_F(GradientTest, LinearFieldZ_2nd) {
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
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      2);
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
TEST_F(GradientTest, QuadraticFieldX_2nd) {
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
    test_gradient_2nd_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      0);
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
TEST_F(GradientTest, LinearFieldX_4th) {
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
    test_gradient_4th_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      0);
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
TEST_F(GradientTest, CubicFieldX_4th) {
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
    test_gradient_4th_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_grad.data(), N, N, N, h, h, h,
                                                      0);
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

// Each helper divides by the spacing of its own axis. The tests above use h = 1 on all
// three axes, where / (2 h) and / 2 give the same bits and any axis's spacing stands for
// another's, and they run the 4th-order helpers along x only. Here dx, dy, dz = 3/8, 5/8,
// 7/4 (none 1, all different) on a 9 x 10 x 11 grid. q and c are polynomials with integer
// coefficients in X = x dx, Y = y dy, Z = z dz, so every value, every difference and every
// derivative is a dyadic rational of at most 23 significant bits (denominator <= 512,
// |value| < 2^14), exactly representable on host and GPU. The central difference is exact
// for the quadratic q and the five-point one for the cubic c (polynomial identities). 2 h
// and 12 h (0.75, 1.25, 3.5, 4.5, 7.5, 21) and 8 phi are exact, so FMA contraction changes
// nothing, and CUDA's double division (IEEE, round to nearest; -prec-div and
// --use_fast_math affect float only) returns a representable quotient exactly. Every
// helper must therefore return the analytic derivative bit for bit.
TEST_F(GradientTest, EveryHelperDividesByItsOwnAxisSpacing) {
    const int Nx = 9, Ny = 10, Nz = 11;
    const double dx = 0.375, dy = 0.625, dz = 1.75;
    const std::size_t total = static_cast<std::size_t>(Nx) * Ny * Nz;

    // q = X^2 + 3XY - 2Y^2 + YZ + 5Z^2 - 7X + 2,  c = X^3 - 2X^2 Y + XYZ + 3Y^3 - Z^3 + 4YZ^2
    std::vector<double> q_h(total), c_h(total);
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                const double X = x * dx, Y = y * dy, Z = z * dz;
                q_h[idx(x, y, z, Ny, Nz)] =
                    X * X + 3.0 * X * Y - 2.0 * Y * Y + Y * Z + 5.0 * Z * Z - 7.0 * X + 2.0;
                c_h[idx(x, y, z, Ny, Nz)] = X * X * X - 2.0 * X * X * Y + X * Y * Z +
                                            3.0 * Y * Y * Y - Z * Z * Z + 4.0 * Y * Z * Z;
            }

    DeviceField<double> d_q(total), d_c(total), d_out(6 * total);
    d_q.copy_from_host(q_h.data());
    d_c.copy_from_host(c_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto cfg = LaunchConfig::for_1d(total);
    test_all_gradients_kernel<<<cfg.grid, cfg.block>>>(d_q.data(), d_c.data(), d_out.data(), Nx, Ny,
                                                       Nz, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> out(6 * total);
    d_out.copy_to_host(out.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    const char* const name[6] = {"gradient_x",     "gradient_y",     "gradient_z",
                                 "gradient_x_4th", "gradient_y_4th", "gradient_z_4th"};
    int checked[6] = {}, wrong[6] = {};
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                const double X = x * dx, Y = y * dy, Z = z * dz;
                const double exact[6] = {2.0 * X + 3.0 * Y - 7.0,
                                         3.0 * X - 4.0 * Y + Z,
                                         Y + 10.0 * Z,
                                         3.0 * X * X - 4.0 * X * Y + Y * Z,
                                         -2.0 * X * X + X * Z + 9.0 * Y * Y + 4.0 * Z * Z,
                                         X * Y - 3.0 * Z * Z + 8.0 * Y * Z};
                const bool fits2 =
                    x >= 1 && x < Nx - 1 && y >= 1 && y < Ny - 1 && z >= 1 && z < Nz - 1;
                const bool fits4 =
                    x >= 2 && x < Nx - 2 && y >= 2 && y < Ny - 2 && z >= 2 && z < Nz - 2;
                for (int k = 0; k < 6; ++k) {
                    if (!(k < 3 ? fits2 : fits4))
                        continue;
                    ++checked[k];
                    const double got =
                        out[static_cast<std::size_t>(k) * total + idx(x, y, z, Ny, Nz)];
                    if (got != exact[k] && wrong[k]++ < 3)
                        ADD_FAILURE() << name[k] << " at (" << x << "," << y << "," << z
                                      << ") = " << got << ", exact " << exact[k];
                }
            }
    for (int k = 0; k < 6; ++k) {
        EXPECT_EQ(checked[k], k < 3 ? 7 * 8 * 9 : 5 * 6 * 7) << name[k];
        EXPECT_EQ(wrong[k], 0) << name[k];
    }
}
