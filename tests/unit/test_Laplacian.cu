#include "common/Gpu.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class LaplacianTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// Kernel that evaluates the Laplacian at every interior point.
__global__ void test_laplacian_kernel(const double* __restrict__ phi, double* __restrict__ lap,
                                      KernelParams p, int stencil_type) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.Nx * p.Ny * p.Nz;
    if (tid >= static_cast<unsigned>(total))
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        lap[c] = 0.0;
        return;
    }

    p.stencil_type = stencil_type;
    lap[c] = laplacian(phi, x, y, z, p);
}

// Test: Laplacian of f(x,y,z) = x^2 + y^2 + z^2 should be 6.0 (exactly for 2nd order)
TEST_F(LaplacianTest, QuadraticField7pt) {
    const int N = 32;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;
    p.stencil_type = 0;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    // f(x,y,z) = (x*dx)^2 + (y*dy)^2 + (z*dz)^2
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x * N * N + y * N + z] =
                    (x * h) * (x * h) + (y * h) * (y * h) + (z * h) * (z * h);

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
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double lap_val = lap_host[x * N * N + y * N + z];
                EXPECT_NEAR(lap_val, 6.0, 1e-10)
                    << "Failed at (" << x << "," << y << "," << z << ")";
            }
}

// Test: 27-point Laplacian on quadratic field (should also give 6.0, isotropic)
TEST_F(LaplacianTest, QuadraticField27pt) {
    const int N = 32;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;
    p.stencil_type = 1;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x * N * N + y * N + z] =
                    (x * h) * (x * h) + (y * h) * (y * h) + (z * h) * (z * h);

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
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double lap_val = lap_host[x * N * N + y * N + z];
                EXPECT_NEAR(lap_val, 6.0, 1e-8)
                    << "Failed at (" << x << "," << y << "," << z << ")";
            }
}

// The 27-point stencil read off as its response to a unit impulse. The
// Laplacian at cell c0 - e of the field that is 1 at c0 and 0 elsewhere is the
// weight w_e of the neighbour at offset e. Two kinds of check follow.
//
// 1. The weights are those of docs/DESIGN.md 2.2 (Patra-Karttunen): 14 for the
//    6 faces, 3 for the 12 edges, 1 for the 8 corners and -128 for the centre,
//    over 30 h^2. The stencil reaches no farther than +-1 cell.
//
// 2. The moments that make it a Laplacian with an isotropic h^2 error.
//    On exp(i k.x) the stencil gives sum_e w_e cos(theta.e), theta = k h:
//      M0 - (M2/2)|theta|^2
//        + (M4 sum_a theta_a^4 + 6 M22 sum_{a<b} theta_a^2 theta_b^2)/24
//        + O(theta^6),
//    where
//      M0 = sum_e w_e,
//      M2 = sum_e w_e e_a^2,
//      M4 = sum_e w_e e_a^4,
//      M22 = sum_e w_e e_a^2 e_b^2 (a != b).
//    The odd moments and sum_e w_e e_a e_b vanish by symmetry.
//    - Consistency needs M0 = 0 and M2 = 2/h^2; the leading term is then -|k|^2.
//    - The h^2 term depends on |k| alone (it is the same along <100>, <110>
//      and <111>) exactly when it is a multiple of
//      |theta|^4 = sum theta_a^4 + 2 sum theta_a^2 theta_b^2, that is when
//      M4 = 3 M22.
//    - Patra-Karttunen has M4 = 60 and M22 = 20 over 30 h^2, so its error is
//      h^2 |k|^4 / 12 in every direction. The 7-point stencil has M22 = 0.
//
// Rounding:
// - h = 0.5 makes 30 h^2 = 7.5 exact. The numerator is an exact integer, so
//   every weight is one correctly rounded quotient n/7.5 on the GPU (double
//   '/' is IEEE) and on the host: the weights are compared bitwise.
// - A moment sums the 27 weights times c_e in {0, +-1, -2}, and those products
//   are exact. One rounding in each weight plus 26 in the recursive sum put it
//   within gamma_28 sum_e |c_e w_e| of its exact value (Higham, Accuracy and
//   Stability of Numerical Algorithms, 2nd ed., Lemma 3.1 and eq. 4.4).
TEST_F(LaplacianTest, IsotropicStencilWeightsAndMoments) {
    const int N = 7, c0 = 3; // the impulse's 3x3x3 neighbourhood is interior
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;
    p.stencil_type = 1;

    const std::size_t total = static_cast<std::size_t>(N) * N * N;
    auto at = [&](int x, int y, int z) {
        return (static_cast<std::size_t>(x) * N + static_cast<std::size_t>(y)) * N +
               static_cast<std::size_t>(z);
    };
    std::vector<double> phi_host(total, 0.0);
    phi_host[at(c0, c0, c0)] = 1.0;

    DeviceField<double> d_phi(total);
    DeviceField<double> d_lap(total);
    d_phi.copy_from_host(phi_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(total);
    test_laplacian_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_lap.data(), p, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> lap(total);
    d_lap.copy_to_host(lap.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reach: exactly 0 at every interior cell more than one cell from the impulse.
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z)
                if (std::abs(x - c0) > 1 || std::abs(y - c0) > 1 || std::abs(z - c0) > 1)
                    EXPECT_EQ(lap[at(x, y, z)], 0.0)
                        << "the stencil reaches (" << x << "," << y << "," << z << ")";

    // w[e] = lap(c0 - e), e in {-1, 0, 1}^3.
    double w[3][3][3];
    const double divisor = 30.0 * h * h; // 7.5, exact
    for (int ex = -1; ex <= 1; ++ex)
        for (int ey = -1; ey <= 1; ++ey)
            for (int ez = -1; ez <= 1; ++ez) {
                const int nonzero = std::abs(ex) + std::abs(ey) + std::abs(ez);
                const double weight = nonzero == 0   ? -128.0
                                      : nonzero == 1 ? 14.0
                                      : nonzero == 2 ? 3.0
                                                     : 1.0;
                const double got = lap[at(c0 - ex, c0 - ey, c0 - ez)];
                w[ex + 1][ey + 1][ez + 1] = got;
                EXPECT_EQ(got, weight / divisor)
                    << "weight of offset (" << ex << "," << ey << "," << ez << ")";
            }

    const double unit = std::numeric_limits<double>::epsilon() / 2.0;
    const double gamma28 = 28.0 * unit / (1.0 - 28.0 * unit);
    // sum_e c(e) w_e and its rounding bound gamma_28 sum_e |c(e) w_e|.
    auto moment = [&](auto c) {
        double sum = 0.0, abs_sum = 0.0;
        for (int ex = -1; ex <= 1; ++ex)
            for (int ey = -1; ey <= 1; ++ey)
                for (int ez = -1; ez <= 1; ++ez) {
                    const int e[3] = {ex, ey, ez};
                    const double term = c(e) * w[ex + 1][ey + 1][ez + 1];
                    sum += term;
                    abs_sum += std::fabs(term);
                }
        return std::pair{sum, gamma28 * abs_sum};
    };

    const auto [m0, m0_bound] = moment([](const int*) { return 1.0; });
    EXPECT_LE(std::fabs(m0), m0_bound) << "M0 (the Laplacian of a constant)";
    for (int a = 0; a < 3; ++a) {
        const auto [m2, m2_bound] =
            moment([a](const int* e) { return static_cast<double>(e[a] * e[a]); });
        EXPECT_LE(std::fabs(m2 - 2.0 / (h * h)), m2_bound) << "M2 along axis " << a;
        for (int b = 0; b < 3; ++b) {
            if (b == a)
                continue;
            const auto [m11, m11_bound] =
                moment([a, b](const int* e) { return static_cast<double>(e[a] * e[b]); });
            EXPECT_LE(std::fabs(m11), m11_bound) << "mixed second moment, axes " << a << b;
            // M4_a - 3 M22_ab with e_a^4 = e_a^2: c = e_a^2 (1 - 3 e_b^2), in {0, 1, -2}.
            const auto [iso, iso_bound] = moment([a, b](const int* e) {
                return static_cast<double>(e[a] * e[a] * (1 - 3 * e[b] * e[b]));
            });
            EXPECT_LE(std::fabs(iso), iso_bound)
                << "M4 - 3 M22, axes " << a << ", " << b << ": the h^2 error is anisotropic";
        }
    }
}

// Test: Laplacian of constant field should be 0
TEST_F(LaplacianTest, ConstantFieldIsZero) {
    const int N = 16;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total, 42.0); // constant

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

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(lap_host[x * N * N + y * N + z], 0.0, 1e-12);
            }
}

// Test: Laplacian of linear field should be 0
TEST_F(LaplacianTest, LinearFieldIsZero) {
    const int N = 16;
    const double h = 0.5;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi_host(total);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi_host[x * N * N + y * N + z] = 3.0 * x * h + 2.0 * y * h - 1.0 * z * h;

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

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(lap_host[x * N * N + y * N + z], 0.0, 1e-12);
            }
}
