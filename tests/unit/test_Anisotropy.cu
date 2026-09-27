#include "common/Gpu.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class AnisotropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// Test kernel that evaluates compute_An at given gradient values.
__global__ void test_an_kernel(const double* __restrict__ gradients, // [phix, phiy, phiz] x N
                               double* __restrict__ results, double epsilon, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    double phix = gradients[i * 3 + 0];
    double phiy = gradients[i * 3 + 1];
    double phiz = gradients[i * 3 + 2];
    results[i] = compute_An(phix, phiy, phiz, epsilon);
}

__global__ void test_dfunc_kernel(const double* __restrict__ inputs, // [l, m, n] x N
                                  double* __restrict__ results, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    results[i] = dFunc(inputs[i * 3], inputs[i * 3 + 1], inputs[i * 3 + 2]);
}

// Test: An along axis should equal 1 + epsilon (cubic symmetry)
// When gradient is aligned with a crystal axis: (1,0,0), (0,1,0), (0,0,1)
// qrt/sq^2 = 1, so An = (1-3*eps)*(1 + 4*eps/(1-3*eps)) = 1+eps
TEST_F(AnisotropyTest, AlongAxis) {
    double epsilon = 0.07;

    std::vector<double> grads = {
        1.0, 0.0, 0.0, // x-axis
        0.0, 1.0, 0.0, // y-axis
        0.0, 0.0, 1.0, // z-axis
    };

    DeviceField<double> d_grads(grads.size());
    DeviceField<double> d_results(3);
    d_grads.copy_from_host(grads.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_an_kernel<<<1, 3>>>(d_grads.data(), d_results.data(), epsilon, 3);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(3);
    d_results.copy_to_host(results.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    double expected = 1.0 + epsilon;
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(results[i], expected, 1e-12) << "Failed for axis " << i;
    }
}

// Test: An along diagonal should equal 1 - 5/3 * epsilon
// When gradient is (1,1,1)/sqrt(3): qrt/sq^2 = 3*(1/3)^4 / (3*(1/3)^2)^2 = 3/81 / (9/9) = 1/27
// Wait, let me recalculate:
// phix=phiy=phiz=1/sqrt(3), sq = 1, qrt = 3*(1/3)^2 = 3/9 = 1/3
// qrt/sq^2 = 1/3
// An = (1-3*eps)*(1 + 4*eps/(1-3*eps) * 1/3) = (1-3*eps) + 4/3*eps = 1 - 5/3*eps
TEST_F(AnisotropyTest, AlongDiagonal) {
    double epsilon = 0.07;
    double s3 = 1.0 / std::sqrt(3.0);

    std::vector<double> grads = {s3, s3, s3};

    DeviceField<double> d_grads(3);
    DeviceField<double> d_results(1);
    d_grads.copy_from_host(grads.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_an_kernel<<<1, 1>>>(d_grads.data(), d_results.data(), epsilon, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_results.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    double expected = 1.0 - (5.0 / 3.0) * epsilon;
    EXPECT_NEAR(result, expected, 1e-12);
}

// Test: An with zero gradient should return spherical average 1 - 3*epsilon/5
// (Mean of n_x^4+n_y^4+n_z^4 over unit sphere = 3/5)
TEST_F(AnisotropyTest, ZeroGradient) {
    double epsilon = 0.07;
    std::vector<double> grads = {0.0, 0.0, 0.0};

    DeviceField<double> d_grads(3);
    DeviceField<double> d_results(1);
    d_grads.copy_from_host(grads.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_an_kernel<<<1, 1>>>(d_grads.data(), d_results.data(), epsilon, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_results.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    double expected = 1.0 - 3.0 * epsilon / 5.0;
    EXPECT_NEAR(result, expected, 1e-12);
}

// Test: An is always positive for valid epsilon
TEST_F(AnisotropyTest, AlwaysPositive) {
    double epsilon = 0.07;

    // Test many random-ish directions
    std::vector<double> grads;
    int N = 0;
    for (double a = -1.0; a <= 1.0; a += 0.5) {
        for (double b = -1.0; b <= 1.0; b += 0.5) {
            for (double c = -1.0; c <= 1.0; c += 0.5) {
                grads.push_back(a);
                grads.push_back(b);
                grads.push_back(c);
                ++N;
            }
        }
    }

    DeviceField<double> d_grads(grads.size());
    DeviceField<double> d_results(static_cast<std::size_t>(N));
    d_grads.copy_from_host(grads.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cfg = LaunchConfig::for_1d(static_cast<std::size_t>(N));
    test_an_kernel<<<cfg.grid, cfg.block>>>(d_grads.data(), d_results.data(), epsilon, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(static_cast<std::size_t>(N));
    d_results.copy_to_host(results.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < N; ++i) {
        EXPECT_GT(results[i], 0.0) << "An is non-positive for test case " << i;
    }
}

// Test: dFunc(0,0,0) should return 0
TEST_F(AnisotropyTest, dFuncZero) {
    std::vector<double> inputs = {0.0, 0.0, 0.0};

    DeviceField<double> d_inputs(3);
    DeviceField<double> d_results(1);
    d_inputs.copy_from_host(inputs.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_dfunc_kernel<<<1, 1>>>(d_inputs.data(), d_results.data(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_results.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    EXPECT_DOUBLE_EQ(result, 0.0);
}

/// Test kernel for dF_dphi evaluation
__global__ void test_dfdphi_kernel(const double* __restrict__ phi_vals,
                                   const double* __restrict__ u_vals, double lambda,
                                   double* __restrict__ results, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    results[i] = dF_dphi(phi_vals[i], u_vals[i], lambda);
}

// Test: dF/dphi at phi=0 should be lambda*u
TEST_F(AnisotropyTest, dFdphiAtZero) {
    double lambda = 6.383;
    double u_val = -0.3;

    std::vector<double> phi_vals = {0.0};
    std::vector<double> u_vals = {u_val};

    DeviceField<double> d_phi(1), d_u(1), d_results(1);
    d_phi.copy_from_host(phi_vals.data());
    d_u.copy_from_host(u_vals.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_dfdphi_kernel<<<1, 1>>>(d_phi.data(), d_u.data(), lambda, d_results.data(), 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    double result;
    d_results.copy_to_host(&result);
    CUDA_CHECK(cudaDeviceSynchronize());

    // dF_dphi(0, u, lambda) = -0*(1-0) + lambda*u*(1-0)^2 = lambda*u
    EXPECT_NEAR(result, lambda * u_val, 1e-12);
}

// Test: dF/dphi at phi=+/-1 should be 0 (double-well minima)
TEST_F(AnisotropyTest, dFdphiAtMinima) {
    double lambda = 6.383;
    double u_val = -0.3;

    std::vector<double> phi_vals = {1.0, -1.0};
    std::vector<double> u_vals = {u_val, u_val};

    DeviceField<double> d_phi(2), d_u(2), d_results(2);
    d_phi.copy_from_host(phi_vals.data());
    d_u.copy_from_host(u_vals.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    test_dfdphi_kernel<<<1, 2>>>(d_phi.data(), d_u.data(), lambda, d_results.data(), 2);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> results(2);
    d_results.copy_to_host(results.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // At phi=+/-1: omp2 = 1-1 = 0, so dF_dphi = 0
    EXPECT_NEAR(results[0], 0.0, 1e-12);
    EXPECT_NEAR(results[1], 0.0, 1e-12);
}

/// Gradient-energy density of the anisotropic model, f(g) = 1/2 W0^2 A(g)^2 |g|^2
/// with A(g) = (1 - 3 eps) + 4 eps (sum g_i^4) / |g|^4, in long double.
static long double energy_density(const long double g[3], long double W0, long double eps) {
    const long double s = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
    if (s == 0.0L)
        return 0.0L;
    const long double q =
        g[0] * g[0] * g[0] * g[0] + g[1] * g[1] * g[1] * g[1] + g[2] * g[2] * g[2] * g[2];
    const long double a = (1.0L - 3.0L * eps) + 4.0L * eps * q / (s * s);
    return 0.5L * W0 * W0 * a * a * s;
}

__global__ void test_flux_kernel(const double* __restrict__ grads, double* __restrict__ out,
                                 double W0, double eps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const double gx = grads[3 * i], gy = grads[3 * i + 1], gz = grads[3 * i + 2];
    // Cyclic order puts the flux component first (A and dFunc are symmetric
    // in the other two), exactly as anisotropic_divergence calls it.
    out[3 * i + 0] = anisotropic_flux(gx, gy, gz, W0, eps);
    out[3 * i + 1] = anisotropic_flux(gy, gz, gx, W0, eps);
    out[3 * i + 2] = anisotropic_flux(gz, gx, gy, W0, eps);
}

__global__ void test_divergence_kernel(const double* __restrict__ phi, double* __restrict__ div,
                                       double* __restrict__ lap, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<unsigned>(p.Nx * p.Ny * p.Nz))
        return;
    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1)
        return;
    div[tid] = anisotropic_divergence(phi, x, y, z, p);
    lap[tid] = laplacian_7pt(phi, x, y, z, p.Ny, p.Nz, p.dx, p.dy, p.dz);
}

// anisotropic_flux must be the exact variational derivative F = df/dp of the
// gradient energy density, compared with an independent long-double central
// difference of f.
TEST_F(AnisotropyTest, FluxIsVariationalDerivativeOfGradientEnergy) {
    const double W0 = 1.3;
    const std::vector<double> grads = {1.0,  0.0, 0.0, 0.0, -2.0, 0.0,  0.0,  0.0,
                                       0.7,  1.0, 1.0, 0.0, 1.0,  -1.0, 1.0,  0.3,
                                       -1.2, 0.8, 2.1, 0.4, -0.9, -0.5, -0.5, 1.5};
    const int n = static_cast<int>(grads.size() / 3);
    DeviceField<double> d_g(grads.size()), d_f(grads.size());
    d_g.copy_from_host(grads.data());
    std::vector<double> got(grads.size());

    for (double eps : {0.0, 0.05, 0.1, 0.2}) {
        test_flux_kernel<<<1, 64>>>(d_g.data(), d_f.data(), W0, eps, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        d_f.copy_to_host(got.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        for (int k = 0; k < n; ++k) {
            const long double g[3] = {grads[3 * k], grads[3 * k + 1], grads[3 * k + 2]};
            const long double gn = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            const long double eta = 1e-6L * (gn > 1.0L ? gn : 1.0L);
            for (int i = 0; i < 3; ++i) {
                long double gp[3] = {g[0], g[1], g[2]}, gm[3] = {g[0], g[1], g[2]};
                gp[i] += eta;
                gm[i] -= eta;
                const double expected = static_cast<double>(
                    (energy_density(gp, W0, eps) - energy_density(gm, W0, eps)) / (2.0L * eta));
                ASSERT_NEAR(got[3 * k + i], expected, 1e-8 * (1.0 + std::fabs(expected)))
                    << "eps=" << eps << " g=(" << g[0] << "," << g[1] << "," << g[2]
                    << ") component " << i;
            }
        }
    }
}

namespace {

// Smooth test field with |grad phi| bounded away from 0 (the linear part
// dominates the oscillating one), so A(n) is smooth along it.
constexpr long double kK[3] = {1.1L, 0.9L, 0.8L};
constexpr long double kPh[3] = {0.3L, -0.2L, 0.5L};
constexpr long double kAmp = 0.4L;
constexpr long double kLin[3] = {1.0L, 0.5L, 0.3L};

long double field(long double x, long double y, long double z) {
    return kAmp * std::sin(kK[0] * x + kPh[0]) * std::cos(kK[1] * y + kPh[1]) *
               std::sin(kK[2] * z + kPh[2]) +
           kLin[0] * x + kLin[1] * y + kLin[2] * z;
}

/// Exact gradient and Hessian of field().
void field_derivatives(long double x, long double y, long double z, long double g[3],
                       long double H[3][3]) {
    const long double a = kK[0] * x + kPh[0], b = kK[1] * y + kPh[1], c = kK[2] * z + kPh[2];
    const long double sa = std::sin(a), ca = std::cos(a), sb = std::sin(b), cb = std::cos(b),
                      sc = std::sin(c), cc = std::cos(c);
    g[0] = kAmp * kK[0] * ca * cb * sc + kLin[0];
    g[1] = -kAmp * kK[1] * sa * sb * sc + kLin[1];
    g[2] = kAmp * kK[2] * sa * cb * cc + kLin[2];
    H[0][0] = -kAmp * kK[0] * kK[0] * sa * cb * sc;
    H[1][1] = -kAmp * kK[1] * kK[1] * sa * cb * sc;
    H[2][2] = -kAmp * kK[2] * kK[2] * sa * cb * sc;
    H[0][1] = H[1][0] = -kAmp * kK[0] * kK[1] * ca * sb * sc;
    H[0][2] = H[2][0] = kAmp * kK[0] * kK[2] * ca * cb * cc;
    H[1][2] = H[2][1] = -kAmp * kK[1] * kK[2] * sa * sb * cc;
}

/// Exact div F(grad phi) = sum_{d,e} (d^2 f / dp_d dp_e) (d^2 phi / dx_d dx_e),
/// with the Hessian of f from long-double central differences.
long double exact_divergence(long double x, long double y, long double z, long double W0,
                             long double eps) {
    long double g[3], H[3][3];
    field_derivatives(x, y, z, g, H);
    const long double eta = 1e-4L;
    long double sum = 0.0L;
    for (int d = 0; d < 3; ++d)
        for (int e = 0; e < 3; ++e) {
            long double pp[3] = {g[0], g[1], g[2]}, pm[3] = {g[0], g[1], g[2]},
                        mp[3] = {g[0], g[1], g[2]}, mm[3] = {g[0], g[1], g[2]};
            pp[d] += eta, pp[e] += eta;
            pm[d] += eta, pm[e] -= eta;
            mp[d] -= eta, mp[e] += eta;
            mm[d] -= eta, mm[e] -= eta;
            const long double fde = (energy_density(pp, W0, eps) - energy_density(pm, W0, eps) -
                                     energy_density(mp, W0, eps) + energy_density(mm, W0, eps)) /
                                    (4.0L * eta * eta);
            sum += fde * H[d][e];
        }
    return sum;
}

struct DivergenceRun {
    double max_err = 0.0;      // vs exact div F
    double max_lap_diff = 0.0; // |div - W0^2 * 7-point Laplacian| / rounding scale
};

DivergenceRun run_divergence(int N, double L, double W0, double eps) {
    const double h = L / (N - 1);
    const std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> phi(total), div(total), lap(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                phi[(static_cast<std::size_t>(x) * N + y) * N + z] =
                    static_cast<double>(field(x * h, y * h, z * h));
    DeviceField<double> d_phi(total), d_div(total), d_lap(total);
    d_phi.copy_from_host(phi.data());
    KernelParams p{};
    p.Nx = p.Ny = p.Nz = N;
    p.dx = p.dy = p.dz = h;
    p.W0 = W0;
    p.epsilon = eps;
    auto cfg = LaunchConfig::for_1d(total, 256);
    test_divergence_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_div.data(), d_lap.data(), p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_div.copy_to_host(div.data());
    d_lap.copy_to_host(lap.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    DivergenceRun r;
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                const std::size_t c = (static_cast<std::size_t>(x) * N + y) * N + z;
                const double exact =
                    static_cast<double>(exact_divergence(x * h, y * h, z * h, W0, eps));
                r.max_err = std::max(r.max_err, std::fabs(div[c] - exact));
                // Rounding scale of either evaluation: W0^2 * sum |phi| / h^2 over
                // the 7-point stencil; both are sums of O(10) such terms.
                double scale = 6.0 * std::fabs(phi[c]);
                const std::size_t sx = static_cast<std::size_t>(N) * N, sy = N;
                for (std::size_t o : {sx, sy, std::size_t{1}})
                    scale += std::fabs(phi[c + o]) + std::fabs(phi[c - o]);
                scale *= W0 * W0 / (h * h);
                r.max_lap_diff =
                    std::max(r.max_lap_diff, std::fabs(div[c] - W0 * W0 * lap[c]) / scale);
            }
    return r;
}

} // namespace

// At eps = 0 the face-flux divergence is W0^2 times the 7-point Laplacian up
// to rounding, i.e. the isotropic operator has no 2h-wide stencil left. The
// bound is 32 unit roundoffs of the stencil's magnitude W0^2 sum|phi|/h^2
// (each side sums ~10 rounded terms of that size).
TEST_F(AnisotropyTest, DivergenceAtZeroAnisotropyIsSevenPointLaplacian) {
    const auto r = run_divergence(17, 3.0, 1.3, 0.0);
    EXPECT_LT(r.max_lap_diff, 32.0 * std::numeric_limits<double>::epsilon());
}

// The face-flux divergence is a second-order approximation of div F(grad phi)
// for the full anisotropic flux: halving h divides the maximum error over all
// interior cells by ~4 (observed order > 1.9) for every eps tested, including
// strong anisotropy.
TEST_F(AnisotropyTest, DivergenceIsSecondOrderAccurate) {
    const double L = 3.0, W0 = 1.3;
    for (double eps : {0.0, 0.05, 0.12, 0.2}) {
        SCOPED_TRACE("eps=" + std::to_string(eps));
        const double e1 = run_divergence(13, L, W0, eps).max_err;
        const double e2 = run_divergence(25, L, W0, eps).max_err;
        const double e3 = run_divergence(49, L, W0, eps).max_err;
        const double order12 = std::log2(e1 / e2), order23 = std::log2(e2 / e3);
        std::printf(
            "eps=%.2f  max|err| h=L/12: %.3e  h=L/24: %.3e  h=L/48: %.3e  order %.3f %.3f\n", eps,
            e1, e2, e3, order12, order23);
        EXPECT_GT(order12, 1.9);
        EXPECT_GT(order23, 1.9);
        EXPECT_LT(order23, 2.1);
    }
}
