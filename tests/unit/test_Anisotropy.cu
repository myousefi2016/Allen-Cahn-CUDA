#include "common/Gpu.hpp"
#include "core/KernelRanges.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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

/// tau = (tau0 A) A as allen_cahn_tau computes it, for each gradient.
__global__ void test_tau_kernel(const double* __restrict__ gradients, double* __restrict__ an,
                                double* __restrict__ tau, double epsilon, double tau0, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;
    const double a =
        compute_An(gradients[i * 3], gradients[i * 3 + 1], gradients[i * 3 + 2], epsilon);
    an[i] = a;
    tau[i] = tau0 * a * a;
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

// Test: An along diagonal should equal 1 - 5/3 * epsilon, the minimum of An
// (qrt/sq^2 >= 1/3 for every nonzero gradient, by Cauchy-Schwarz; the
// zero-gradient value 1 - 3/5 * epsilon is larger)
// When gradient is (1,1,1)/sqrt(3): phix=phiy=phiz=1/sqrt(3), sq = 1,
// qrt = 3*(1/3)^2 = 3/9 = 1/3, qrt/sq^2 = 1/3
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
// validate() bounds the relaxation time of every orientation with
// relaxation_time_range (core/KernelRanges.cpp), derived from compute_An's
// operations: every A the GPU computes, for the axes, face and body diagonals
// and 200000 random gradients from just above the zero-gradient threshold to
// 1e70 (also with subnormal components), lies in [a_lo, a_hi], and at the
// largest tau0 whose range stays finite no GPU tau overflows.
TEST_F(AnisotropyTest, RelaxationTimeRangeContainsTheKernelsValues) {
    std::vector<double> grads;
    for (const double m : {1.0000000000000001e-15, 1e-10, 1e-3, 0.4, 1.0, 7.0, 1e5, 1e40, 1e70}) {
        for (const double v : {m, 0.0, 0.0, 0.0, -m, 0.0,    m,      m, 0.0,        m,  m,
                               m, -m,  m,   -m,  m,  1e-300, 5e-324, m, m * 1e-160, 0.0})
            grads.push_back(v);
    }
    std::uint64_t state = 20261030;
    auto uniform = [&state] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0;
    };
    for (int i = 0; i < 200000; ++i) {
        const double scale = std::pow(10.0, -14.0 + 84.0 * uniform());
        for (int d = 0; d < 3; ++d)
            grads.push_back(scale * (2.0 * uniform() - 1.0));
    }
    const int N = static_cast<int>(grads.size() / 3);
    DeviceField<double> d_grads(grads.size()), d_an(N), d_tau(N);
    d_grads.copy_from_host(grads.data());
    const auto cfg = LaunchConfig::for_1d(static_cast<std::size_t>(N), 256);
    for (const double eps : {0.0, 0.05, 1.0 / 15.0, 0.12, 0.3, std::nextafter(1.0 / 3.0, 0.0)}) {
        SCOPED_TRACE(eps);
        // The largest tau0 whose range stays finite.
        double tau0 = std::numeric_limits<double>::max() / ((1.0 + eps) * (1.0 + eps));
        while (!(relaxation_time_range(tau0, eps).hi <= std::numeric_limits<double>::max()))
            tau0 = std::nextafter(tau0, 0.0);
        const RelaxationTimeRange r = relaxation_time_range(tau0, eps);
        test_tau_kernel<<<cfg.grid, cfg.block>>>(d_grads.data(), d_an.data(), d_tau.data(), eps,
                                                 tau0, N);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> an(N), tau(N);
        d_an.copy_to_host(an.data());
        d_tau.copy_to_host(tau.data());
        int outside = 0;
        for (int i = 0; i < N; ++i) {
            if (!(an[i] >= r.a_lo && an[i] <= r.a_hi) || !std::isfinite(tau[i]) ||
                !(tau[i] >= r.lo && tau[i] <= r.hi)) {
                if (outside++ < 5)
                    ADD_FAILURE() << "gradient " << i << ": A = " << an[i] << ", tau = " << tau[i]
                                  << " outside A in [" << r.a_lo << ", " << r.a_hi << "]";
            }
        }
        EXPECT_EQ(outside, 0);
    }
}

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
    // The kernel writes the interior cells only; the copies below read every cell.
    d_div.zero_async();
    d_lap.zero_async();
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

namespace {

/// gamma_k = k u / (1 - k u): a product of k factors (1 + d)^(+-1), |d| <= u, lies within
/// gamma_k of 1 (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., Lemma 3.1).
long double gamma_k(int k, long double u) {
    return k * u / (1.0L - k * u);
}

/// A(p) = 1 - 3 eps + 4 eps (p_x^4 + p_y^4 + p_z^4) / |p|^4, the cubic anisotropy, p != 0.
long double anisotropy_ld(const long double p[3], long double eps) {
    const long double s = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    const long double q =
        p[0] * p[0] * p[0] * p[0] + p[1] * p[1] * p[1] * p[1] + p[2] * p[2] * p[2] * p[2];
    return 1.0L - 3.0L * eps + 4.0L * eps * q / (s * s);
}

/// Component d of F(p) = w^2 p + 16 eps W0 w dFunc (the flux anisotropic_flux documents;
/// FluxIsVariationalDerivativeOfGradientEnergy checks it against df/dp), w = W0 A(p), and
/// the magnitude M = |w^2 p_d| + 16 eps W0 |w| (|t1| + |t2|) / |p|^4 >= |F_d|, with
/// t1 = p_d^3 (p_a^2 + p_b^2), t2 = p_d (p_a^4 + p_b^4), which bounds the kernel's rounding.
struct FluxRef {
    long double F, M;
};
FluxRef flux_ref(const long double p[3], int d, long double W0, long double eps) {
    const long double pd = p[d], pa = p[(d + 1) % 3], pb = p[(d + 2) % 3];
    const long double w = W0 * anisotropy_ld(p, eps);
    const long double s = pd * pd + pa * pa + pb * pb;
    const long double t1 = pd * pd * pd * (pa * pa + pb * pb);
    const long double t2 = pd * (pa * pa * pa * pa + pb * pb * pb * pb);
    const long double c = 16.0L * eps * W0 * w / (s * s);
    return {w * w * pd + c * (t1 - t2),
            std::fabs(w * w * pd) + std::fabs(c) * (std::fabs(t1) + std::fabs(t2))};
}

} // namespace

/// allen_cahn_tau and anisotropic_divergence at the interior cells (0 on the boundary).
__global__ void test_tau_divergence_kernel(const double* __restrict__ phi, double* __restrict__ tau,
                                           double* __restrict__ div, KernelParams p) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= static_cast<unsigned>(p.Nx * p.Ny * p.Nz))
        return;
    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        tau[tid] = div[tid] = 0.0;
        return;
    }
    tau[tid] = allen_cahn_tau(phi, x, y, z, p);
    div[tid] = anisotropic_divergence(phi, x, y, z, p);
}

namespace {

/// phi on the host in long double: its value at a cell, and the central difference along
/// axis e (exact here). Member functions, not lambdas: under nvcc's --expt-relaxed-constexpr
/// a lambda that could be constexpr is also compiled for the device, where long double is
/// double (warning 20208-D).
struct HostField {
    const std::vector<double>& phi;
    const int (&N)[3];
    const double (&h)[3];

    long double value(const int c[3]) const {
        const std::size_t i =
            (static_cast<std::size_t>(c[0]) * N[1] + static_cast<std::size_t>(c[1])) * N[2] +
            static_cast<std::size_t>(c[2]);
        return static_cast<long double>(phi[i]);
    }
    long double central(const int c[3], int e) const {
        int a[3] = {c[0], c[1], c[2]}, b[3] = {c[0], c[1], c[2]};
        ++a[e];
        --b[e];
        return (value(a) - value(b)) / (2.0L * h[e]);
    }
};

} // namespace

// allen_cahn_tau and anisotropic_divergence take each axis's spacing from KernelParams.
// Every test above runs them with dx = dy = dz. There, tau0 A(grad phi)^2 cannot tell
// / (2 h) from / 2, because A depends only on the direction of the gradient, nor one axis's
// spacing from another's, and the divergence cannot tell its tangential 0.25 / h_d apart.
// Here dx, dy, dz = 1/2, 1/4, 1/8 on a 7 x 9 x 8 grid, and phi is a cubic with integer
// coefficients in X = x dx, Y = y dy, Z = z dz. Every phi value is a dyadic rational of
// at most 23 significant bits. Every gradient the kernels form (central differences
// / (2 h), compact face differences / h, tangential sums * 0.25 / h, h a power of two) is
// then computed exactly and equals the host's long-double value. What remains is the
// rounding of A, of the flux and of the sums, bounded per cell with u = 2^-53 and every
// operation counted unfused (a fused multiply-add has one rounding fewer):
//   tau: sq 3 roundings, qrt 5, q = qrt / sq^2 13, 1 - 3 eps 2 (for eps <= 1/6),
//        c = 4 eps / (1 - 3 eps) 3, c q 17, 1 + c q 18, A = (1 - 3 eps)(1 + c q) 21,
//        tau0 A A 44:  |tau - tau_exact| <= gamma_44 tau_exact.
//   div: w = W0 A 22, w w p_d 46; dFunc within gamma_14 (|t1| + |t2|) / |p|^4 (t1, t2 5
//        each, their difference 6, |p|^4 7, the division 1); 16 eps W0 w dFunc gamma_39 of
//        its magnitude; F_d = their sum: |F - F_exact| <= gamma_47 M; the face difference
//        gamma_48 (M+ + M-); / h_d exact (a power of two); the sum of the three axes
//        gamma_50:  |div - div_exact| <= gamma_50 sum_d (M_d+ + M_d-) / h_d.
// The long-double reference rounds no more often than this, so its own error is the same
// bound with u_L = LDBL_EPSILON / 2. The rounding of the magnitudes, of the difference
// and of the tolerance itself (fewer than 100 long-double roundings in all) is covered by
// the factor 1 / (1 - gamma_100(u_L)).
TEST_F(AnisotropyTest, TauAndDivergenceUseTheSpacingOfEachAxis) {
    const int N[3] = {7, 9, 8};
    const double h[3] = {0.5, 0.25, 0.125};
    const double W0 = 1.3, tau0 = 0.75;
    const std::size_t total = static_cast<std::size_t>(N[0]) * N[1] * N[2];
    auto at = [&N](int x, int y, int z) {
        return (static_cast<std::size_t>(x) * N[1] + y) * N[2] + z;
    };
    std::vector<double> phi(total);
    for (int x = 0; x < N[0]; ++x)
        for (int y = 0; y < N[1]; ++y)
            for (int z = 0; z < N[2]; ++z) {
                const double X = x * h[0], Y = y * h[1], Z = z * h[2];
                phi[at(x, y, z)] = X * X * X - 2.0 * X * X * Y + X * Y * Z + 3.0 * Y * Y * Y -
                                   Z * Z * Z + 4.0 * Y * Z * Z;
            }
    DeviceField<double> d_phi(total), d_tau(total), d_div(total);
    d_phi.copy_from_host(phi.data());

    const long double uD = std::numeric_limits<double>::epsilon() / 2.0L;
    const long double uL = std::numeric_limits<long double>::epsilon() / 2.0L;
    const long double host = 1.0L / (1.0L - gamma_k(100, uL));
    // compute_An and dFunc take their nonzero branch: the rounded sq (within gamma_3 of
    // the exact one) exceeds 1e-30.
    const long double sq_min = 1e-30L / (1.0L - gamma_k(3, uD));
    const HostField field{phi, N, h};

    for (const double eps : {0.05, 0.12}) {
        SCOPED_TRACE("eps=" + std::to_string(eps));
        KernelParams p{};
        p.Nx = N[0];
        p.Ny = N[1];
        p.Nz = N[2];
        p.dx = h[0];
        p.dy = h[1];
        p.dz = h[2];
        p.W0 = W0;
        p.epsilon = eps;
        p.tau0 = tau0;
        const auto cfg = LaunchConfig::for_1d(total, 256);
        test_tau_divergence_kernel<<<cfg.grid, cfg.block>>>(d_phi.data(), d_tau.data(),
                                                            d_div.data(), p);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> tau(total), div(total);
        d_tau.copy_to_host(tau.data());
        d_div.copy_to_host(div.data());
        CUDA_CHECK(cudaDeviceSynchronize());

        int cells = 0, tau_out = 0, div_out = 0;
        for (int x = 1; x < N[0] - 1; ++x)
            for (int y = 1; y < N[1] - 1; ++y)
                for (int z = 1; z < N[2] - 1; ++z) {
                    ++cells;
                    const int cell[3] = {x, y, z};
                    const std::size_t c = at(x, y, z);

                    const long double g[3] = {field.central(cell, 0), field.central(cell, 1),
                                              field.central(cell, 2)};
                    ASSERT_GT(g[0] * g[0] + g[1] * g[1] + g[2] * g[2], sq_min);
                    const long double a = anisotropy_ld(g, eps);
                    const long double tau_ref = tau0 * a * a;
                    const long double tau_tol =
                        (gamma_k(44, uD) + gamma_k(44, uL)) * tau_ref * host;
                    if (!(std::fabs(tau[c] - tau_ref) <= tau_tol) && tau_out++ < 5)
                        ADD_FAILURE() << "tau at (" << x << "," << y << "," << z << ") = " << tau[c]
                                      << ", reference " << static_cast<double>(tau_ref)
                                      << ", bound " << static_cast<double>(tau_tol);

                    long double div_ref = 0.0L, mag = 0.0L;
                    for (int d = 0; d < 3; ++d) {
                        FluxRef f[2];
                        for (int side = 0; side < 2; ++side) {
                            // The face between cell + (side - 1) e_d and cell + side e_d.
                            int lo[3] = {x, y, z};
                            lo[d] += side - 1;
                            int hi[3] = {lo[0], lo[1], lo[2]};
                            ++hi[d];
                            long double pf[3];
                            pf[d] = (field.value(hi) - field.value(lo)) / h[d];
                            for (int e = 0; e < 3; ++e)
                                if (e != d)
                                    pf[e] = (field.central(lo, e) + field.central(hi, e)) / 2.0L;
                            ASSERT_GT(pf[0] * pf[0] + pf[1] * pf[1] + pf[2] * pf[2], sq_min);
                            f[side] = flux_ref(pf, d, W0, eps);
                        }
                        div_ref += (f[1].F - f[0].F) / h[d];
                        mag += (f[1].M + f[0].M) / h[d];
                    }
                    const long double div_tol = (gamma_k(50, uD) + gamma_k(50, uL)) * mag * host;
                    if (!(std::fabs(div[c] - div_ref) <= div_tol) && div_out++ < 5)
                        ADD_FAILURE() << "div at (" << x << "," << y << "," << z << ") = " << div[c]
                                      << ", reference " << static_cast<double>(div_ref)
                                      << ", bound " << static_cast<double>(div_tol);
                }
        EXPECT_EQ(cells, 5 * 7 * 6);
        EXPECT_EQ(tau_out, 0);
        EXPECT_EQ(div_out, 0);
    }
}
