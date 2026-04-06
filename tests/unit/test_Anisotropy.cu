#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class AnisotropyTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
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

// Test: An with zero gradient should return 1 - 5/3*epsilon
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

    double expected = 1.0 - (5.0 / 3.0) * epsilon;
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
