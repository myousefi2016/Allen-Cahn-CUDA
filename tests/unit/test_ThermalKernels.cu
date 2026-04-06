#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class ThermalTest : public ::testing::Test {
protected:
    static constexpr int N = 8;
    static constexpr double dx = 1.0;
    static constexpr double dt = 0.01;
    static constexpr double D = 1.0;

    void SetUp() override {
        Logger::init(spdlog::level::off);
        total_ = static_cast<std::size_t>(N) * N * N;
    }

    KernelParams make_params() {
        KernelParams p{};
        p.Nx = N; p.Ny = N; p.Nz = N;
        p.dx = dx; p.dy = dx; p.dz = dx;
        p.dt = dt;
        p.D = D;
        p.stencil_type = 0;  // 7-point
        p.epsilon = 0.0;
        p.W0 = 1.0;
        p.tau0 = 1.0;
        p.lambda = 1.0;
        p.delta = 0.0;
        return p;
    }

    int idx(int x, int y, int z) const {
        return x * N * N + y * N + z;
    }

    std::size_t total_ = 0;
};

// Uniform phi_new == phi_old, uniform u -> no change
TEST_F(ThermalTest, UniformFieldNoChange)
{
    auto p = make_params();

    std::vector<double> u_old_h(total_, 5.0);
    std::vector<double> u_new_h(total_, 0.0);
    std::vector<double> phi_old_h(total_, 1.0);
    std::vector<double> phi_new_h(total_, 1.0);  // same as phi_old

    DeviceField<double> d_u_old(total_), d_u_new(total_);
    DeviceField<double> d_phi_old(total_), d_phi_new(total_);

    d_u_old.copy_from_host(u_old_h.data());
    d_u_new.copy_from_host(u_new_h.data());
    d_phi_old.copy_from_host(phi_old_h.data());
    d_phi_new.copy_from_host(phi_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_thermal_equation(d_u_old.data(), d_u_new.data(),
                            d_phi_new.data(), d_phi_old.data(),
                            p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_u_new.copy_to_host(u_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Interior points: u_new = u_old + 0 + dt*D*0 = 5.0
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(u_new_h[idx(x, y, z)], 5.0, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// phi_new == phi_old (no latent heat), u = x^2 -> Laplacian = 2/dx^2
TEST_F(ThermalTest, DiffusionOnly)
{
    auto p = make_params();

    std::vector<double> u_old_h(total_);
    std::vector<double> u_new_h(total_, 0.0);
    std::vector<double> phi_old_h(total_, 0.0);
    std::vector<double> phi_new_h(total_, 0.0);

    // u = (x*dx)^2, with dx=1.0 -> u = x^2
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                u_old_h[idx(x, y, z)] = static_cast<double>(x * x);

    DeviceField<double> d_u_old(total_), d_u_new(total_);
    DeviceField<double> d_phi_old(total_), d_phi_new(total_);

    d_u_old.copy_from_host(u_old_h.data());
    d_u_new.copy_from_host(u_new_h.data());
    d_phi_old.copy_from_host(phi_old_h.data());
    d_phi_new.copy_from_host(phi_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_thermal_equation(d_u_old.data(), d_u_new.data(),
                            d_phi_new.data(), d_phi_old.data(),
                            p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_u_new.copy_to_host(u_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Laplacian of x^2 (7pt) = d^2/dx^2(x^2) + 0 + 0 = 2.0 / (dx^2) = 2.0
    // u_new = u_old + 0 + dt * D * 2.0 = x^2 + 0.01 * 1.0 * 2.0 = x^2 + 0.02
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double expected = static_cast<double>(x * x) + dt * D * 2.0;
                EXPECT_NEAR(u_new_h[idx(x, y, z)], expected, 1e-10)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// Uniform u (Laplacian=0), phi_new != phi_old -> verify latent heat
TEST_F(ThermalTest, LatentHeatOnly)
{
    auto p = make_params();

    std::vector<double> u_old_h(total_, 3.0);
    std::vector<double> u_new_h(total_, 0.0);
    std::vector<double> phi_old_h(total_, -1.0);
    std::vector<double> phi_new_h(total_, 0.5);

    DeviceField<double> d_u_old(total_), d_u_new(total_);
    DeviceField<double> d_phi_old(total_), d_phi_new(total_);

    d_u_old.copy_from_host(u_old_h.data());
    d_u_new.copy_from_host(u_new_h.data());
    d_phi_old.copy_from_host(phi_old_h.data());
    d_phi_new.copy_from_host(phi_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_thermal_equation(d_u_old.data(), d_u_new.data(),
                            d_phi_new.data(), d_phi_old.data(),
                            p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_u_new.copy_to_host(u_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // latent_heat = 0.5 * (0.5 - (-1.0)) = 0.5 * 1.5 = 0.75
    // u_new = 3.0 + 0.75 + 0 = 3.75
    double expected = 3.0 + 0.5 * (0.5 - (-1.0));
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_NEAR(u_new_h[idx(x, y, z)], expected, 1e-12)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// Both diffusion and latent heat present
TEST_F(ThermalTest, CombinedUpdate)
{
    auto p = make_params();

    std::vector<double> u_old_h(total_);
    std::vector<double> u_new_h(total_, 0.0);
    std::vector<double> phi_old_h(total_, 0.0);
    std::vector<double> phi_new_h(total_, 0.4);

    // u = x^2 => Laplacian = 2.0
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                u_old_h[idx(x, y, z)] = static_cast<double>(x * x);

    DeviceField<double> d_u_old(total_), d_u_new(total_);
    DeviceField<double> d_phi_old(total_), d_phi_new(total_);

    d_u_old.copy_from_host(u_old_h.data());
    d_u_new.copy_from_host(u_new_h.data());
    d_phi_old.copy_from_host(phi_old_h.data());
    d_phi_new.copy_from_host(phi_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_thermal_equation(d_u_old.data(), d_u_new.data(),
                            d_phi_new.data(), d_phi_old.data(),
                            p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_u_new.copy_to_host(u_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // u_new = x^2 + 0.5*(0.4 - 0.0) + dt*D*2.0
    //       = x^2 + 0.2 + 0.02
    double latent = 0.5 * (0.4 - 0.0);
    double diffusion = dt * D * 2.0;
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double expected = static_cast<double>(x * x) + latent + diffusion;
                EXPECT_NEAR(u_new_h[idx(x, y, z)], expected, 1e-10)
                    << "at (" << x << "," << y << "," << z << ")";
            }
}

// Verify boundary points remain unchanged (kernel does not write them)
TEST_F(ThermalTest, BoundaryUntouched)
{
    auto p = make_params();

    // Initialize u_new to a sentinel value
    double sentinel = -999.0;
    std::vector<double> u_old_h(total_, 1.0);
    std::vector<double> u_new_h(total_, sentinel);
    std::vector<double> phi_old_h(total_, 0.0);
    std::vector<double> phi_new_h(total_, 1.0);

    DeviceField<double> d_u_old(total_), d_u_new(total_);
    DeviceField<double> d_phi_old(total_), d_phi_new(total_);

    d_u_old.copy_from_host(u_old_h.data());
    d_u_new.copy_from_host(u_new_h.data());
    d_phi_old.copy_from_host(phi_old_h.data());
    d_phi_new.copy_from_host(phi_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_thermal_equation(d_u_old.data(), d_u_new.data(),
                            d_phi_new.data(), d_phi_old.data(),
                            p, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_u_new.copy_to_host(u_new_h.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Boundary points (where x=0, x=N-1, y=0, y=N-1, z=0, z=N-1)
    // should remain at the sentinel value
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                if (x == 0 || x == N - 1 ||
                    y == 0 || y == N - 1 ||
                    z == 0 || z == N - 1) {
                    EXPECT_DOUBLE_EQ(u_new_h[idx(x, y, z)], sentinel)
                        << "Boundary modified at (" << x << "," << y << "," << z << ")";
                }
            }

    // Also verify at least some interior points were updated (not sentinel)
    bool any_interior_updated = false;
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                if (u_new_h[idx(x, y, z)] != sentinel) {
                    any_interior_updated = true;
                }
            }
    EXPECT_TRUE(any_interior_updated);
}
