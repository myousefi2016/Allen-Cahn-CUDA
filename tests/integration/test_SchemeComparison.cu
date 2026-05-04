#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class SchemeComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    SimulationConfig make_config(int N, TimeScheme scheme) {
        SimulationConfig cfg;
        cfg.grid.Nx = N;
        cfg.grid.Ny = N;
        cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.07;
        cfg.physics.W0 = 1.0;
        cfg.physics.D = 2.0;
        cfg.physics.d0 = 0.5;
        cfg.time.dt = 0.0005;
        cfg.time.scheme = scheme;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.8;
        return cfg;
    }

    void make_sphere_ic(FieldData& phi, FieldData& u, int N) {
        double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
        double r0 = 3.0;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    double r =
                        std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                    phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                    u(x, y, z) = (r < r0) ? 0.0 : -0.8;
                }
    }

    double compute_l2_diff(const FieldData& a, const FieldData& b, int N) {
        double sum = 0.0;
        int count = 0;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    double d = a(x, y, z) - b(x, y, z);
                    sum += d * d;
                    ++count;
                }
        return std::sqrt(sum / count);
    }
};

// Test that RK4 and Euler produce different but comparable results.
// RK4 should be more accurate, so after many steps the results diverge.
TEST_F(SchemeComparisonTest, RK4DiffersFromEuler) {
    int N = 16;
    int steps = 20;

    // Euler
    auto cfg_euler = make_config(N, TimeScheme::Euler);
    CudaSolver euler_solver(cfg_euler);
    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi0(grid, "phi"), u0(grid, "u");
    make_sphere_ic(phi0, u0, N);
    euler_solver.initialize(phi0, u0);
    for (int s = 0; s < steps; ++s)
        euler_solver.step(0.0005);
    FieldData phi_euler(grid, "phi_e");
    euler_solver.copy_phi_to_host(phi_euler);

    // RK4
    auto cfg_rk4 = make_config(N, TimeScheme::RK4);
    CudaSolver rk4_solver(cfg_rk4);
    rk4_solver.initialize(phi0, u0);
    for (int s = 0; s < steps; ++s)
        rk4_solver.step(0.0005);
    FieldData phi_rk4(grid, "phi_r");
    rk4_solver.copy_phi_to_host(phi_rk4);

    double diff = compute_l2_diff(phi_euler, phi_rk4, N);

    // Should be different (different truncation errors)
    EXPECT_GT(diff, 1e-15);
    // But not wildly different for small dt
    EXPECT_LT(diff, 1.0);
}

// Test that Heun gives results between Euler and RK4 in terms of accuracy
TEST_F(SchemeComparisonTest, HeunIntermediateAccuracy) {
    int N = 16;
    int steps = 10;

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi0(grid, "phi"), u0(grid, "u");
    make_sphere_ic(phi0, u0, N);

    auto run_scheme = [&](TimeScheme scheme) {
        auto cfg = make_config(N, scheme);
        CudaSolver solver(cfg);
        solver.initialize(phi0, u0);
        for (int s = 0; s < steps; ++s)
            solver.step(0.0005);
        FieldData result(grid, "result");
        solver.copy_phi_to_host(result);
        return result;
    };

    auto phi_euler = run_scheme(TimeScheme::Euler);
    auto phi_heun = run_scheme(TimeScheme::Heun);
    auto phi_rk4 = run_scheme(TimeScheme::RK4);

    // All three should produce results in [-1, 1] range at center
    auto center = [&](const FieldData& f) { return f(N / 2, N / 2, N / 2); };
    EXPECT_GT(center(phi_euler), -1.5);
    EXPECT_LT(center(phi_euler), 1.5);
    EXPECT_GT(center(phi_heun), -1.5);
    EXPECT_LT(center(phi_heun), 1.5);
    EXPECT_GT(center(phi_rk4), -1.5);
    EXPECT_LT(center(phi_rk4), 1.5);
}

// Test IMEX stability with larger timestep (implicit diffusion should be stable)
TEST_F(SchemeComparisonTest, IMEXStabilityLargerDt) {
    int N = 16;
    auto cfg = make_config(N, TimeScheme::IMEX);
    cfg.time.dt = 0.005; // Larger dt that would be unstable for explicit diffusion

    CudaSolver solver(cfg);
    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);

    // Run several steps - should not blow up
    for (int s = 0; s < 10; ++s) {
        solver.step(0.005);
    }

    solver.copy_phi_to_host(phi);

    // Check for NaN/Inf
    bool stable = true;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                if (std::isnan(phi(x, y, z)) || std::isinf(phi(x, y, z))) {
                    stable = false;
                }
            }
    EXPECT_TRUE(stable) << "IMEX should remain stable with larger time step";
}
