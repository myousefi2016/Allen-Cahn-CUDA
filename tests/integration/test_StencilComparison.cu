#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace ac;
using namespace ac::cuda;

class StencilComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    SimulationConfig make_config(StencilType stencil) {
        SimulationConfig cfg;
        cfg.grid.Nx = 16;
        cfg.grid.Ny = 16;
        cfg.grid.Nz = 16;
        // Equal spacing required for 27-point stencil
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.07;
        cfg.physics.W0 = 1.0;
        cfg.physics.D = 2.0;
        cfg.physics.d0 = 0.5;
        cfg.time.dt = 0.0005;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.stencil = stencil;
        cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
        cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
        cfg.output.frequency = 10000;
        cfg.checkpoint.frequency = 0;
        return cfg;
    }

    void make_sphere_ic(FieldData& phi, FieldData& u, int N, double r0, double delta) {
        double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    double r =
                        std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                    phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                    u(x, y, z) = (r < r0) ? 0.0 : -delta * (1.0 - std::exp(-(r - r0)));
                }
    }

    /// Run a solver with the given stencil for a number of steps.
    /// Returns the final phi field on the host.
    FieldData run_with_stencil(StencilType stencil, int steps) {
        auto cfg = make_config(stencil);
        cfg.validate();

        const int N = cfg.grid.Nx;
        Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        make_sphere_ic(phi, u, N, 3.0, cfg.physics.delta);

        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        for (int s = 0; s < steps; ++s) {
            solver.step(cfg.time.dt);
        }

        FieldData phi_out(grid, "phi_out");
        solver.copy_phi_to_host(phi_out);
        return phi_out;
    }

    /// Compute the solid fraction (fraction of points with phi > 0).
    double solid_fraction(const FieldData& f) {
        double count = 0.0;
        for (std::size_t i = 0; i < f.size(); ++i) {
            if (f.data()[i] > 0.0)
                count += 1.0;
        }
        return count / static_cast<double>(f.size());
    }

    /// Compute max gradient magnitude using central differences on the interior.
    /// This measures how sharp the field transitions are.
    double max_gradient_magnitude(const FieldData& phi, double dx) {
        int Nx = phi.Nx(), Ny = phi.Ny(), Nz = phi.Nz();
        double max_grad = 0.0;
        // Only check interior points (avoid boundary effects)
        for (int x = 2; x < Nx - 2; ++x)
            for (int y = 2; y < Ny - 2; ++y)
                for (int z = 2; z < Nz - 2; ++z) {
                    double gx = (phi(x + 1, y, z) - phi(x - 1, y, z)) / (2.0 * dx);
                    double gy = (phi(x, y + 1, z) - phi(x, y - 1, z)) / (2.0 * dx);
                    double gz = (phi(x, y, z + 1) - phi(x, y, z - 1)) / (2.0 * dx);
                    double mag = std::sqrt(gx * gx + gy * gy + gz * gz);
                    max_grad = std::max(max_grad, mag);
                }
        return max_grad;
    }
};

/// Verify that both stencils produce physically valid fields:
/// phi in [-1, 1], no NaN or Inf values.
TEST_F(StencilComparisonTest, BothStencilsConverge) {
    const int steps = 20;

    auto phi_7pt = run_with_stencil(StencilType::Standard7Point, steps);
    auto phi_27pt = run_with_stencil(StencilType::Isotropic27Point, steps);

    // Check 7-point stencil result
    for (std::size_t i = 0; i < phi_7pt.size(); ++i) {
        double v = phi_7pt.data()[i];
        ASSERT_FALSE(std::isnan(v)) << "7pt: NaN at index " << i;
        ASSERT_FALSE(std::isinf(v)) << "7pt: Inf at index " << i;
        EXPECT_GE(v, -1.5) << "7pt: phi below -1.5 at index " << i;
        EXPECT_LE(v, 1.5) << "7pt: phi above 1.5 at index " << i;
    }

    // Check 27-point stencil result
    for (std::size_t i = 0; i < phi_27pt.size(); ++i) {
        double v = phi_27pt.data()[i];
        ASSERT_FALSE(std::isnan(v)) << "27pt: NaN at index " << i;
        ASSERT_FALSE(std::isinf(v)) << "27pt: Inf at index " << i;
        EXPECT_GE(v, -1.5) << "27pt: phi below -1.5 at index " << i;
        EXPECT_LE(v, 1.5) << "27pt: phi above 1.5 at index " << i;
    }

    // Both should have a meaningful phase distribution (not all one phase)
    double sf_7pt = solid_fraction(phi_7pt);
    double sf_27pt = solid_fraction(phi_27pt);
    EXPECT_GT(sf_7pt, 0.0) << "7pt: should have some solid phase";
    EXPECT_LT(sf_7pt, 1.0) << "7pt: should have some liquid phase";
    EXPECT_GT(sf_27pt, 0.0) << "27pt: should have some solid phase";
    EXPECT_LT(sf_27pt, 1.0) << "27pt: should have some liquid phase";
}

/// The 27-point isotropic stencil should produce smoother (less grid-anisotropic)
/// results compared to the 7-point standard stencil. We verify this by comparing
/// the maximum gradient magnitude: the 27-point result should have lower or equal
/// max gradient since it better approximates the continuous Laplacian.
TEST_F(StencilComparisonTest, IsotropicStencilSmoother) {
    const int steps = 20;
    const double dx = 0.4;

    auto phi_7pt = run_with_stencil(StencilType::Standard7Point, steps);
    auto phi_27pt = run_with_stencil(StencilType::Isotropic27Point, steps);

    double max_grad_7pt = max_gradient_magnitude(phi_7pt, dx);
    double max_grad_27pt = max_gradient_magnitude(phi_27pt, dx);

    // Both should have non-trivial gradients (there is an interface)
    EXPECT_GT(max_grad_7pt, 0.1) << "7pt max gradient should be non-trivial (interface exists)";
    EXPECT_GT(max_grad_27pt, 0.1) << "27pt max gradient should be non-trivial (interface exists)";

    // The 27-point stencil, being more isotropic, should produce a smoother
    // interface (lower or equal max gradient). Allow a small tolerance in case
    // they are very close.
    EXPECT_LE(max_grad_27pt, max_grad_7pt * 1.05)
        << "27pt stencil should produce smoother (or comparably smooth) interface. "
        << "7pt max grad = " << max_grad_7pt << ", 27pt max grad = " << max_grad_27pt;
}

/// Both stencils should preserve consistent physics: given the same initial
/// conditions and parameters, the solid fraction should evolve in the same
/// direction (both growing or both shrinking).
TEST_F(StencilComparisonTest, BothPreservePhysics) {
    // Need enough steps for the interface to advect across at least one cell
    // (sharp +/-1 IC + small dt means fewer steps just diffuses inside cells).
    const int steps = 400;

    // Get initial solid fraction (same for both since same IC)
    auto cfg = make_config(StencilType::Standard7Point);
    cfg.validate();
    Grid grid = cfg.make_grid();
    FieldData phi0(grid, "phi0"), u0(grid, "u0");
    make_sphere_ic(phi0, u0, cfg.grid.Nx, 3.0, cfg.physics.delta);
    double initial_sf = solid_fraction(phi0);

    auto phi_7pt = run_with_stencil(StencilType::Standard7Point, steps);
    auto phi_27pt = run_with_stencil(StencilType::Isotropic27Point, steps);

    double sf_7pt = solid_fraction(phi_7pt);
    double sf_27pt = solid_fraction(phi_27pt);

    // The solid fraction should have changed from the initial state for both
    EXPECT_NE(sf_7pt, initial_sf) << "7pt: solid fraction should evolve from initial value";
    EXPECT_NE(sf_27pt, initial_sf) << "27pt: solid fraction should evolve from initial value";

    // Both stencils should show the same qualitative behavior
    // (both growing or both shrinking relative to initial)
    double delta_7pt = sf_7pt - initial_sf;
    double delta_27pt = sf_27pt - initial_sf;

    // Same sign means same direction of evolution
    // Use a product check: positive means same sign
    EXPECT_GT(delta_7pt * delta_27pt, 0.0)
        << "Both stencils should show same direction of solid fraction change. "
        << "7pt delta = " << delta_7pt << ", 27pt delta = " << delta_27pt;

    // The magnitude of the change should be in the same ballpark
    // (within an order of magnitude)
    double ratio = std::abs(delta_7pt) / std::abs(delta_27pt);
    EXPECT_GT(ratio, 0.1) << "Solid fraction changes should be comparable in magnitude";
    EXPECT_LT(ratio, 10.0) << "Solid fraction changes should be comparable in magnitude";
}
