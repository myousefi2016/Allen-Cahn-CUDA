// On the GPU, the solver's explicit updates have exactly the stability
// limits of core/StabilityLimits.hpp (docs/THEORY.md 10.1): a checkerboard
// perturbation decays at 0.99 times the limit and grows at 1.01 times it.
#include "common/Gpu.hpp"
#include "core/StabilityLimits.hpp"
#include "cuda/CudaSolver.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>

using namespace ac;

namespace {

/// The default physics (eps 0.05, D = 2, d0 = 0.5, W0 = 1), h = 0.4,
/// periodic faces and 16 interior cells per axis: the checkerboard is an
/// exact mode.
SimulationConfig config(TimeScheme scheme, StencilType stencil) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 18;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
    cfg.physics.epsilon = 0.05;
    cfg.physics.D = 2.0;
    cfg.time.scheme = scheme;
    cfg.stencil = stencil;
    BoundaryConfig periodic;
    periodic.type = BCType::Periodic;
    cfg.boundary.phi_bc = periodic;
    cfg.boundary.u_bc = periodic;
    return cfg;
}

class StabilityLimitsGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

double checkerboard(int x, int y, int z) {
    return (x + y + z) % 2 == 0 ? 1.0 : -1.0;
}

/// Perturbs u (thermal) or phi (phase field) of the liquid by eta = 1e-6
/// times the checkerboard and returns the factor per step over 40 steps,
/// measured at one cell.
double amplification(const SimulationConfig& cfg, double dt, bool thermal) {
    const Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    const double eta = 1e-6;
    for (int x = 0; x < cfg.grid.Nx; ++x)
        for (int y = 0; y < cfg.grid.Ny; ++y)
            for (int z = 0; z < cfg.grid.Nz; ++z) {
                phi(x, y, z) = -1.0 + (thermal ? 0.0 : eta * checkerboard(x, y, z));
                u(x, y, z) = thermal ? -0.8 + eta * checkerboard(x, y, z) : 0.0;
            }
    cuda::CudaSolver solver(cfg);
    solver.initialize(phi, u);
    const int steps = 40;
    for (int s = 0; s < steps; ++s)
        solver.step(dt);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);
    const double a = thermal ? u(5, 5, 5) + 0.8 : phi(5, 5, 5) + 1.0;
    return std::pow(std::fabs(a / (eta * checkerboard(5, 5, 5))), 1.0 / steps);
}

} // namespace

TEST_F(StabilityLimitsGpuTest, ThermalLimitIsSharpForEverySchemeAndStencil) {
    for (TimeScheme scheme : {TimeScheme::Euler, TimeScheme::Heun, TimeScheme::RK4}) {
        for (StencilType stencil : {StencilType::Standard7Point, StencilType::Isotropic27Point}) {
            auto cfg = config(scheme, stencil);
            const double limit = thermal_dt_limit(cfg);
            const double below = amplification(cfg, 0.99 * limit, true);
            const double above = amplification(cfg, 1.01 * limit, true);
            std::printf("thermal scheme %d stencil %d: limit %.6f, |g| %.9f at 0.99x, %.9f at "
                        "1.01x\n",
                        static_cast<int>(scheme), static_cast<int>(stencil), limit, below, above);
            EXPECT_LT(below, 1.0);
            EXPECT_GT(above, 1.0);
        }
    }
}

// The phase field (IMEX: u implicit, phi explicit as in Euler) without and
// with a Willmore term too weak to damp the mode; a Willmore term at least
// lam / Lcb damps it at any dt (here 10 times the limit without it).
TEST_F(StabilityLimitsGpuTest, PhaseFieldLimitIsSharp) {
    for (double eps : {0.0, 0.05}) {
        for (double beta : {0.0, 0.0005}) {
            auto cfg = config(TimeScheme::IMEX, StencilType::Standard7Point);
            cfg.physics.epsilon = eps;
            cfg.physics.willmore_beta = beta;
            const double limit = phase_field_dt_limit(cfg);
            const double below = amplification(cfg, 0.99 * limit, false);
            const double above = amplification(cfg, 1.01 * limit, false);
            std::printf("phase field eps %.2f beta %.4f: limit %.6f, |g| %.9f at 0.99x, %.9f at "
                        "1.01x\n",
                        eps, beta, limit, below, above);
            EXPECT_LT(below, 1.0);
            EXPECT_GT(above, 1.0);
        }
    }
    auto cfg = config(TimeScheme::IMEX, StencilType::Standard7Point);
    const double unregularised = phase_field_dt_limit(cfg);
    cfg.physics.willmore_beta = 0.0149;
    ASSERT_EQ(phase_field_dt_limit(cfg), std::numeric_limits<double>::infinity());
    EXPECT_LT(amplification(cfg, 10.0 * unregularised, false), 1.0);
}
