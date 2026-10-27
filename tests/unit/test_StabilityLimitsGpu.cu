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

/// The checkerboard over the axes in the bitmask `axes` (bit d: axis d).
double checkerboard(int x, int y, int z, int axes = 7) {
    const int sum = ((axes & 1) ? x : 0) + ((axes & 2) ? y : 0) + ((axes & 4) ? z : 0);
    return sum % 2 == 0 ? 1.0 : -1.0;
}

/// Perturbs u (thermal) or phi (phase field) of the liquid by eta = 1e-6
/// times the checkerboard over `axes` and returns the factor per step over
/// 40 steps, measured at one cell. `background` adds to phi the periodic ramp
/// 1e-6 cos(2 pi (x + y + z - 3) / 16), whose central differences point
/// along -<111> at the measured cell (5, 5, 5), with phi still -1 to 1e-6.
double amplification(const SimulationConfig& cfg, double dt, bool thermal, int axes = 7,
                     bool background = false) {
    const Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    const double eta = background ? 1e-9 : 1e-6;
    const double pi = 3.14159265358979323846;
    auto base = [&](int x, int y, int z) {
        return background ? 1e-6 * std::cos(2 * pi * (x + y + z - 3) / 16.0) : 0.0;
    };
    for (int x = 0; x < cfg.grid.Nx; ++x)
        for (int y = 0; y < cfg.grid.Ny; ++y)
            for (int z = 0; z < cfg.grid.Nz; ++z) {
                phi(x, y, z) =
                    -1.0 + base(x, y, z) + (thermal ? 0.0 : eta * checkerboard(x, y, z, axes));
                u(x, y, z) = thermal ? -0.8 + eta * checkerboard(x, y, z, axes) : 0.0;
            }
    cuda::CudaSolver solver(cfg);
    solver.initialize(phi, u);
    // The background alone evolves too: subtract its own step-by-step
    // evolution, run without the checkerboard.
    FieldData phi_bg(grid, "phi_bg");
    if (background && !thermal) {
        FieldData p0(grid, "p0"), u0(grid, "u0");
        for (int x = 0; x < cfg.grid.Nx; ++x)
            for (int y = 0; y < cfg.grid.Ny; ++y)
                for (int z = 0; z < cfg.grid.Nz; ++z) {
                    p0(x, y, z) = -1.0 + base(x, y, z);
                    u0(x, y, z) = 0.0;
                }
        cuda::CudaSolver bg(cfg);
        bg.initialize(p0, u0);
        for (int s = 0; s < 40; ++s)
            bg.step(dt);
        bg.copy_phi_to_host(phi_bg);
    }
    const int steps = 40;
    for (int s = 0; s < steps; ++s)
        solver.step(dt);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);
    double a = 0.0;
    if (thermal)
        a = u(5, 5, 5) + 0.8;
    else if (background)
        a = phi(5, 5, 5) - phi_bg(5, 5, 5);
    else
        a = phi(5, 5, 5) + 1.0;
    return std::pow(std::fabs(a / (eta * checkerboard(5, 5, 5, axes))), 1.0 / steps);
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

// The uniform liquid's checkerboards over a subset S of the axes are exact
// modes of the phase-field update (IMEX: u implicit, phi explicit as in
// Euler): at 0.99 and 1.01 times their limit 2 tau_cb / (Lam_S - beta L_S)
// they decay and grow. Without the Willmore term the full checkerboard binds;
// with beta = 0.0149 (eps 0.07, the physics of default.json) the full
// checkerboard is damped at every dt, but the 1-D and 2-D ones are not.
TEST_F(StabilityLimitsGpuTest, UniformLiquidCheckerboardLimitsAreSharp) {
    struct Case {
        double eps, beta;
        int axes;
    };
    for (const Case c :
         {Case{0.0, 0.0, 7}, Case{0.05, 0.0, 7}, Case{0.07, 0.0149, 1}, Case{0.07, 0.0149, 3}}) {
        auto cfg = config(TimeScheme::IMEX, StencilType::Standard7Point);
        cfg.physics.epsilon = c.eps;
        cfg.physics.willmore_beta = c.beta;
        const int n_axes = (c.axes & 1) + ((c.axes >> 1) & 1) + ((c.axes >> 2) & 1);
        const double mu = n_axes * 25.0; // 4 / h^2 per axis, h = 0.4
        const double tau_cb = cfg.physics.tau0() * (1 - 0.6 * c.eps) * (1 - 0.6 * c.eps);
        const double lam = (1 + c.eps) * (1 + c.eps) * mu + 2, l_s = (mu + 2) * (mu + 2);
        const double limit = 2 * tau_cb / (lam - c.beta * l_s);
        const double below = amplification(cfg, 0.99 * limit, false, c.axes);
        const double above = amplification(cfg, 1.01 * limit, false, c.axes);
        std::printf("uniform liquid eps %.2f beta %.4f axes %d: limit %.6f, |g| %.9f at 0.99x, "
                    "%.9f at 1.01x\n",
                    c.eps, c.beta, c.axes, limit, below, above);
        EXPECT_LT(below, 1.0);
        EXPECT_GT(above, 1.0);
        // validate()'s limit is at most this one (to rounding).
        EXPECT_LE(phase_field_dt_limit(cfg), limit * (1 + 1e-13));
    }
    auto cfg = config(TimeScheme::IMEX, StencilType::Standard7Point);
    cfg.physics.epsilon = 0.07;
    cfg.physics.willmore_beta = 0.0149;
    // dt = 0.06, which validate() accepted when only the full checkerboard
    // counted: that one decays, the 1-D one grows.
    const double full = amplification(cfg, 0.06, false, 7),
                 one_d = amplification(cfg, 0.06, false, 1);
    std::printf("eps 0.07 beta 0.0149 at dt 0.06: |g| full %.6f, 1-D %.6f\n", full, one_d);
    EXPECT_LT(full, 1.0);
    EXPECT_GT(one_d, 1.0);
}

// A liquid with a small smooth gradient along <111> (phi = -1 to 1e-6):
// tau = tau0 A(<111>)^2 and the face fluxes respond with H_dd(<111>), so the
// checkerboard grows faster than in the uniform liquid. phase_field_dt_limit
// is this case's frozen-coefficient limit on a cubic grid without the
// Willmore term (the host test checks it against the closed form): the
// perturbation decays at 0.99 and grows at 1.01 times it, and grows at 0.99
// times the uniform liquid's limit, which validate() accepted before.
TEST_F(StabilityLimitsGpuTest, GradientInTheLiquidLowersThePhaseFieldLimit) {
    auto cfg = config(TimeScheme::IMEX, StencilType::Standard7Point);
    const double limit = phase_field_dt_limit(cfg);
    const double tau_cb = cfg.physics.tau0() * 0.97 * 0.97;
    const double uniform = 2 * tau_cb / (1.05 * 1.05 * 75 + 2);
    const double below = amplification(cfg, 0.99 * limit, false, 7, true);
    const double above = amplification(cfg, 1.01 * limit, false, 7, true);
    const double at_uniform = amplification(cfg, 0.99 * uniform, false, 7, true);
    std::printf("<111> gradient: limit %.6f (uniform liquid %.6f), |g| %.6f at 0.99x, %.6f at "
                "1.01x, %.6f at 0.99x the uniform limit\n",
                limit, uniform, below, above, at_uniform);
    EXPECT_LT(below, 1.0);
    EXPECT_GT(above, 1.0);
    EXPECT_GT(at_uniform, 1.0);
}
