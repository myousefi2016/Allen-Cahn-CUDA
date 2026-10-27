#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

namespace {

constexpr double kPi = 3.14159265358979323846;

SimulationConfig periodic_config(int N, double h, double beta, double dt, double W0 = 1.0,
                                 double dy = 0.0, double dz = 0.0) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = N;
    cfg.grid.dx = h;
    cfg.grid.dy = dy > 0.0 ? dy : h;
    cfg.grid.dz = dz > 0.0 ? dz : h;
    cfg.physics.epsilon = 0.0;
    cfg.physics.W0 = W0;
    cfg.physics.willmore_beta = beta;
    cfg.time.dt = dt;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.stencil = StencilType::Standard7Point;
    BoundaryConfig periodic;
    periodic.type = BCType::Periodic;
    cfg.boundary.phi_bc = periodic;
    cfg.boundary.u_bc = periodic;
    return cfg;
}

class WillmoreTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

} // namespace

// Linearised about the liquid bulk phi = -1 with u = 0 and eps = 0, one
// Willmore step multiplies a Fourier mode of amplitude a by
//   g = 1 + dt (-W0^2 s - 2 - beta L) / (tau0 + dt beta L),
//   s = sum_d (4/h_d^2) sin^2(k_d h_d / 2)  (7-point symbol),
//   L = W0^4 s^2 + 4 W0^2 s + 4              (symbol of the implicit operator,
//                                              which equals M's linearisation here).
// This checks omega, M, the implicit operator, the CG solve and the update
// against an independent formula; the near-Nyquist mode also shows the step
// is stable where explicit Euler is not (explicit factor < -100). W0 != 1 and
// a different spacing on each axis make every power of W0 and every axis's
// 1/h_d^2 visible (tau0 = a1 a2 W0^3 / (d0 D) is taken from the config).
TEST_F(WillmoreTest, LinearisedAmplificationMatchesAnalyticSymbol) {
    const int N = 18, P = N - 2; // periodic ring of P interior cells
    const double h[3] = {0.5, 0.42, 0.61}, W0 = 1.3, beta = 1.5, amp = 1e-4;
    struct Mode {
        int m[3];
        double dt;
    };
    const Mode modes[] = {{{1, 0, 0}, 0.01}, {{3, 2, 1}, 0.05}, {{8, 8, 7}, 0.05}};

    for (const auto& mode : modes) {
        SCOPED_TRACE("mode " + std::to_string(mode.m[0]) + "," + std::to_string(mode.m[1]) + "," +
                     std::to_string(mode.m[2]));
        auto cfg = periodic_config(N, h[0], beta, mode.dt, W0, h[1], h[2]);
        Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        auto shape = [&](int x, int y, int z) {
            return std::cos(2.0 * kPi *
                            (mode.m[0] * (x - 1) + mode.m[1] * (y - 1) + mode.m[2] * (z - 1)) / P);
        };
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    phi(x, y, z) = -1.0 + amp * shape(x, y, z);
                    u(x, y, z) = 0.0;
                }
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        solver.step(mode.dt);
        solver.copy_phi_to_host(phi);

        double num = 0.0, den = 0.0;
        for (int x = 1; x < N - 1; ++x)
            for (int y = 1; y < N - 1; ++y)
                for (int z = 1; z < N - 1; ++z) {
                    num += (phi(x, y, z) + 1.0) * shape(x, y, z);
                    den += shape(x, y, z) * shape(x, y, z);
                }
        const double measured = num / den / amp;

        double s = 0.0;
        for (int d = 0; d < 3; ++d) {
            const double sn = std::sin(kPi * mode.m[d] / P);
            s += 4.0 / (h[d] * h[d]) * sn * sn;
        }
        const double W2 = W0 * W0, L = W2 * W2 * s * s + 4.0 * W2 * s + 4.0;
        const double tau0 = cfg.physics.tau0();
        const double g = 1.0 + mode.dt * (-W2 * s - 2.0 - beta * L) / (tau0 + mode.dt * beta * L);
        const double g_explicit = 1.0 + mode.dt * (-W2 * s - 2.0 - beta * L) / tau0;
        std::printf("mode (%d,%d,%d) dt=%.3f: measured %.10f  analytic %.10f  (explicit %.3f)  "
                    "CG iterations %d\n",
                    mode.m[0], mode.m[1], mode.m[2], mode.dt, measured, g, g_explicit,
                    solver.last_willmore_stats().iterations);
        EXPECT_NEAR(measured, g, 1e-7 * (1.0 + std::fabs(g)));
        EXPECT_LT(std::fabs(g), 1.0);
    }
}

// As beta -> 0 the linearly implicit step reduces to the explicit Euler step:
// with beta = 1e-13 (every beta term <= 1e-11 here) it must match the
// unregularised solver to rounding, for a strongly anisotropic curved seed.
TEST_F(WillmoreTest, VanishingBetaReducesToExplicitEuler) {
    const int N = 16;
    const double h = 0.4, dt = 0.002;
    auto make = [&](double beta) {
        SimulationConfig cfg;
        cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = h;
        cfg.physics.epsilon = 0.12;
        cfg.physics.willmore_beta = beta;
        cfg.time.dt = dt;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.6;
        return cfg;
    };
    Grid grid(Dim3{N, N, N}, Spacing{h, h, h});
    FieldData phi0(grid, "phi"), u0(grid, "u");
    const double c = 0.37 * (N - 1);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                const double r =
                    std::sqrt((x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c)) * h;
                phi0(x, y, z) = -std::tanh((r - 1.8) / std::sqrt(2.0));
                u0(x, y, z) = -0.6 + 0.05 * std::sin(0.7 * x) * std::cos(0.5 * y + z);
            }

    FieldData explicit_phi(grid, "e"), implicit_phi(grid, "i");
    {
        CudaSolver s(make(0.0));
        s.initialize(phi0, u0);
        s.step(dt);
        s.copy_phi_to_host(explicit_phi);
    }
    CudaSolver s(make(1e-13));
    s.initialize(phi0, u0);
    s.step(dt);
    s.copy_phi_to_host(implicit_phi);

    double max_diff = 0.0, max_change = 0.0;
    for (std::size_t i = 0; i < explicit_phi.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(explicit_phi.data()[i] - implicit_phi.data()[i]));
        max_change = std::max(max_change, std::fabs(explicit_phi.data()[i] - phi0.data()[i]));
    }
    EXPECT_GT(max_change, 1e-4) << "the step must actually move phi";
    EXPECT_LT(max_diff, 1e-12);
}

// The Willmore right-hand side takes omega's boundary cells from the
// homogeneous form of phi's boundary conditions: the discrete energy
// (beta/2)|omega_h|^2, omega_h = W0^2 (lap_0 phi + c) + h(phi) with lap_0 the
// Laplacian under the homogeneous ghost rule and c the walls' constant part,
// has the gradient beta (W0^2 lap_0 omega + h'(phi) omega). The two rules
// differ only on inhomogeneous walls. phi = -1, u = 0 is an exact equilibrium
// of every term (h(-1) = 0, (1 - phi^2)^2 = 0, omega = 0) inside walls whose
// rule holds phi = -1: Dirichlet -1, and Robin alpha phi + beta dphi/dn =
// gamma with gamma = -alpha (ghost (gamma - beta/h)/(alpha + beta/h) = -1
// exactly). So every step must leave phi exactly -1. The inhomogeneous rule
// on omega would put omega = -1 (Dirichlet) or gamma/(alpha + beta/h)
// (Robin) into the ghost cells and a source next to every wall.
TEST_F(WillmoreTest, UniformLiquidInsideInhomogeneousWallsStaysExact) {
    auto bc = [](BCType type, double value, double alpha, double beta, double gamma) {
        BoundaryConfig b;
        b.type = type;
        b.value = value;
        b.alpha = alpha;
        b.beta = beta;
        b.gamma = gamma;
        return b;
    };
    const BoundaryConfig dirichlet = bc(BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0),
                         robin_a = bc(BCType::Robin, 0.0, 1.0, 0.5, -1.0),
                         robin_b = bc(BCType::Robin, 0.0, 2.0, 0.3, -2.0);
    for (TimeScheme scheme : {TimeScheme::Euler, TimeScheme::IMEX}) {
        SCOPED_TRACE(scheme == TimeScheme::Euler ? "euler" : "imex");
        SimulationConfig cfg;
        cfg.grid.Nx = 8;
        cfg.grid.Ny = 7;
        cfg.grid.Nz = 9;
        cfg.grid.dx = 0.4;
        cfg.grid.dy = 0.5;
        cfg.grid.dz = 0.3;
        cfg.physics.epsilon = 0.12;
        cfg.physics.W0 = 1.3;
        cfg.physics.willmore_beta = 1.0;
        cfg.time.dt = 0.005;
        cfg.time.scheme = scheme;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.per_face = true;
        cfg.boundary.phi_faces.faces = {dirichlet, dirichlet, robin_a, robin_a, dirichlet, robin_b};
        BoundaryConfig zero_flux;
        zero_flux.type = BCType::Neumann;
        cfg.boundary.u_faces = PerFaceBoundary::uniform(zero_flux);
        cfg.validate();
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        for (std::size_t i = 0; i < phi.size(); ++i) {
            phi.data()[i] = -1.0;
            u.data()[i] = 0.0;
        }
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        for (int s = 0; s < 3; ++s)
            solver.step(cfg.time.dt);
        solver.copy_phi_to_host(phi);
        int changed = 0;
        for (std::size_t i = 0; i < phi.size(); ++i)
            changed += phi.data()[i] != -1.0;
        EXPECT_EQ(changed, 0);
    }
}

// The fourth-order term is only integrated by the Euler / IMEX phase-field
// update; the explicit multi-stage schemes must refuse it.
TEST_F(WillmoreTest, RequiresEulerOrImex) {
    for (TimeScheme s : {TimeScheme::Heun, TimeScheme::RK4}) {
        auto cfg = periodic_config(8, 0.5, 1.0, 0.01);
        cfg.time.scheme = s;
        EXPECT_THROW(CudaSolver solver(cfg), std::invalid_argument);
        EXPECT_THROW(cfg.validate(), std::invalid_argument);
    }
    for (TimeScheme s : {TimeScheme::Euler, TimeScheme::IMEX}) {
        auto cfg = periodic_config(8, 0.5, 1.0, 0.01);
        cfg.time.scheme = s;
        EXPECT_NO_THROW(CudaSolver solver(cfg));
    }
}

// eps > 1/15 is only accepted with the regularisation.
TEST(WillmoreConfig, StrongAnisotropyRequiresRegularisation) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 10;
    // Below the explicit phase-field limit at eps = 1/15 (0.0096, THEORY 10.1),
    // so only the regularisation rule decides.
    cfg.time.dt = 0.005;
    cfg.physics.epsilon = 1.0 / 15.0;
    EXPECT_NO_THROW(cfg.validate());
    cfg.physics.epsilon = 0.0667;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.physics.willmore_beta = 1.0;
    EXPECT_NO_THROW(cfg.validate());
    cfg.physics.willmore_beta = -1.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// The linearly implicit step is first-order consistent in time: on a fixed
// grid, a strongly anisotropic regularised seed advanced to t = 0.4 has
// successive max-norm differences shrinking by ~2 per halving of dt. The
// order approaches 1 from below as dt shrinks: measured on an RTX 4090 from
// dt = 0.02 it is 0.68, 0.82, 0.91, 0.95, 0.97, so the test uses the
// asymptotic range dt = 0.0025 ... 0.0003125.
TEST_F(WillmoreTest, FirstOrderConvergenceInTime) {
    const int N = 24;
    const double h = 0.4, T = 0.4;
    auto run = [&](double dt) {
        SimulationConfig cfg;
        cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = h;
        cfg.physics.epsilon = 0.12;
        cfg.physics.willmore_beta = 1.0;
        cfg.physics.delta = 0.55;
        cfg.time.dt = dt;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.55;
        Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        const double c = 0.5 * (N - 1);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    const double r =
                        std::sqrt((x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c)) * h;
                    phi(x, y, z) = -std::tanh((r - 2.5) / std::sqrt(2.0));
                    u(x, y, z) = -0.55;
                }
        CudaSolver s(cfg);
        s.initialize(phi, u);
        const int steps = static_cast<int>(std::lround(T / dt));
        for (int k = 0; k < steps; ++k)
            s.step(dt);
        s.copy_phi_to_host(phi);
        return phi;
    };
    const double dts[] = {0.0025, 0.00125, 0.000625, 0.0003125};
    std::vector<FieldData> sol;
    for (double dt : dts)
        sol.push_back(run(dt));
    auto maxdiff = [](const FieldData& a, const FieldData& b) {
        double m = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            m = std::max(m, std::fabs(a.data()[i] - b.data()[i]));
        return m;
    };
    const double d1 = maxdiff(sol[0], sol[1]), d2 = maxdiff(sol[1], sol[2]),
                 d3 = maxdiff(sol[2], sol[3]);
    std::printf("time refinement: |dt - dt/2| %.3e  |dt/2 - dt/4| %.3e  |dt/4 - dt/8| %.3e  "
                "orders %.3f %.3f\n",
                d1, d2, d3, std::log2(d1 / d2), std::log2(d2 / d3));
    EXPECT_GT(std::log2(d1 / d2), 0.9);
    EXPECT_GT(std::log2(d2 / d3), 0.9);
    EXPECT_LT(std::log2(d2 / d3), 1.1);
}
