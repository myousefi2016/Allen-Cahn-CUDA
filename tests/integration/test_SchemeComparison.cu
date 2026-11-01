#include "common/Gpu.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class SchemeComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }

    SimulationConfig make_config(int N, TimeScheme scheme) {
        SimulationConfig cfg;
        cfg.grid.Nx = N;
        cfg.grid.Ny = N;
        cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.05; // <= 1/15: well-posed without regularisation
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

// The explicit schemes' accuracy on this problem, of orders 1 (Euler), 2
// (Heun) and 4 (RK4) (docs/THEORY.md Appendix B). Each scheme runs to
// T = 10 dt0, dt0 = 0.0005, in steps dt_k = dt0 / 2^k (k = 0, 1, 2; dividing
// by a power of two is exact, so every run takes the same T), and its error is
// the RMS difference of phi and of u from RK4 at dt0 / 16. For each field:
//  - at every dt_k, err(RK4) < err(Heun) < err(Euler);
//  - each scheme's observed order log2(err(dt_k) / err(dt_k+1)) lies within
//    1/2 of its order q: nearer q than q - 1 or q + 1. The reference's own
//    error, (1/4)^4 = 1/256 of RK4's at dt0 / 4 (the smallest error compared),
//    changes an error ratio by at most a factor 257/255 (triangle
//    inequality), an order by at most log2(257/255) < 0.012.
// A Heun step without its average is two Euler steps: it integrates to 2T, so
// its error does not shrink with dt (order about 0) and exceeds Euler's.
TEST_F(SchemeComparisonTest, HeunIntermediateAccuracy) {
    constexpr int N = 16, steps0 = 10, levels = 3;
    const double dt0 = 0.0005;
    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi0(grid, "phi"), u0(grid, "u");
    make_sphere_ic(phi0, u0, N);

    struct Fields {
        FieldData phi, u;
    };
    // `scheme` to T = steps0 * dt0 in steps of dt0 / 2^k.
    auto run = [&](TimeScheme scheme, int k) {
        auto cfg = make_config(N, scheme);
        cfg.time.dt = dt0 / (1 << k);
        CudaSolver solver(cfg);
        solver.initialize(phi0, u0);
        for (int s = 0; s < (steps0 << k); ++s)
            solver.step(cfg.time.dt);
        Fields out{FieldData(grid, "phi"), FieldData(grid, "u")};
        solver.copy_phi_to_host(out.phi);
        solver.copy_u_to_host(out.u);
        return out;
    };
    const Fields ref = run(TimeScheme::RK4, 4);

    struct Scheme {
        TimeScheme scheme;
        const char* name;
        int order;
    };
    const Scheme schemes[] = {{TimeScheme::Euler, "Euler", 1},
                              {TimeScheme::Heun, "Heun", 2},
                              {TimeScheme::RK4, "RK4", 4}};
    double err[3][2][levels]; // [scheme][phi, u][k]
    for (int s = 0; s < 3; ++s)
        for (int k = 0; k < levels; ++k) {
            const Fields run_k = run(schemes[s].scheme, k);
            err[s][0][k] = compute_l2_diff(run_k.phi, ref.phi, N);
            err[s][1][k] = compute_l2_diff(run_k.u, ref.u, N);
        }

    const char* field[] = {"phi", "u"};
    for (int f = 0; f < 2; ++f) {
        for (int s = 0; s < 3; ++s) {
            const double* e = err[s][f];
            std::printf("%-3s %-5s RMS error %.3e %.3e %.3e, observed order %.3f %.3f\n", field[f],
                        schemes[s].name, e[0], e[1], e[2], std::log2(e[0] / e[1]),
                        std::log2(e[1] / e[2]));
            for (int k = 0; k + 1 < levels; ++k)
                EXPECT_NEAR(std::log2(e[k] / e[k + 1]), schemes[s].order, 0.5)
                    << field[f] << " " << schemes[s].name << ", dt0/" << (1 << k) << " to dt0/"
                    << (2 << k);
        }
        for (int k = 0; k < levels; ++k) {
            EXPECT_LT(err[2][f][k], err[1][f][k]) << field[f] << " at dt0/" << (1 << k);
            EXPECT_LT(err[1][f][k], err[0][f][k]) << field[f] << " at dt0/" << (1 << k);
        }
    }
}

// IMEX treats thermal diffusion implicitly. At dt = 0.04, h = 0.4, D = 2 the
// explicit thermal update has dt*D*rho = 6 (rho = 12/h^2, 7-point stencil),
// three times its stability limit 2 (docs/THEORY.md 10.1): validate()
// rejects the Euler configuration, and run anyway Euler overflows, while
// IMEX is accepted and stays finite. d0 = 0.1 keeps the phase-field update,
// explicit in both schemes, inside its own limit (0.0520 here, at a <111>
// gradient of the liquid; docs/THEORY.md 10.1).
TEST_F(SchemeComparisonTest, IMEXStableBeyondTheExplicitThermalLimit) {
    const int N = 16, steps = 500;
    const double dt = 0.04;
    auto config = [&](TimeScheme scheme) {
        auto cfg = make_config(N, scheme);
        cfg.physics.d0 = 0.1;
        cfg.time.dt = dt;
        return cfg;
    };
    auto finite_after_run = [&](const SimulationConfig& cfg) {
        CudaSolver solver(cfg);
        Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
        FieldData phi(grid, "phi"), u(grid, "u");
        make_sphere_ic(phi, u, N);
        solver.initialize(phi, u);
        for (int s = 0; s < steps; ++s)
            solver.step(dt);
        solver.copy_phi_to_host(phi);
        solver.copy_u_to_host(u);
        for (std::size_t i = 0; i < phi.size(); ++i)
            if (!std::isfinite(phi.data()[i]) || !std::isfinite(u.data()[i]))
                return false;
        return true;
    };
    const SimulationConfig euler = config(TimeScheme::Euler), imex = config(TimeScheme::IMEX);
    try {
        euler.validate();
        ADD_FAILURE() << "Euler at dt = 0.04 accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("explicit thermal update"), std::string::npos)
            << e.what();
    }
    EXPECT_FALSE(finite_after_run(euler)) << "dt is not beyond the explicit limit";
    EXPECT_NO_THROW(imex.validate());
    EXPECT_TRUE(finite_after_run(imex));
}
