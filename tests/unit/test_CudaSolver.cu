#include "common/Gpu.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class CudaSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }

    SimulationConfig make_config(int N = 16, TimeScheme scheme = TimeScheme::Euler) {
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
        cfg.time.dt = 0.001;
        cfg.time.scheme = scheme;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.8;
        return cfg;
    }

    void make_sphere_ic(FieldData& phi, FieldData& u, int N, double r0 = 3.0) {
        double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    double r =
                        std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                    phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                    u(x, y, z) = (r < r0) ? 0.0 : -0.8;
                }
    }
};

// The solver checks the point count itself (it may be given an unvalidated config), before
// it allocates anything, and without overflow: 2^21 points per axis are 2^63 points.
TEST_F(CudaSolverTest, RejectsGridsBeyond32BitIndicesBeforeAllocating) {
    auto cfg = make_config();
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 1 << 21;
    try {
        CudaSolver solver(cfg);
        FAIL() << "constructed a solver for 2^63 points";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("32-bit indexing"), std::string::npos) << e.what();
    }
}

TEST_F(CudaSolverTest, EulerStepPreservesSymmetry) {
    int N = 16;
    auto cfg = make_config(N, TimeScheme::Euler);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    solver.step(0.001);
    solver.copy_phi_to_host(phi);

    // Check that solution is not all zeros (something happened)
    double sum = 0.0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                sum += std::abs(phi(x, y, z));
    EXPECT_GT(sum, 0.0);
}

TEST_F(CudaSolverTest, HeunStepRuns) {
    int N = 16;
    auto cfg = make_config(N, TimeScheme::Heun);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    EXPECT_NO_THROW(solver.step(0.001));

    solver.copy_phi_to_host(phi);
    double sum = 0.0;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                sum += std::abs(phi(x, y, z));
    EXPECT_GT(sum, 0.0);
}

// A Heun step is the average of the state and two Euler steps from it
// (CudaSolver::step_heun_stage1/2). With E the Euler step (Allen-Cahn update
// and phi BC, then the thermal update with latent heat and u BC),
//   y_{n+1} = BC(0.5 (y_n + E(E(y_n)))) = y_n + dt/2 (f(y_n) + f(E(y_n))).
// The stages launch the kernels that step_euler launches, on the same
// operands, so an Euler solver started from y_n reproduces E(y_n) and
// E(E(y_n)) bit for bit. y_n and E(E(y_n)) satisfy the BCs: a zero-flux
// Neumann phi cell holds the cell next to it inward + 0.0 * h, an exact copy
// (an interior cell, or at edges and corners a boundary cell that an earlier
// face of the Z, Y, X order already set), and Dirichlet u writes -0.8. So the
// final BC leaves their average unchanged: a boundary cell's average is that
// of the cell it copies, and 0.5 (v + v) = v. 0.5 (a + b) is one rounded sum
// and an exact halving on the host and on the device alike, so the comparison
// is exact. Two steps: the second starts from the state the first left in the
// Heun solver.
TEST_F(CudaSolverTest, HeunStepIsTheAverageOfTheStateAndTwoEulerSteps) {
    const int N = 16;
    const double h = 0.4, dt = 0.001;
    Grid grid(Dim3{N, N, N}, Spacing{h, h, h});

    // A diffuse seed, phi = -tanh((r - 2.4) / sqrt(2)) (W0 = 1), centred at
    // (1.0, 3.0, 3.1) and cut by the x = 0 wall, with u from -0.8 in the liquid
    // to 0 in the solid: every term of both updates is active, also next to
    // the wall, where the stencils read the cells the zero-flux BC copies.
    FieldData phi(grid, "phi"), u(grid, "u");
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                const double rx = x * h - 1.0, ry = y * h - 3.0, rz = z * h - 3.1;
                const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                phi(x, y, z) = -std::tanh((r - 2.4) / std::sqrt(2.0));
                u(x, y, z) = -0.4 * (1.0 - phi(x, y, z));
            }
    CudaSolver heun(make_config(N, TimeScheme::Heun));
    heun.initialize(phi, u);
    heun.copy_phi_to_host(phi); // y_0: the BCs applied
    heun.copy_u_to_host(u);

    // got == 0.5 (y + e2) in every cell, and that average differs from both y
    // and e2 somewhere (else a missing average or a missing step would not
    // change the result).
    auto expect_average =
        [&](const FieldData& got, const FieldData& y, const FieldData& e2, const char* field) {
            std::size_t mismatches = 0, distinct = 0, first = 0;
            for (std::size_t i = 0; i < got.size(); ++i) {
                const double avg = 0.5 * (y.data()[i] + e2.data()[i]);
                if (avg != y.data()[i] && avg != e2.data()[i])
                    ++distinct;
                if (got.data()[i] != avg && mismatches++ == 0)
                    first = i;
            }
            const auto n = static_cast<std::size_t>(N);
            EXPECT_EQ(mismatches, 0u)
                << field << ": first at (" << first / (n * n) << ", " << first / n % n << ", "
                << first % n << "): " << got.data()[first]
                << " vs 0.5 (y + E(E(y))) = " << 0.5 * (y.data()[first] + e2.data()[first]);
            EXPECT_GT(distinct, 0u) << field << ": E(E(y)) and y do not differ";
        };

    for (int step = 1; step <= 2; ++step) {
        SCOPED_TRACE("step " + std::to_string(step));
        FieldData e1_phi(grid, "e1_phi"), e2_phi(grid, "e2_phi"), e2_u(grid, "e2_u");
        CudaSolver euler(make_config(N, TimeScheme::Euler));
        euler.initialize(phi, u);
        euler.step(dt);
        euler.copy_phi_to_host(e1_phi);
        euler.step(dt);
        euler.copy_phi_to_host(e2_phi);
        euler.copy_u_to_host(e2_u);

        heun.step(dt);
        FieldData h_phi(grid, "h_phi"), h_u(grid, "h_u");
        heun.copy_phi_to_host(h_phi);
        heun.copy_u_to_host(h_u);
        expect_average(h_phi, phi, e2_phi, "phi");
        expect_average(h_u, u, e2_u, "u");

        // The corrector's phi depends on the predicted u: an Euler step from
        // (phi of E(y_n), u_n) is the corrector reading u_n instead, and its
        // average with phi_n differs from the reference somewhere, so such a
        // corrector cannot pass the comparison above.
        CudaSolver stale(make_config(N, TimeScheme::Euler));
        stale.initialize(e1_phi, u);
        stale.step(dt);
        FieldData stale_phi(grid, "stale_phi");
        stale.copy_phi_to_host(stale_phi);
        std::size_t coupled = 0;
        for (std::size_t i = 0; i < phi.size(); ++i) {
            const double p = phi.data()[i];
            if (0.5 * (p + stale_phi.data()[i]) != 0.5 * (p + e2_phi.data()[i]))
                ++coupled;
        }
        EXPECT_GT(coupled, 0u);

        phi = h_phi;
        u = h_u;
    }
}

TEST_F(CudaSolverTest, RK4StepIncludesLatentHeat) {
    int N = 16;
    auto cfg = make_config(N, TimeScheme::RK4);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    solver.step(0.001);

    FieldData phi_after(grid, "phi_after"), u_after(grid, "u_after");
    solver.copy_phi_to_host(phi_after);
    solver.copy_u_to_host(u_after);

    // If latent heat is working, u should change near the interface
    // where phi is changing. Check that u differs from initial at some point.
    bool u_changed = false;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                if (std::abs(u_after(x, y, z) - u(x, y, z)) > 1e-15) {
                    u_changed = true;
                }
            }
    EXPECT_TRUE(u_changed) << "u field should change due to latent heat coupling in RK4";
}

TEST_F(CudaSolverTest, IMEXStepRuns) {
    int N = 16;
    auto cfg = make_config(N, TimeScheme::IMEX);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    EXPECT_NO_THROW(solver.step(0.001));
}

TEST_F(CudaSolverTest, ComputeMaxDphi) {
    int N = 16;
    auto cfg = make_config(N);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    solver.step(0.001);

    double max_dphi = solver.compute_max_dphi();
    EXPECT_GT(max_dphi, 0.0);
    EXPECT_LT(max_dphi, 10.0); // Sanity bound
}

TEST_F(CudaSolverTest, PerFaceBoundaryConditions) {
    int N = 16;
    auto cfg = make_config(N);
    cfg.boundary.per_face = true;
    // X faces: Neumann (zero flux)
    cfg.boundary.phi_faces = PerFaceBoundary::uniform(cfg.boundary.phi_bc);
    cfg.boundary.u_faces = PerFaceBoundary::uniform(cfg.boundary.u_bc);

    // Set X_lo explicitly through the per-face table (zero-flux Neumann)
    cfg.boundary.phi_faces[Face::XLo].type = BCType::Neumann;
    cfg.boundary.phi_faces[Face::XLo].flux = 0.0;

    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    EXPECT_NO_THROW(solver.step(0.001));
    EXPECT_NO_THROW(solver.apply_boundary_conditions());
}

// Equilibrium phi = -1, u = -0.8 with Dirichlet u = -0.8 is a fixed point of the
// coupled system: the Allen-Cahn RHS vanishes, so the IMEX thermal solve is
// (I - dt*D*Lap) u = u_old whose exact solution is u = -0.8 everywhere. A large
// dt (dt*D/h^2 = 0.625) makes any corruption of the Jacobi iterate's boundary
// cells visible in the cells next to every wall.
TEST_F(CudaSolverTest, IMEXPreservesUniformEquilibrium) {
    const int N = 12;
    auto cfg = make_config(N, TimeScheme::IMEX);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    solver.initialize(phi, u);
    for (int s = 0; s < 3; ++s)
        solver.step(0.05);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                ASSERT_NEAR(phi(x, y, z), -1.0, 1e-12) << x << "," << y << "," << z;
                ASSERT_NEAR(u(x, y, z), -0.8, 1e-12) << x << "," << y << "," << z;
            }
}

// Per-face BCs must be enforced by every time-integration stage, not only by
// initialize(). The uniform phi_bc/u_bc are deliberately left at values that
// differ from every face so that any stage falling back to them is detected.
TEST_F(CudaSolverTest, PerFaceBoundaryConditionsHoldAfterStepping) {
    const int N = 12;
    const TimeScheme schemes[] = {TimeScheme::Euler, TimeScheme::Heun, TimeScheme::RK4,
                                  TimeScheme::IMEX};
    for (TimeScheme scheme : schemes) {
        SCOPED_TRACE(static_cast<int>(scheme));
        auto cfg = make_config(N, scheme);
        cfg.boundary.phi_bc.type = BCType::Dirichlet; // must never be applied
        cfg.boundary.phi_bc.value = 0.9;
        cfg.boundary.u_bc.type = BCType::Dirichlet; // must never be applied
        cfg.boundary.u_bc.value = 0.9;
        cfg.boundary.per_face = true;

        BoundaryConfig dir;
        dir.type = BCType::Dirichlet;
        BoundaryConfig neu;
        neu.type = BCType::Neumann;
        neu.flux = 0.0;

        auto& pf = cfg.boundary.phi_faces;
        pf[Face::XLo] = dir;
        pf[Face::XLo].value = 0.3;
        pf[Face::XHi] = dir;
        pf[Face::XHi].value = -0.7;
        pf[Face::YLo] = neu;
        pf[Face::YHi] = neu;
        pf[Face::ZLo] = dir;
        pf[Face::ZLo].value = 0.1;
        pf[Face::ZHi] = dir;
        pf[Face::ZHi].value = 0.2;

        cfg.boundary.u_faces = PerFaceBoundary::uniform(dir);
        for (auto& f : cfg.boundary.u_faces.faces)
            f.value = -0.8;
        cfg.boundary.u_faces[Face::XLo].value = -0.5;

        CudaSolver solver(cfg);
        Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
        FieldData phi(grid, "phi"), u(grid, "u");
        make_sphere_ic(phi, u, N);
        solver.initialize(phi, u);
        for (int s = 0; s < 3; ++s)
            solver.step(0.001);
        solver.copy_phi_to_host(phi);
        solver.copy_u_to_host(u);

        // X faces own their whole plane (applied last).
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                ASSERT_DOUBLE_EQ(phi(0, y, z), 0.3) << "y=" << y << " z=" << z;
                ASSERT_DOUBLE_EQ(phi(N - 1, y, z), -0.7) << "y=" << y << " z=" << z;
                ASSERT_DOUBLE_EQ(u(0, y, z), -0.5) << "y=" << y << " z=" << z;
            }
        // Y faces (zero-flux Neumann) own x in [1, N-2] and every z.
        for (int x = 1; x < N - 1; ++x)
            for (int z = 0; z < N; ++z) {
                ASSERT_DOUBLE_EQ(phi(x, 0, z), phi(x, 1, z)) << "x=" << x << " z=" << z;
                ASSERT_DOUBLE_EQ(phi(x, N - 1, z), phi(x, N - 2, z)) << "x=" << x << " z=" << z;
            }
        // Z faces own the remaining interior of their plane.
        for (int x = 1; x < N - 1; ++x)
            for (int y = 1; y < N - 1; ++y) {
                ASSERT_DOUBLE_EQ(phi(x, y, 0), 0.1) << "x=" << x << " y=" << y;
                ASSERT_DOUBLE_EQ(phi(x, y, N - 1), 0.2) << "x=" << x << " y=" << y;
                ASSERT_DOUBLE_EQ(u(x, y, N - 1), -0.8) << "x=" << x << " y=" << y;
            }
    }
}

// NaN in the field must surface in both reductions the engine relies on
// (adaptive dt and the saturation guard) instead of being dropped by fmax.
TEST_F(CudaSolverTest, ReductionsPropagateNaN) {
    const int N = 12;
    auto cfg = make_config(N);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);
    // Next to the x_lo wall: the zero-flux Neumann BC copies it onto the wall.
    phi(1, 5, 5) = std::nan("");
    solver.initialize(phi, u);

    EXPECT_TRUE(std::isnan(solver.compute_boundary_max_phi()));
    solver.step(0.001);
    EXPECT_TRUE(std::isnan(solver.compute_max_dphi()));
}

TEST_F(CudaSolverTest, MultipleStepsConverge) {
    int N = 16;
    auto cfg = make_config(N);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    // Start with uniform phi=-1 (liquid), u=-0.8
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                phi(x, y, z) = -1.0;
                u(x, y, z) = -0.8;
            }

    solver.initialize(phi, u);

    // Run 10 steps - should be stable with uniform field
    for (int i = 0; i < 10; ++i) {
        solver.step(0.001);
    }

    solver.copy_phi_to_host(phi);

    // Interior should still be approximately -1 (no driving force)
    double center = phi(N / 2, N / 2, N / 2);
    EXPECT_NEAR(center, -1.0, 0.1);
}
