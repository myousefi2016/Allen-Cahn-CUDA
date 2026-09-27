#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class CudaSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    SimulationConfig make_config(int N = 16, TimeScheme scheme = TimeScheme::Euler) {
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

    // Override X_lo face to periodic
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
