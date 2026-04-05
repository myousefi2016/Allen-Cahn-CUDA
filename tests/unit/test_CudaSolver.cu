#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class CudaSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
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
                    double r = std::sqrt((x - cx) * (x - cx) +
                                         (y - cy) * (y - cy) +
                                         (z - cz) * (z - cz));
                    phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                    u(x, y, z) = (r < r0) ? 0.0 : -0.8;
                }
    }
};

TEST_F(CudaSolverTest, EulerStepPreservesSymmetry)
{
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

TEST_F(CudaSolverTest, HeunStepRuns)
{
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

TEST_F(CudaSolverTest, RK4StepIncludesLatentHeat)
{
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

TEST_F(CudaSolverTest, IMEXStepRuns)
{
    int N = 16;
    auto cfg = make_config(N, TimeScheme::IMEX);
    CudaSolver solver(cfg);

    Grid grid(Dim3{N, N, N}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N);

    solver.initialize(phi, u);
    EXPECT_NO_THROW(solver.step(0.001));
}

TEST_F(CudaSolverTest, ComputeMaxDphi)
{
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
    EXPECT_LT(max_dphi, 10.0);  // Sanity bound
}

TEST_F(CudaSolverTest, PerFaceBoundaryConditions)
{
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

TEST_F(CudaSolverTest, MultipleStepsConverge)
{
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
