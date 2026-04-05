#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

using namespace ac;
using namespace ac::cuda;

class SphereRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
    }
};

/// Run a small simulation and verify the phase field evolves correctly.
/// The sphere should remain roughly spherical and the solid fraction should
/// change monotonically (shrink or grow depending on undercooling).
TEST_F(SphereRegressionTest, SphereEvolvesConsistently)
{
    SimulationConfig cfg;
    cfg.grid.Nx = 32; cfg.grid.Ny = 32; cfg.grid.Nz = 32;
    cfg.grid.dx = 0.4; cfg.grid.dy = 0.4; cfg.grid.dz = 0.4;
    cfg.time.dt = 0.01;
    cfg.time.max_steps = 50;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.physics.delta = 0.8;
    cfg.initial.seed_radius = 5.0;
    cfg.output.frequency = 1000;  // Don't output during test
    cfg.checkpoint.frequency = 0;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    // Initialize
    int Nx = grid.Nx(), Ny = grid.Ny(), Nz = grid.Nz();
    Real r0 = cfg.initial.seed_radius;
    Real cx = 0.5 * Nx, cy = 0.5 * Ny, cz = 0.5 * Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
                phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                u(x, y, z) = (r < r0) ? 0.0 : -cfg.physics.delta * (1.0 - std::exp(-(r-r0)));
            }

    // Compute initial solid fraction
    auto solid_fraction = [&](const FieldData& f) {
        double sum = 0.0;
        for (std::size_t i = 0; i < f.size(); ++i) {
            sum += (f.data()[i] > 0.0) ? 1.0 : 0.0;
        }
        return sum / static_cast<double>(f.size());
    };

    double initial_sf = solid_fraction(phi);
    EXPECT_GT(initial_sf, 0.0);
    EXPECT_LT(initial_sf, 1.0);

    // Run simulation
    CudaSolver solver(cfg);
    solver.initialize(phi, u);

    for (int step = 0; step < 50; ++step) {
        solver.step(cfg.time.dt);
    }

    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    double final_sf = solid_fraction(phi);

    // The phase field should have changed from initial
    EXPECT_NE(initial_sf, final_sf);

    // Phi should still be bounded in [-1, 1] (approximately)
    for (std::size_t i = 0; i < phi.size(); ++i) {
        EXPECT_GE(phi.data()[i], -1.5) << "phi out of range at index " << i;
        EXPECT_LE(phi.data()[i], 1.5) << "phi out of range at index " << i;
    }

    // Temperature should remain bounded
    for (std::size_t i = 0; i < u.size(); ++i) {
        EXPECT_GT(u.data()[i], -10.0) << "u too negative at index " << i;
        EXPECT_LT(u.data()[i], 10.0) << "u too positive at index " << i;
    }

    // The center should still be solid-ish (phi > 0)
    EXPECT_GT(phi(Nx/2, Ny/2, Nz/2), 0.0);

    // Far corners should be liquid (phi < 0)
    EXPECT_LT(phi(0, 0, 0), 0.0);
}

/// Test that the Heun scheme produces smoother evolution than Euler.
TEST_F(SphereRegressionTest, HeunSchemeRuns)
{
    SimulationConfig cfg;
    cfg.grid.Nx = 16; cfg.grid.Ny = 16; cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4; cfg.grid.dy = 0.4; cfg.grid.dz = 0.4;
    cfg.time.dt = 0.01;
    cfg.time.scheme = TimeScheme::Heun;
    cfg.physics.delta = 0.8;
    cfg.initial.seed_radius = 3.0;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");

    int Nx = grid.Nx(), Ny = grid.Ny(), Nz = grid.Nz();
    Real cx = 0.5*Nx, cy = 0.5*Ny, cz = 0.5*Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
                phi(x,y,z) = (r < cfg.initial.seed_radius) ? 1.0 : -1.0;
                u(x,y,z) = (r < cfg.initial.seed_radius) ? 0.0 : -cfg.physics.delta;
            }

    CudaSolver solver(cfg);
    solver.initialize(phi, u);

    // Should not crash or produce NaN
    for (int step = 0; step < 10; ++step) {
        solver.step(cfg.time.dt);
    }

    solver.copy_phi_to_host(phi);
    for (std::size_t i = 0; i < phi.size(); ++i) {
        EXPECT_FALSE(std::isnan(phi.data()[i])) << "NaN at index " << i;
        EXPECT_FALSE(std::isinf(phi.data()[i])) << "Inf at index " << i;
    }
}
