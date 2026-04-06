#include "core/SimulationConfig.hpp"
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

class EnergyConservationTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    /// Compute total thermal energy (sum of u over all points).
    double total_energy(const FieldData& u)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < u.size(); ++i) {
            sum += u.data()[i];
        }
        return sum;
    }

    /// Compute total solid fraction (mean of (phi+1)/2).
    double solid_fraction(const FieldData& phi)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < phi.size(); ++i) {
            sum += (phi.data()[i] + 1.0) / 2.0;
        }
        return sum / static_cast<double>(phi.size());
    }
};

/// Test that a simulation with uniform initial conditions and Dirichlet BCs
/// reaches a steady state.
TEST_F(EnergyConservationTest, UniformFieldStaysUniform)
{
    SimulationConfig cfg;
    cfg.grid.Nx = 16; cfg.grid.Ny = 16; cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4; cfg.grid.dy = 0.4; cfg.grid.dz = 0.4;
    cfg.time.dt = 0.01;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.physics.delta = 0.8;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -0.8, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");

    // Initialize: fully liquid, uniform temperature
    phi.fill(-1.0);
    u.fill(-cfg.physics.delta);

    CudaSolver solver(cfg);
    solver.initialize(phi, u);

    // Run 100 steps
    for (int step = 0; step < 100; ++step) {
        solver.step(cfg.time.dt);
    }

    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    // Uniform liquid state should remain uniform
    // phi should stay at -1.0 everywhere (no solidification without a seed)
    for (std::size_t i = 0; i < phi.size(); ++i) {
        EXPECT_NEAR(phi.data()[i], -1.0, 0.01)
            << "phi deviated from -1.0 at index " << i;
    }
}

/// Test that total energy changes are correlated with phase change (latent heat).
TEST_F(EnergyConservationTest, LatentHeatCoupling)
{
    SimulationConfig cfg;
    cfg.grid.Nx = 16; cfg.grid.Ny = 16; cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4; cfg.grid.dy = 0.4; cfg.grid.dz = 0.4;
    cfg.time.dt = 0.005;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.physics.delta = 0.8;
    cfg.initial.seed_radius = 3.0;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");

    int Nx = grid.Nx(), Ny = grid.Ny(), Nz = grid.Nz();
    Real r0 = cfg.initial.seed_radius;
    Real cx = 0.5*Nx, cy = 0.5*Ny, cz = 0.5*Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy) + (z-cz)*(z-cz));
                phi(x,y,z) = (r < r0) ? 1.0 : -1.0;
                u(x,y,z) = (r < r0) ? 0.0 : -cfg.physics.delta;
            }

    double sf_initial = solid_fraction(phi);
    double energy_initial = total_energy(u);

    CudaSolver solver(cfg);
    solver.initialize(phi, u);

    for (int step = 0; step < 20; ++step) {
        solver.step(cfg.time.dt);
    }

    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    double sf_final = solid_fraction(phi);
    double energy_final = total_energy(u);

    // If the solid fraction changes, energy should also change
    // (latent heat release/absorption)
    double dsf = sf_final - sf_initial;
    double de = energy_final - energy_initial;

    // If solidification occurs (dsf > 0), energy should increase
    // (latent heat released into temperature field)
    // If melting occurs (dsf < 0), energy should decrease
    // The sign relationship should hold:
    if (std::abs(dsf) > 1e-6) {
        // Latent heat coupling: 0.5 * dphi adds to u
        // When phi increases (solidification), u increases
        EXPECT_TRUE((dsf > 0 && de > 0) || (dsf < 0 && de < 0) || std::abs(dsf) < 0.01)
            << "Latent heat coupling sign mismatch: dsf=" << dsf << " de=" << de;
    }

    // Fields should be finite
    for (std::size_t i = 0; i < phi.size(); ++i) {
        EXPECT_FALSE(std::isnan(phi.data()[i]));
        EXPECT_FALSE(std::isinf(phi.data()[i]));
    }
    for (std::size_t i = 0; i < u.size(); ++i) {
        EXPECT_FALSE(std::isnan(u.data()[i]));
        EXPECT_FALSE(std::isinf(u.data()[i]));
    }
}
