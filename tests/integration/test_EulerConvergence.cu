#include "core/SimulationConfig.hpp"
#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class EulerConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    /// Run a simulation with the given dt and return the final phi field.
    FieldData run_simulation(double dt, int num_steps)
    {
        SimulationConfig cfg;
        cfg.grid.Nx = 16; cfg.grid.Ny = 16; cfg.grid.Nz = 16;
        cfg.grid.dx = 0.4; cfg.grid.dy = 0.4; cfg.grid.dz = 0.4;
        cfg.time.dt = dt;
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
                    u(x,y,z) = (r < r0) ? 0.0 : -cfg.physics.delta * (1.0 - std::exp(-(r-r0)));
                }

        CudaSolver solver(cfg);
        solver.initialize(phi, u);

        for (int step = 0; step < num_steps; ++step) {
            solver.step(dt);
        }

        solver.copy_phi_to_host(phi);
        return phi;
    }

    /// Compute L2 norm of difference between two fields.
    double l2_diff(const FieldData& a, const FieldData& b)
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            double d = a.data()[i] - b.data()[i];
            sum += d * d;
        }
        return std::sqrt(sum / static_cast<double>(a.size()));
    }
};

/// Test that halving dt roughly halves the error (first-order convergence).
/// We use a "reference" solution at very small dt and compare coarser solutions.
TEST_F(EulerConvergenceTest, FirstOrderConvergence)
{
    double T = 0.1;  // Total simulation time

    // Reference solution with small dt
    double dt_ref = 0.0005;
    int steps_ref = static_cast<int>(T / dt_ref);
    auto phi_ref = run_simulation(dt_ref, steps_ref);

    // Coarse solution
    double dt1 = 0.01;
    int steps1 = static_cast<int>(T / dt1);
    auto phi1 = run_simulation(dt1, steps1);

    // Medium solution
    double dt2 = 0.005;
    int steps2 = static_cast<int>(T / dt2);
    auto phi2 = run_simulation(dt2, steps2);

    double err1 = l2_diff(phi1, phi_ref);
    double err2 = l2_diff(phi2, phi_ref);

    // For first-order method, err1/err2 should be approximately dt1/dt2 = 2.0
    // Allow generous tolerance due to nonlinearity
    if (err1 > 1e-10 && err2 > 1e-10) {
        double ratio = err1 / err2;
        EXPECT_GT(ratio, 1.2) << "Halving dt should reduce error";
        EXPECT_LT(ratio, 4.0) << "Convergence ratio out of expected range";
    }

    // Both errors should be small
    EXPECT_LT(err1, 1.0) << "Error with dt=0.01 is too large";
    EXPECT_LT(err2, 1.0) << "Error with dt=0.005 is too large";
}

/// Test that finer dt produces results closer to the reference.
TEST_F(EulerConvergenceTest, FinerDtReducesError)
{
    double T = 0.05;

    double dt_ref = 0.0002;
    auto phi_ref = run_simulation(dt_ref, static_cast<int>(T / dt_ref));

    double dt_coarse = 0.005;
    auto phi_coarse = run_simulation(dt_coarse, static_cast<int>(T / dt_coarse));

    double dt_fine = 0.001;
    auto phi_fine = run_simulation(dt_fine, static_cast<int>(T / dt_fine));

    double err_coarse = l2_diff(phi_coarse, phi_ref);
    double err_fine = l2_diff(phi_fine, phi_ref);

    EXPECT_LT(err_fine, err_coarse)
        << "Finer dt should produce smaller error than coarser dt";
}
