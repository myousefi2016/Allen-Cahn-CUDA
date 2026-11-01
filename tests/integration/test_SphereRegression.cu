#include "common/EulerReference.hpp"
#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <string>

using namespace ac;
using namespace ac::cuda;
using ac::test::LD;

class SphereRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// Every one of 50 Euler steps from a phi = 1 ball of radius 5 cells is the discrete scheme's
/// step at every cell. The run is 32^3 at h = 0.4 with default physics (eps = 0.05),
/// delta = 0.8, and Dirichlet phi = -1 and u = -0.8 on the walls. From the GPU's fields before
/// a step, the host reference (common/EulerReference.hpp: the kernels' operators written
/// independently, in long double) gives phi and u after it to within its bound, the sum of the
/// forward rounding-error bounds of the GPU's and the reference's arithmetic. Each comparison
/// starts from the GPU's own state, which keeps earlier steps' rounding out of it: the bound does
/// not grow along the run, and the run is the scheme's trajectory to within one bound per step.
/// Each step must also move phi, and release latent heat, by more than the bound somewhere, so
/// a step that leaves phi unchanged, or drops or misplaces the latent heat, fails.
TEST_F(SphereRegressionTest, EulerRunIsTheDiscreteSchemeAtEveryStep) {
    SimulationConfig cfg;
    cfg.grid.Nx = 32;
    cfg.grid.Ny = 32;
    cfg.grid.Nz = 32;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.4;
    cfg.time.dt = 0.01;
    cfg.time.max_steps = 50;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.physics.delta = 0.8;
    cfg.initial.seed_radius = 5.0;
    cfg.output.frequency = 1000;
    cfg.checkpoint.frequency = 0;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    int Nx = grid.Nx(), Ny = grid.Ny(), Nz = grid.Nz();
    Real r0 = cfg.initial.seed_radius;
    Real cx = 0.5 * Nx, cy = 0.5 * Ny, cz = 0.5 * Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                u(x, y, z) = (r < r0) ? 0.0 : -cfg.physics.delta * (1.0 - std::exp(-(r - r0)));
            }

    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);
    const ac::test::EulerReference ref(cfg);
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
                if (ref.is_boundary(x, y, z)) {
                    ASSERT_EQ(phi(x, y, z), -1.0) << x << "," << y << "," << z;
                    ASSERT_EQ(u(x, y, z), -cfg.physics.delta) << x << "," << y << "," << z;
                }

    FieldData phi_next(grid, "phi_next"), u_next(grid, "u_next");
    for (int step = 1; step <= cfg.time.max_steps; ++step) {
        SCOPED_TRACE("step " + std::to_string(step));
        solver.step(cfg.time.dt);
        solver.copy_phi_to_host(phi_next);
        solver.copy_u_to_host(u_next);
        const auto phi_ref = ref.phi_step(phi.data(), u.data(), cfg.time.dt);
        const auto u_ref = ref.u_step(u.data(), phi.data(), phi_next.data(), cfg.time.dt);
        LD phi_margin = -1.0L, latent_margin = -1.0L;
        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z) {
                    const std::size_t c = ref.index(x, y, z);
                    ASSERT_LE(std::fabs(phi_next.data()[c] - phi_ref.value[c]), phi_ref.bound[c])
                        << "phi at " << x << "," << y << "," << z << ": GPU " << phi_next.data()[c]
                        << ", reference " << static_cast<double>(phi_ref.value[c]);
                    ASSERT_LE(std::fabs(u_next.data()[c] - u_ref.value[c]), u_ref.bound[c])
                        << "u at " << x << "," << y << "," << z << ": GPU " << u_next.data()[c]
                        << ", reference " << static_cast<double>(u_ref.value[c]);
                    phi_margin = std::max(phi_margin, std::fabs(phi_ref.value[c] - phi.data()[c]) -
                                                          phi_ref.bound[c]);
                    latent_margin = std::max(
                        latent_margin,
                        std::fabs(static_cast<LD>(phi_next.data()[c]) - phi.data()[c]) / 2 -
                            2 * u_ref.bound[c]);
                }
        ASSERT_GT(phi_margin, 0.0L) << "a step that leaves phi unchanged would pass";
        ASSERT_GT(latent_margin, 0.0L) << "a u update without the latent heat would pass";
        std::copy(phi_next.data(), phi_next.data() + phi_next.size(), phi.data());
        std::copy(u_next.data(), u_next.data() + u_next.size(), u.data());
    }
}

/// Smoke test: 10 Heun steps at dt = 0.01 (accepted by validate()) leave phi finite.
/// Heun's accuracy is tested by SchemeComparisonTest.HeunIntermediateAccuracy and
/// CudaSolverTest.HeunStepIsTheAverageOfTheStateAndTwoEulerSteps.
TEST_F(SphereRegressionTest, HeunSchemeRuns) {
    SimulationConfig cfg;
    cfg.grid.Nx = 16;
    cfg.grid.Ny = 16;
    cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.4;
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
    Real cx = 0.5 * Nx, cy = 0.5 * Ny, cz = 0.5 * Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                phi(x, y, z) = (r < cfg.initial.seed_radius) ? 1.0 : -1.0;
                u(x, y, z) = (r < cfg.initial.seed_radius) ? 0.0 : -cfg.physics.delta;
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
