#include "common/TempDir.hpp"
#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

using namespace ac;

class InitialConditionTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("test_initial_condition");
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

    SimulationConfig make_config() {
        SimulationConfig cfg;
        cfg.grid.Nx = 16;
        cfg.grid.Ny = 16;
        cfg.grid.Nz = 16;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;

        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.07;
        cfg.physics.W0 = 1.0;
        cfg.physics.D = 2.0;
        cfg.physics.d0 = 0.5;

        cfg.time.dt = 0.001;
        cfg.time.max_steps = 0; // Run zero steps to inspect IC only
        cfg.time.scheme = TimeScheme::Euler;
        cfg.time.adaptive = false;
        cfg.time.dt_min = 1e-6;
        cfg.time.dt_max = 0.1;

        cfg.stencil = StencilType::Standard7Point;

        cfg.initial.seed_radius = 1.6; // physical: 4 cells at dx = 0.4

        cfg.output.output_dir = test_dir_ / "out";
        cfg.output.frequency = 10000;
        cfg.output.async_io = false;

        cfg.checkpoint.checkpoint_dir = test_dir_ / "ckpt";
        cfg.checkpoint.frequency = 10000;
        cfg.checkpoint.keep_last = 1;

        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.8;

        cfg.gpu.device_ids = {0};
        cfg.gpu.multi_gpu = false;

        return cfg;
    }

    std::filesystem::path test_dir_;
};

/// Analytic initial profile at physical distance r from the domain midpoint.
static double expected_phi(const SimulationConfig& cfg, int x, int y, int z) {
    const double cx = 0.5 * (cfg.grid.Nx - 1), cy = 0.5 * (cfg.grid.Ny - 1),
                 cz = 0.5 * (cfg.grid.Nz - 1);
    const double rx = (x - cx) * cfg.grid.dx, ry = (y - cy) * cfg.grid.dy,
                 rz = (z - cz) * cfg.grid.dz;
    const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
    return -std::tanh((r - cfg.initial.seed_radius) / (std::sqrt(2.0) * cfg.physics.W0));
}

/// Every interior cell holds the equilibrium profile -tanh((r - r0)/(sqrt(2) W0))
/// with r measured in physical units (grid spacing 0.4, so 2.5 cells per W0).
TEST_F(InitialConditionTest, PhiMatchesEquilibriumProfile) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run(); // 0 steps, just initializes

    const auto& phi = engine.phi();
    for (int x = 1; x < cfg.grid.Nx - 1; ++x)
        for (int y = 1; y < cfg.grid.Ny - 1; ++y)
            for (int z = 1; z < cfg.grid.Nz - 1; ++z)
                ASSERT_DOUBLE_EQ(phi(x, y, z), expected_phi(cfg, x, y, z))
                    << x << "," << y << "," << z;
}

/// The seed is centred at the domain midpoint 0.5*(N-1), so the field is
/// exactly mirror-symmetric about all three mid-planes (including the
/// Neumann boundary cells, which copy symmetric neighbours).
TEST_F(InitialConditionTest, SeedIsMirrorSymmetric) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    const int N = cfg.grid.Nx;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                ASSERT_EQ(phi(x, y, z), phi(N - 1 - x, y, z));
                ASSERT_EQ(phi(x, y, z), phi(x, N - 1 - y, z));
                ASSERT_EQ(phi(x, y, z), phi(x, y, N - 1 - z));
            }
}

/// phi is positive (solid) strictly inside the seed radius and negative
/// (liquid) strictly outside it, so the phi = 0 surface is the sphere r = r0.
TEST_F(InitialConditionTest, InterfaceAtSeedRadius) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    const double c = 0.5 * (cfg.grid.Nx - 1);
    int inside = 0, outside = 0;
    for (int x = 1; x < cfg.grid.Nx - 1; ++x)
        for (int y = 1; y < cfg.grid.Ny - 1; ++y)
            for (int z = 1; z < cfg.grid.Nz - 1; ++z) {
                const double h = cfg.grid.dx;
                const double r =
                    h * std::sqrt((x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c));
                if (r < cfg.initial.seed_radius) {
                    ASSERT_GT(phi(x, y, z), 0.0);
                    ++inside;
                } else if (r > cfg.initial.seed_radius) {
                    ASSERT_LT(phi(x, y, z), 0.0);
                    ++outside;
                }
            }
    EXPECT_GT(inside, 0);
    EXPECT_GT(outside, 0);
}

/// Verify that u = -delta everywhere.
TEST_F(InitialConditionTest, UniformUndercooling) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run();

    const auto& u = engine.u();
    double expected_u = -cfg.physics.delta;
    int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;

    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                EXPECT_DOUBLE_EQ(u(x, y, z), expected_u)
                    << "u should be -delta at (" << x << "," << y << "," << z << ")";
            }
}
