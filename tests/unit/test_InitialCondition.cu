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
        test_dir_ = std::filesystem::temp_directory_path() / "test_initial_condition";
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

        cfg.initial.seed_radius = 4.0;

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

/// Verify phi at the center of the domain (inside the seed).
/// Center is at (8,8,8) in a 16^3 grid with cx=cy=cz=8.
/// r = 0, phi = -tanh((0-4)/(sqrt(2)*1)) = -tanh(-2.828) ≈ +0.9937
TEST_F(InitialConditionTest, PhiAtCenter) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run(); // 0 steps, just initializes

    const auto& phi = engine.phi();
    double r0 = cfg.initial.seed_radius;
    double W0 = cfg.physics.W0;
    double inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * W0);
    double expected = -std::tanh((0.0 - r0) * inv_sqrt2_W0);

    EXPECT_NEAR(phi(8, 8, 8), expected, 1e-4) << "phi at center should be ≈ " << expected;
    EXPECT_GT(phi(8, 8, 8), 0.99) << "phi at center should be close to +1 (solid)";
}

/// Verify phi far from center (liquid region).
/// At (0,0,0): r = sqrt(8^2+8^2+8^2) = 8*sqrt(3) ≈ 13.86
/// phi = -tanh((13.86-4)/sqrt(2)) = -tanh(6.97) ≈ -1.0
TEST_F(InitialConditionTest, PhiFarFromCenter) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    int Nx = cfg.grid.Nx;
    double cx = 0.5 * Nx;
    double r_corner = std::sqrt(cx * cx + cx * cx + cx * cx);
    double W0 = cfg.physics.W0;
    double inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * W0);
    double r0 = cfg.initial.seed_radius;
    double expected = -std::tanh((r_corner - r0) * inv_sqrt2_W0);

    EXPECT_NEAR(phi(0, 0, 0), expected, 1e-4) << "phi at corner should be ≈ " << expected;
    EXPECT_LT(phi(0, 0, 0), -0.999) << "phi at corner should be ≈ -1 (liquid)";
}

/// Verify phi at the interface (r ≈ r0): phi ≈ -tanh(0) = 0.
/// Find a point at distance ~4 from center. For (8+4, 8, 8) = (12, 8, 8),
/// r = 4, so phi = -tanh(0) = 0.
TEST_F(InitialConditionTest, PhiAtInterface) {
    auto cfg = make_config();
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    // (12, 8, 8) is at distance 4 from center (8,8,8)
    EXPECT_NEAR(phi(12, 8, 8), 0.0, 0.05) << "phi at r=r0 should be ≈ 0 (interface)";
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
