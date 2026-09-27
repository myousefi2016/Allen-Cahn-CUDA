#include "common/TempDir.hpp"
#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

using namespace ac;

class SimulationEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("test_sim_engine");
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

    SimulationConfig make_config(int N = 8, int max_steps = 10) {
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
        cfg.time.max_steps = max_steps;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.time.adaptive = false;
        cfg.time.dt_min = 1e-6;
        cfg.time.dt_max = 0.1;

        cfg.stencil = StencilType::Standard7Point;

        cfg.initial.seed_radius = 2.0;

        // Output to temp dir, frequency higher than max_steps to avoid writes
        cfg.output.output_dir = test_dir_ / "out";
        cfg.output.frequency = max_steps + 100;
        cfg.output.async_io = false;

        cfg.checkpoint.checkpoint_dir = test_dir_ / "ckpt";
        cfg.checkpoint.frequency = max_steps + 100;
        cfg.checkpoint.keep_last = 2;

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

TEST_F(SimulationEngineTest, Construction) {
    auto cfg = make_config(8, 10);
    EXPECT_NO_THROW({
        SimulationEngine engine(cfg);
        EXPECT_EQ(engine.grid().Nx(), 8);
        EXPECT_EQ(engine.grid().Ny(), 8);
        EXPECT_EQ(engine.grid().Nz(), 8);
    });
}

// The initial condition is the Karma-Rappel equilibrium profile
// phi = -tanh((r - r0) / (sqrt(2) W0)) in physical distance r from the domain
// midpoint, with u = -delta. run() with zero steps returns the fields after
// the solver applied the BCs, so interior cells must equal the profile and
// boundary cells must satisfy the configured Neumann/Dirichlet relations.
TEST_F(SimulationEngineTest, InitializeFieldsPattern) {
    const int N = 12;
    auto cfg = make_config(N, 5);
    cfg.initial.seed_radius = 1.5;
    cfg.time.max_steps = 0;
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    const auto& u = engine.u();
    const double c = 0.5 * (N - 1);
    const double h = cfg.grid.dx;
    const double inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * cfg.physics.W0);
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double rx = (x - c) * h, ry = (y - c) * h, rz = (z - c) * h;
                double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                double expected = -std::tanh((r - cfg.initial.seed_radius) * inv_sqrt2_W0);
                ASSERT_DOUBLE_EQ(phi(x, y, z), expected) << x << "," << y << "," << z;
            }
    // Zero-flux Neumann phi: every X-face cell copies its inner neighbour.
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            ASSERT_DOUBLE_EQ(phi(0, y, z), phi(1, y, z));
            ASSERT_DOUBLE_EQ(phi(N - 1, y, z), phi(N - 2, y, z));
        }
    // u = -delta everywhere, which also satisfies Dirichlet u = -0.8 = -delta.
    for (std::size_t i = 0; i < u.size(); ++i)
        ASSERT_DOUBLE_EQ(u.data()[i], -cfg.physics.delta);
}

TEST_F(SimulationEngineTest, RunShortSimulation) {
    auto cfg = make_config(8, 5);
    SimulationEngine engine(cfg);

    EXPECT_NO_THROW(engine.run());

    // Verify no NaN in output fields
    const auto& phi = engine.phi();
    const auto& u = engine.u();
    int N = 8;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_FALSE(std::isnan(phi(x, y, z)))
                    << "NaN in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isnan(u(x, y, z)))
                    << "NaN in u at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(phi(x, y, z)))
                    << "Inf in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(u(x, y, z)))
                    << "Inf in u at (" << x << "," << y << "," << z << ")";
            }

    // phi should remain bounded in [-1, 1] approximately
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_GE(phi(x, y, z), -2.0);
                EXPECT_LE(phi(x, y, z), 2.0);
            }
}

TEST_F(SimulationEngineTest, AdaptiveTimeStep) {
    auto cfg = make_config(8, 5);
    cfg.time.adaptive = true;
    cfg.time.adaptive_tolerance = 0.01;
    cfg.time.dt_min = 1e-6;
    cfg.time.dt_max = 0.05;
    cfg.time.dt = 0.001;

    SimulationEngine engine(cfg);
    EXPECT_NO_THROW(engine.run());

    // Verify the simulation completed without error and fields are valid
    const auto& phi = engine.phi();
    int N = 8;
    bool all_finite = true;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                if (std::isnan(phi(x, y, z)) || std::isinf(phi(x, y, z))) {
                    all_finite = false;
                }
            }
    EXPECT_TRUE(all_finite) << "Adaptive stepping produced non-finite values";
}

TEST_F(SimulationEngineTest, SaturationDetection) {
    auto cfg = make_config(8, 10);
    cfg.time.exit_on_saturation = true;
    cfg.time.saturation_threshold = -0.5;
    cfg.time.saturation_check_freq = 1;

    SimulationEngine engine(cfg);
    EXPECT_NO_THROW(engine.run());
}

TEST_F(SimulationEngineTest, ConfigValidationRejectsInvalid) {
    auto cfg = make_config(8, 10);
    cfg.output.format = "invalid_format";
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    auto cfg2 = make_config(8, 10);
    cfg2.checkpoint.keep_last = 0;
    EXPECT_THROW(cfg2.validate(), std::invalid_argument);

    auto cfg3 = make_config(8, 10);
    cfg3.initial.seed_radius = -1.0;
    EXPECT_THROW(cfg3.validate(), std::invalid_argument);
}
