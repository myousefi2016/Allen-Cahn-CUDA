#include "core/SimulationEngine.hpp"
#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>

using namespace ac;

class SimulationEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = std::filesystem::temp_directory_path() / "test_sim_engine";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

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

TEST_F(SimulationEngineTest, Construction)
{
    auto cfg = make_config(8, 10);
    EXPECT_NO_THROW({
        SimulationEngine engine(cfg);
        EXPECT_EQ(engine.grid().Nx(), 8);
        EXPECT_EQ(engine.grid().Ny(), 8);
        EXPECT_EQ(engine.grid().Nz(), 8);
    });
}

TEST_F(SimulationEngineTest, InitializeFieldsPattern)
{
    auto cfg = make_config(8, 5);
    SimulationEngine engine(cfg);

    // run() initializes fields then runs the loop
    // Instead, we run the engine and inspect the final state.
    // But we need to check the IC. Since run() overwrites, we run 0 steps.
    cfg.time.max_steps = 0;
    SimulationEngine engine0(cfg);
    engine0.run();

    const auto& phi = engine0.phi();
    int N = 8;
    double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
    double r0 = cfg.initial.seed_radius;

    // Center should be inside seed -> phi = 1.0
    double center_r = std::sqrt((N/2 - cx)*(N/2 - cx) +
                                (N/2 - cy)*(N/2 - cy) +
                                (N/2 - cz)*(N/2 - cz));
    if (center_r < r0) {
        EXPECT_DOUBLE_EQ(phi(N/2, N/2, N/2), 1.0);
    }

    // Far corner (0,0,0) should be outside seed -> phi = -1.0
    double corner_r = std::sqrt(cx*cx + cy*cy + cz*cz);
    if (corner_r >= r0) {
        EXPECT_DOUBLE_EQ(phi(0, 0, 0), -1.0);
    }

    // Verify u at the center is 0.0 (inside seed)
    const auto& u = engine0.u();
    if (center_r < r0) {
        EXPECT_DOUBLE_EQ(u(N/2, N/2, N/2), 0.0);
    }
}

TEST_F(SimulationEngineTest, RunShortSimulation)
{
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

TEST_F(SimulationEngineTest, AdaptiveTimeStep)
{
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
