#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace ac;

class SimulationConfigTest : public ::testing::Test {
protected:
    void SetUp() override { Logger::init(spdlog::level::off); }
};

TEST_F(SimulationConfigTest, DefaultValues) {
    SimulationConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.physics.delta, 0.8);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.07);
    EXPECT_DOUBLE_EQ(cfg.physics.W0, 1.0);
    EXPECT_DOUBLE_EQ(cfg.physics.D, 2.0);
    EXPECT_EQ(cfg.grid.Nx, 600);
    EXPECT_EQ(cfg.grid.Ny, 600);
    EXPECT_EQ(cfg.grid.Nz, 600);
    EXPECT_DOUBLE_EQ(cfg.time.dt, 0.01);
    EXPECT_EQ(cfg.time.max_steps, 6000);
}

TEST_F(SimulationConfigTest, DerivedQuantities) {
    PhysicsParams p;
    double expected_lambda = p.W0 * p.a1 / p.d0;
    double expected_tau0 =
        (p.W0 * p.W0 * p.W0 * p.a1 * p.a2) / (p.d0 * p.D) + (p.W0 * p.W0 * p.beta0) / p.d0;

    EXPECT_NEAR(p.lambda(), expected_lambda, 1e-12);
    EXPECT_NEAR(p.tau0(), expected_tau0, 1e-12);
}

TEST_F(SimulationConfigTest, ParseFromJsonString) {
    std::string json = R"({
        "physics": {
            "delta": 0.5,
            "epsilon": 0.05,
            "W0": 2.0,
            "D": 3.0
        },
        "grid": {
            "Nx": 64, "Ny": 64, "Nz": 64,
            "dx": 0.2, "dy": 0.2, "dz": 0.2
        },
        "time": {
            "dt": 0.005,
            "max_steps": 100,
            "scheme": "rk4",
            "adaptive": true
        },
        "stencil": "27pt",
        "output": {
            "frequency": 10,
            "output_dir": "/tmp/ac_test",
            "format": "raw"
        },
        "initial": {
            "seed_radius": 3.0
        },
        "boundary": {
            "phi": { "type": "dirichlet", "value": -1.0 },
            "u":   { "type": "neumann", "flux": 0.0 }
        }
    })";

    auto cfg = SimulationConfig::from_json_string(json);

    EXPECT_DOUBLE_EQ(cfg.physics.delta, 0.5);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.05);
    EXPECT_DOUBLE_EQ(cfg.physics.W0, 2.0);
    EXPECT_DOUBLE_EQ(cfg.physics.D, 3.0);
    EXPECT_EQ(cfg.grid.Nx, 64);
    EXPECT_DOUBLE_EQ(cfg.grid.dx, 0.2);
    EXPECT_DOUBLE_EQ(cfg.time.dt, 0.005);
    EXPECT_EQ(cfg.time.max_steps, 100);
    EXPECT_EQ(cfg.time.scheme, TimeScheme::RK4);
    EXPECT_TRUE(cfg.time.adaptive);
    EXPECT_EQ(cfg.stencil, StencilType::Isotropic27Point);
    EXPECT_EQ(cfg.output.frequency, 10);
    EXPECT_EQ(cfg.output.format, "raw");
    EXPECT_DOUBLE_EQ(cfg.initial.seed_radius, 3.0);
    EXPECT_EQ(cfg.boundary.phi_bc.type, BCType::Dirichlet);
    EXPECT_DOUBLE_EQ(cfg.boundary.phi_bc.value, -1.0);
    EXPECT_EQ(cfg.boundary.u_bc.type, BCType::Neumann);
}

TEST_F(SimulationConfigTest, ValidationPasses) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    EXPECT_NO_THROW(cfg.validate());
}

TEST_F(SimulationConfigTest, ValidationFailsSmallGrid) {
    SimulationConfig cfg;
    cfg.grid.Nx = 2;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, ValidationFailsNegativeSpacing) {
    SimulationConfig cfg;
    cfg.grid.dx = -1.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, ValidationFailsInvalidEpsilon) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    cfg.physics.epsilon = 0.5; // >= 1/3
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, ValidationFailsNegativeDt) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    cfg.time.dt = -0.01;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, MakeGrid) {
    SimulationConfig cfg;
    cfg.grid.Nx = 50;
    cfg.grid.Ny = 60;
    cfg.grid.Nz = 70;
    cfg.grid.dx = 0.3;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.5;
    auto grid = cfg.make_grid();
    EXPECT_EQ(grid.Nx(), 50);
    EXPECT_EQ(grid.Ny(), 60);
    EXPECT_EQ(grid.Nz(), 70);
    EXPECT_DOUBLE_EQ(grid.dx(), 0.3);
}

TEST_F(SimulationConfigTest, ParseTimeSchemes) {
    auto test_scheme = [](const std::string& scheme_str, TimeScheme expected) {
        std::string json = R"({"time": {"scheme": ")" + scheme_str + R"("}})";
        auto cfg = SimulationConfig::from_json_string(json);
        EXPECT_EQ(cfg.time.scheme, expected) << "Failed for scheme: " << scheme_str;
    };

    test_scheme("euler", TimeScheme::Euler);
    test_scheme("heun", TimeScheme::Heun);
    test_scheme("rk4", TimeScheme::RK4);
    test_scheme("imex", TimeScheme::IMEX);
}

TEST_F(SimulationConfigTest, ParseBCTypes) {
    auto test_bc = [](const std::string& bc_str, BCType expected) {
        std::string json = R"({"boundary": {"phi": {"type": ")" + bc_str + R"("}}})";
        auto cfg = SimulationConfig::from_json_string(json);
        EXPECT_EQ(cfg.boundary.phi_bc.type, expected) << "Failed for BC: " << bc_str;
    };

    test_bc("dirichlet", BCType::Dirichlet);
    test_bc("neumann", BCType::Neumann);
    test_bc("periodic", BCType::Periodic);
    test_bc("robin", BCType::Robin);
}

TEST_F(SimulationConfigTest, ParseGPUConfig) {
    std::string json = R"({
        "gpu": {
            "device_ids": [0, 1],
            "block_size": 512,
            "multi_gpu": true
        }
    })";
    auto cfg = SimulationConfig::from_json_string(json);
    EXPECT_EQ(cfg.gpu.device_ids.size(), 2u);
    EXPECT_EQ(cfg.gpu.device_ids[0], 0);
    EXPECT_EQ(cfg.gpu.device_ids[1], 1);
    EXPECT_EQ(cfg.gpu.block_size_1d, 512);
    EXPECT_TRUE(cfg.gpu.multi_gpu);
}
