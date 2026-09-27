#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace ac;

class GridValidationTest : public ::testing::Test {
protected:
    void SetUp() override { Logger::init(spdlog::level::off); }

    /// Return a valid base config that passes validation.
    SimulationConfig base_config() {
        SimulationConfig cfg;
        cfg.grid.Nx = 16;
        cfg.grid.Ny = 16;
        cfg.grid.Nz = 16;
        cfg.grid.dx = 0.4;
        cfg.grid.dy = 0.4;
        cfg.grid.dz = 0.4;
        return cfg;
    }
};

/// 27-point stencil with dx != dy should throw (requires cubic spacing).
TEST_F(GridValidationTest, Stencil27ptAnisotropicThrows) {
    auto cfg = base_config();
    cfg.stencil = StencilType::Isotropic27Point;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.5; // != dx
    cfg.grid.dz = 0.4;

    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

/// 27-point stencil with dx == dy == dz should NOT throw.
TEST_F(GridValidationTest, Stencil27ptIsotropicPasses) {
    auto cfg = base_config();
    cfg.stencil = StencilType::Isotropic27Point;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.4;

    EXPECT_NO_THROW(cfg.validate());
}

/// 7-point stencil with dx != dy should NOT throw (handles anisotropic grids).
TEST_F(GridValidationTest, Stencil7ptAnisotropicPasses) {
    auto cfg = base_config();
    cfg.stencil = StencilType::Standard7Point;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.5; // != dx
    cfg.grid.dz = 0.6; // != dx, dy

    EXPECT_NO_THROW(cfg.validate());
}
