#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

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

    /// validate()'s message, or "accepted".
    static std::string validation_error(const SimulationConfig& cfg) {
        try {
            cfg.validate();
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
        return "accepted";
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

/// The 27-point stencil takes dx for all three directions (laplacian_27pt and the IMEX
/// Jacobi sweep), so any other dy or dz is refused, one ulp included.
TEST_F(GridValidationTest, Stencil27ptRefusesASpacingOneUlpFromDx) {
    auto cfg = base_config();
    cfg.stencil = StencilType::Isotropic27Point;
    cfg.grid.dy = std::nextafter(0.4, 1.0);
    EXPECT_EQ(validation_error(cfg),
              "27-point stencil: dy = 0.40000000000000008 and dz = 0.40000000000000002 must "
              "equal dx = 0.40000000000000002: it uses dx for all three directions");

    cfg.grid.dy = 0.4;
    cfg.grid.dz = std::nextafter(0.4, 0.0);
    EXPECT_EQ(validation_error(cfg),
              "27-point stencil: dy = 0.40000000000000002 and dz = 0.39999999999999997 must "
              "equal dx = 0.40000000000000002: it uses dx for all three directions");
}

/// The rule does not depend on the length unit: base_config() with lengths scaled by
/// s = 2^-40 and times by s^2 (D unchanged; the scaling is exact in binary floating
/// point) is accepted with dx = dy = dz and refused with dy 2 % above dx, which an
/// absolute tolerance of 1e-14 accepted (dy - dx = 7.3e-15 here).
TEST_F(GridValidationTest, Stencil27ptRuleIsTheSameInEveryLengthUnit) {
    const double s = std::ldexp(1.0, -40);
    auto cfg = base_config();
    cfg.stencil = StencilType::Isotropic27Point;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4 * s;
    cfg.physics.W0 *= s;
    cfg.physics.d0 *= s;
    cfg.physics.beta0 *= s;
    cfg.time.dt *= s * s;
    cfg.time.dt_min *= s * s;
    cfg.time.dt_max *= s * s;
    EXPECT_EQ(validation_error(cfg), "accepted");

    cfg.grid.dy = cfg.grid.dx * 1.02;
    EXPECT_EQ(validation_error(cfg),
              "27-point stencil: dy = 3.7107383832335475e-13 and dz = 3.6379788070917132e-13 "
              "must equal dx = 3.6379788070917132e-13: it uses dx for all three directions");
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
