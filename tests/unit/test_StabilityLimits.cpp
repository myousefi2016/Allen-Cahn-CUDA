// Explicit stability limits (docs/THEORY.md 10.1): their closed forms and
// the validation built on them. test_StabilityLimitsGpu.cu checks on the GPU
// that the solver's update has exactly these limits.
#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <string>

using namespace ac;

namespace {

/// The default physics (eps 0.05, D = 2, d0 = 0.5, W0 = 1) on a small grid.
SimulationConfig config(TimeScheme scheme, StencilType stencil, double dx = 0.4, double dy = 0.4,
                        double dz = 0.4) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 18;
    cfg.grid.dx = dx;
    cfg.grid.dy = dy;
    cfg.grid.dz = dz;
    cfg.physics.epsilon = 0.05;
    cfg.physics.D = 2.0;
    cfg.time.scheme = scheme;
    cfg.stencil = stencil;
    return cfg;
}

/// validate() must throw invalid_argument whose message contains `needle`.
void expect_invalid(const SimulationConfig& cfg, const std::string& needle) {
    try {
        cfg.validate();
        ADD_FAILURE() << "accepted dt " << cfg.time.dt << ", dt_min " << cfg.time.dt_min;
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

} // namespace

// ── Host: closed forms ─────────────────────────────────────────────────────

// RK4's interval is the double nearest the real root of z^3 + 4z^2 + 12z + 24
// (R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 = 1 there): the cubic changes sign
// between the neighbouring doubles (evaluated in long double).
TEST(StabilityLimits, RK4IntervalIsTheRootOfItsStabilityPolynomial) {
    const double r = explicit_stability_interval(TimeScheme::RK4);
    auto p = [](long double z) { return ((z + 4) * z + 12) * z + 24; };
    const long double below = p(-static_cast<long double>(std::nextafter(r, 0.0)));
    const long double above = p(-static_cast<long double>(std::nextafter(r, 10.0)));
    EXPECT_LT(below * above, 0.0L)
        << "p(-r-) = " << static_cast<double>(below) << ", p(-r+) = " << static_cast<double>(above);
    EXPECT_EQ(explicit_stability_interval(TimeScheme::Euler), 2.0);
    EXPECT_EQ(explicit_stability_interval(TimeScheme::Heun), 2.0);
    EXPECT_EQ(explicit_stability_interval(TimeScheme::IMEX), 2.0);
}

TEST(StabilityLimits, ClosedForms) {
    const auto aniso = config(TimeScheme::Euler, StencilType::Standard7Point, 0.4, 0.5, 0.3);
    const double rho7 = 4.0 * (1 / 0.16 + 1 / 0.25 + 1 / 0.09);
    EXPECT_DOUBLE_EQ(thermal_spectral_radius(aniso), rho7);
    const auto iso27 = config(TimeScheme::Euler, StencilType::Isotropic27Point);
    EXPECT_DOUBLE_EQ(thermal_spectral_radius(iso27), 184.0 / (30.0 * 0.16));
    EXPECT_DOUBLE_EQ(thermal_dt_limit(aniso), 2.0 / (2.0 * rho7));
    auto rk4 = iso27;
    rk4.time.scheme = TimeScheme::RK4;
    EXPECT_DOUBLE_EQ(thermal_dt_limit(rk4), 2.785293563405282 / (2.0 * 184.0 / (30.0 * 0.16)));
    auto imex = iso27;
    imex.time.scheme = TimeScheme::IMEX;
    EXPECT_EQ(thermal_dt_limit(imex), std::numeric_limits<double>::infinity());

    // Phase field: 2 tau0 (1 - 3eps/5)^2 / (W0^2 (1+eps)^2 rho7 + 2), and
    // with beta: the Willmore damping beta (W0^2 rho7 + 2)^2 enters.
    const double tau_cb = aniso.physics.tau0() * 0.97 * 0.97;
    const double lam = 1.05 * 1.05 * rho7 + 2.0, lcb = (rho7 + 2.0) * (rho7 + 2.0);
    EXPECT_DOUBLE_EQ(phase_field_dt_limit(aniso), 2.0 * tau_cb / lam);
    // beta Lcb = 0.0005 * 7645.8 = 3.8 < lam = 96.2: a finite, larger limit;
    // beta Lcb = 0.0149 * 7645.8 = 113.9 >= lam: damped at every dt.
    auto willmore = aniso;
    willmore.physics.willmore_beta = 0.0005;
    EXPECT_DOUBLE_EQ(phase_field_dt_limit(willmore), 2.0 * tau_cb / (lam - 0.0005 * lcb));
    EXPECT_GT(phase_field_dt_limit(willmore), phase_field_dt_limit(aniso));
    willmore.physics.willmore_beta = 0.0149;
    EXPECT_EQ(phase_field_dt_limit(willmore), std::numeric_limits<double>::infinity());
}

// A fixed dt is accepted up to the smaller limit and rejected above it, with
// the message naming the update that limits it; IMEX has no thermal limit.
TEST(StabilityLimits, ValidateRejectsAFixedDtAboveTheLimit) {
    for (TimeScheme scheme : {TimeScheme::Euler, TimeScheme::Heun, TimeScheme::RK4}) {
        for (StencilType stencil : {StencilType::Standard7Point, StencilType::Isotropic27Point}) {
            auto cfg = config(scheme, stencil);
            cfg.physics.d0 = 0.1; // phase-field limit far above the thermal one
            ASSERT_LT(thermal_dt_limit(cfg), phase_field_dt_limit(cfg));
            cfg.time.dt = thermal_dt_limit(cfg);
            EXPECT_NO_THROW(cfg.validate());
            cfg.time.dt = std::nextafter(thermal_dt_limit(cfg), 1.0);
            expect_invalid(cfg, "explicit thermal update");
        }
    }
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point); // d0 = 0.5
    ASSERT_LT(phase_field_dt_limit(cfg), thermal_dt_limit(cfg));
    cfg.time.dt = phase_field_dt_limit(cfg);
    EXPECT_NO_THROW(cfg.validate());
    cfg.time.dt = std::nextafter(phase_field_dt_limit(cfg), 1.0);
    expect_invalid(cfg, "explicit phase-field update");
    auto imex = config(TimeScheme::IMEX, StencilType::Standard7Point);
    imex.physics.d0 = 0.1;
    imex.time.dt = std::nextafter(
        thermal_dt_limit(config(TimeScheme::Euler, StencilType::Standard7Point)), 1.0);
    EXPECT_NO_THROW(imex.validate());
    imex.time.dt = std::nextafter(phase_field_dt_limit(imex), 1.0);
    expect_invalid(imex, "explicit phase-field update");
}

// Adaptive stepping: the first step uses time.dt and the others are capped at
// cfl_safety times the limit, so time.dt and dt_min must not exceed the cap;
// cfl_safety must be in (0, 1].
TEST(StabilityLimits, ValidateRejectsAdaptiveSettingsAboveTheCap) {
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point);
    cfg.time.adaptive = true;
    cfg.time.cfl_safety = 0.9;
    const double cap = 0.9 * std::min(thermal_dt_limit(cfg), phase_field_dt_limit(cfg));
    cfg.time.dt = cap;
    cfg.time.dt_min = 1e-6;
    EXPECT_NO_THROW(cfg.validate());
    cfg.time.dt = std::nextafter(cap, 1.0);
    expect_invalid(cfg, "time.dt (the first adaptive step)");
    cfg.time.dt = 1e-3;
    cfg.time.dt_min = std::nextafter(cap, 1.0);
    expect_invalid(cfg, "time.dt_min");
    cfg.time.dt_min = 1e-6;
    for (double bad : {0.0, -0.5, 1.0000001, std::numeric_limits<double>::quiet_NaN()}) {
        cfg.time.cfl_safety = bad;
        expect_invalid(cfg, "cfl_safety must be in (0, 1]");
    }
    cfg.time.cfl_safety = 1.0;
    EXPECT_NO_THROW(cfg.validate());
}
