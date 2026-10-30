// Explicit stability limits (docs/THEORY.md 10.1): their closed forms and
// the validation built on them. test_StabilityLimitsGpu.cu checks on the GPU
// that the solver's update has exactly these limits.
#include "common/LiquidModeReference.hpp"
#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>

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
}

// The phase-field limit takes the smallest of the uniform liquid's exact modes
// and the liquid-gradient bound. Without the Willmore term the <111> gradient's
// full checkerboard binds (2 tau0 A^2 / (3 W0^2 H 4/h^2 + 2), at eps = 0 the
// uniform liquid's too); with it the binding mode is a wave between the
// checkerboards: at eps 0.07, beta 0.0149 (default.json) the gradient along
// <101> with theta = (0.652, 0, -0.652) pi, limit 2 tau0 / 41.19 = 0.027467,
// 13 % below what the checkerboards gave (0.031570).
TEST(StabilityLimits, PhaseFieldLimitClosedForms) {
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point); // h = 0.4
    const double tau0 = cfg.physics.tau0(), rho7 = 75.0;
    // The certified bound is within kLiquidModeTolerance of the maximum.
    const double tol = 2 * kLiquidModeTolerance;
    cfg.physics.epsilon = 0.0;
    EXPECT_NEAR(phase_field_dt_limit(cfg) / (2 * tau0 / (rho7 + 2)), 1.0, tol);
    for (double eps : {0.05, 1.0 / 15.0}) {
        cfg.physics.epsilon = eps;
        const double a = 1 - 5 * eps / 3, h111 = a * a + 64 * eps * a / 9;
        const double at111 = 2 * tau0 * a * a / (h111 * rho7 + 2);
        const double uniform =
            2 * tau0 * (1 - 0.6 * eps) * (1 - 0.6 * eps) / ((1 + eps) * (1 + eps) * rho7 + 2);
        EXPECT_LT(at111, uniform);
        EXPECT_LE(phase_field_dt_limit(cfg), at111) << "eps " << eps;
        EXPECT_NEAR(phase_field_dt_limit(cfg) / at111, 1.0, tol) << "eps " << eps;
    }
    cfg.physics.epsilon = 0.07;
    cfg.physics.willmore_beta = 0.0149;
    EXPECT_NEAR(phase_field_dt_limit(cfg) / (2 * tau0 / 41.189758899), 1.0, tol);
    EXPECT_NEAR(phase_field_dt_limit(cfg), 0.027467, 1e-6);
    // A large beta no longer damps every mode: the uniform shift alone has
    // Lam - beta L = 2 - 4 beta > 0 for beta < 1/2 (2 tau_cb / (2 - 4 beta)),
    // and the gradient modes bind below it (eps 0.07, beta 0.25: 2 tau0 / 2.0683).
    cfg.physics.willmore_beta = 0.25;
    const double tau_cb = tau0 * (1 - 0.6 * 0.07) * (1 - 0.6 * 0.07);
    EXPECT_LT(phase_field_dt_limit(cfg), 2 * tau_cb / (2 - 4 * 0.25));
    EXPECT_NEAR(phase_field_dt_limit(cfg) / (2 * tau0 / 2.06830220779), 1.0, tol);
    // Above beta = 1/2 every mode is damped at every dt.
    cfg.physics.willmore_beta = 0.6;
    EXPECT_EQ(phase_field_dt_limit(cfg), std::numeric_limits<double>::infinity());
    // At eps = 0 the gradient modes are the uniform liquid's (H = I, A = 1): with beta 0.1
    // Lam - beta L = W0^2 mu + 2 - beta (W0^2 mu + 2)^2 peaks at W0^2 mu + 2 = 1 / (2 beta),
    // at 1 / (4 beta) = 2.5 (mu = 3, inside [0, 75]): 2 tau0 / 2.5. (The branch and bound
    // would not end there: its maximum is a level set of mu.)
    cfg.physics.epsilon = 0.0;
    cfg.physics.willmore_beta = 0.1;
    EXPECT_NEAR(phase_field_dt_limit(cfg) / (2 * tau0 / 2.5), 1.0, 1e-15);
}

// A beta too small for eps > 1/15 at the spacing leaves liquid-gradient modes
// with Lam + beta L < 0, which grow at every dt: validate() rejects the
// configuration and names such a mode with its certified bound (the binary128
// value there, from the definitions, below it). The shipped physics have none.
TEST(StabilityLimits, UnderRegularisedPhysicsIsRejected) {
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point);
    cfg.time.dt = 1e-4;
    for (const auto& [eps, beta] : {std::pair{0.3, 0.1}, std::pair{0.12, 0.01}}) {
        cfg.physics.epsilon = eps;
        cfg.physics.willmore_beta = beta;
        const auto mode = phase_field_unstable_mode(cfg);
        ASSERT_TRUE(mode.has_value()) << eps << " " << beta;
        EXPECT_LT(mode->attained, 0.0);
        const test::Physics p{eps, beta, cfg.physics.W0, {0.4, 0.4, 0.4}};
        const test::Q value = test::damping(p, mode->mode.n, mode->mode.theta);
        EXPECT_TRUE(value <= test::Q(mode->attained))
            << test::qstr(value) << " > " << mode->attained;
        std::ostringstream bound;
        bound << std::setprecision(17) << "Lam + beta L <= " << mode->attained << " < 0";
        expect_invalid(cfg, bound.str());
    }
    for (const auto& [eps, beta] :
         {std::pair{0.05, 0.0}, std::pair{0.07, 0.0149}, std::pair{0.10, 0.153},
          std::pair{0.12, 0.25}, std::pair{0.15, 0.401}}) {
        cfg.physics.epsilon = eps;
        cfg.physics.willmore_beta = beta;
        EXPECT_FALSE(phase_field_unstable_mode(cfg).has_value()) << eps << " " << beta;
    }
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
    for (double bad : {0.0, -0.5, 1.0000001}) {
        cfg.time.cfl_safety = bad;
        expect_invalid(cfg, "cfl_safety must be in (0, 1]");
    }
    cfg.time.cfl_safety = std::numeric_limits<double>::quiet_NaN();
    expect_invalid(cfg, "time.cfl_safety = nan is not a finite number");
    cfg.time.cfl_safety = 1.0;
    EXPECT_NO_THROW(cfg.validate());
}

// A dt one ulp above its limit is reported with both numbers in 17
// significant digits, so they never print alike (with 6 decimals both showed
// as 0.013333).
TEST(StabilityLimits, RejectionMessageTellsDtAndLimitApart) {
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point);
    cfg.physics.d0 = 0.1;
    const double limit = thermal_dt_limit(cfg);
    cfg.time.dt = std::nextafter(limit, 1.0);
    std::ostringstream dt, lim;
    dt << std::setprecision(17) << cfg.time.dt;
    lim << std::setprecision(17) << limit;
    ASSERT_NE(dt.str(), lim.str());
    expect_invalid(cfg, "time.dt = " + dt.str() + " exceeds the stability limit " + lim.str());
}

// --validate-config prints this report: each limit as validate() applies it,
// printed to 17 significant digits, "none" for +infinity, and the adaptive
// cap cfl_safety times the smaller limit only with time.adaptive.
TEST(StabilityLimits, ReportPrintsTheLimitsAndTheAdaptiveCap) {
    auto digits17 = [](double v) {
        std::ostringstream os;
        os << std::setprecision(17) << v;
        return os.str();
    };
    auto cfg = config(TimeScheme::Euler, StencilType::Isotropic27Point);
    EXPECT_EQ(stability_limits_report(cfg),
              "thermal dt limit (scheme euler, stencil 27pt): " + digits17(thermal_dt_limit(cfg)) +
                  "\nphase-field dt limit (scheme euler, any stencil): " +
                  digits17(phase_field_dt_limit(cfg)) + "\n");
    cfg.time.scheme = TimeScheme::IMEX;
    cfg.time.adaptive = true;
    cfg.time.cfl_safety = 0.5;
    EXPECT_EQ(stability_limits_report(cfg),
              "thermal dt limit (scheme imex, stencil 27pt): none\n"
              "phase-field dt limit (scheme imex, any stencil): " +
                  digits17(phase_field_dt_limit(cfg)) +
                  "\nadaptive dt cap (cfl_safety times the smaller limit): " +
                  digits17(0.5 * phase_field_dt_limit(cfg)) + "\n");
    cfg.physics.epsilon = 0.12; // beta >= 1/2 damps every mode of the liquid
    cfg.physics.willmore_beta = 0.6;
    EXPECT_EQ(stability_limits_report(cfg),
              "thermal dt limit (scheme imex, stencil 27pt): none\n"
              "phase-field dt limit (scheme imex, any stencil): none\n"
              "adaptive dt cap (cfl_safety times the smaller limit): none\n");
}
