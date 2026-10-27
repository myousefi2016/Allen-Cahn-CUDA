// Explicit stability limits (docs/THEORY.md 10.1): their closed forms and
// the validation built on them. test_StabilityLimitsGpu.cu checks on the GPU
// that the solver's update has exactly these limits.
#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <sstream>
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
}

// The Hessian diagonal of the gradient energy in closed form: (1 + eps)^2
// along an axis, the stiffness S(eps) = (1 + eps)(1 - 15 eps) across it,
// A^2 + 64 eps A / 9 at <111> (A = 1 - 5 eps/3), and 1 for every direction
// at eps = 0; against a central difference of the flux that
// anisotropic_flux computes (the derivative of F_i along p_i).
TEST(StabilityLimits, GradientEnergyHessianDiagonal) {
    const double x[3] = {1, 0, 0};
    const double r3 = 1.0 / std::sqrt(3.0);
    const double d111[3] = {r3, r3, r3};
    const double odd[3] = {0.48, 0.6, std::sqrt(1.0 - 0.48 * 0.48 - 0.6 * 0.6)};
    for (double eps : {0.0, 0.05, 0.12, 0.3}) {
        EXPECT_DOUBLE_EQ(gradient_energy_hessian_diagonal(x, eps, 0), (1 + eps) * (1 + eps));
        EXPECT_DOUBLE_EQ(gradient_energy_hessian_diagonal(x, eps, 1), (1 + eps) * (1 - 15 * eps));
        // d111 is 1/sqrt(3) rounded: agreement to rounding (values of order 1).
        const double a = 1 - 5 * eps / 3;
        for (int i = 0; i < 3; ++i)
            EXPECT_NEAR(gradient_energy_hessian_diagonal(d111, eps, i), a * a + 64 * eps * a / 9,
                        1e-14);
        // F_i(p) / W0^2 = A^2 p_i + 16 eps A (p_i^3/|p|^2 - s p_i/|p|^4), as
        // anisotropic_flux; its derivative along p_i by a central difference
        // with step 1e-5 (truncation O(1e-10), rounding O(1e-11)).
        auto flux = [eps](const double p[3], int i) {
            const double r = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            const double s =
                p[0] * p[0] * p[0] * p[0] + p[1] * p[1] * p[1] * p[1] + p[2] * p[2] * p[2] * p[2];
            const double an = 1 - 3 * eps + 4 * eps * s / (r * r);
            return an * an * p[i] + 16 * eps * an * (p[i] * p[i] * p[i] / r - s * p[i] / (r * r));
        };
        for (int i = 0; i < 3; ++i) {
            double up[3] = {odd[0], odd[1], odd[2]}, dn[3] = {odd[0], odd[1], odd[2]};
            up[i] += 1e-5;
            dn[i] -= 1e-5;
            EXPECT_NEAR(gradient_energy_hessian_diagonal(odd, eps, i),
                        (flux(up, i) - flux(dn, i)) / 2e-5, 1e-8)
                << "eps " << eps << " i " << i;
        }
    }
}

// The phase-field limit (necessary condition): the uniform liquid's
// checkerboards over every subset of the axes, and the liquid with a small
// gradient. At eps = 0 nothing depends on the direction and the full
// checkerboard of the uniform liquid binds; for eps > 0 on a cubic grid
// without the Willmore term the <111> gradient binds (the minimum of
// tau0 A^2 / (3 W0^2 H 4/h^2 + 2) over the directions); with the Willmore
// term a 1-D or 2-D checkerboard binds, and a large beta damps every case.
TEST(StabilityLimits, PhaseFieldLimitClosedForms) {
    auto cfg = config(TimeScheme::Euler, StencilType::Standard7Point); // h = 0.4
    const double tau0 = cfg.physics.tau0(), rho7 = 75.0;
    // The limit is evaluated at directions computed with sin and cos; where it
    // is stationary that changes it only by rounding: agreement to 1e-13.
    cfg.physics.epsilon = 0.0;
    EXPECT_NEAR(phase_field_dt_limit(cfg) / (2 * tau0 / (rho7 + 2)), 1.0, 1e-13);
    for (double eps : {0.05, 1.0 / 15.0}) {
        cfg.physics.epsilon = eps;
        const double a = 1 - 5 * eps / 3, h111 = a * a + 64 * eps * a / 9;
        const double at111 = 2 * tau0 * a * a / (h111 * rho7 + 2);
        const double uniform =
            2 * tau0 * (1 - 0.6 * eps) * (1 - 0.6 * eps) / ((1 + eps) * (1 + eps) * rho7 + 2);
        EXPECT_LT(at111, uniform);
        EXPECT_NEAR(phase_field_dt_limit(cfg) / at111, 1.0, 1e-13) << "eps " << eps;
    }
    // eps 0.07, beta 0.0149 (default.json): the full checkerboard is damped
    // (beta Lcb = 88.3 > Lam = 87.9), the 1-D one of the uniform liquid is not:
    // 2 tau_cb / (W0^2 (1+eps)^2 25 + 2 - beta 27^2) = 0.0526; the gradient
    // cases lower the limit further (2-D, <110>).
    cfg.physics.epsilon = 0.07;
    cfg.physics.willmore_beta = 0.0149;
    const double tau_cb = tau0 * (1 - 0.6 * 0.07) * (1 - 0.6 * 0.07);
    const double uniform_1d = 2 * tau_cb / (1.07 * 1.07 * 25 + 2 - 0.0149 * 27 * 27);
    EXPECT_LT(phase_field_dt_limit(cfg), uniform_1d);
    EXPECT_GT(phase_field_dt_limit(cfg), 0.0);
    cfg.physics.willmore_beta = 0.25;
    EXPECT_EQ(phase_field_dt_limit(cfg), std::numeric_limits<double>::infinity());
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
    cfg.physics.epsilon = 0.12; // run_dendrite.json: beta damps every checkerboard case
    cfg.physics.willmore_beta = 0.25;
    EXPECT_EQ(stability_limits_report(cfg),
              "thermal dt limit (scheme imex, stencil 27pt): none\n"
              "phase-field dt limit (scheme imex, any stencil): none\n"
              "adaptive dt cap (cfl_safety times the smaller limit): none\n");
}
