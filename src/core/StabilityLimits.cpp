#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace ac {

double explicit_stability_interval(TimeScheme scheme) {
    return scheme == TimeScheme::RK4 ? 2.785293563405282 : 2.0;
}

double thermal_spectral_radius(const SimulationConfig& config) {
    const GridParams& g = config.grid;
    if (config.stencil == StencilType::Isotropic27Point)
        return 184.0 / (30.0 * g.dx * g.dx);
    return 4.0 * (1.0 / (g.dx * g.dx) + 1.0 / (g.dy * g.dy) + 1.0 / (g.dz * g.dz));
}

double thermal_dt_limit(const SimulationConfig& config) {
    if (config.time.scheme == TimeScheme::IMEX)
        return std::numeric_limits<double>::infinity();
    return explicit_stability_interval(config.time.scheme) /
           (config.physics.D * thermal_spectral_radius(config));
}

double gradient_energy_hessian_diagonal(const double n[3], double epsilon, int i) {
    const double s =
        n[0] * n[0] * n[0] * n[0] + n[1] * n[1] * n[1] * n[1] + n[2] * n[2] * n[2] * n[2];
    const double a = 1.0 - 3.0 * epsilon + 4.0 * epsilon * s;
    const double ni = n[i], ni2 = ni * ni;
    const double d = ni2 * ni - s * ni;
    const double dd = 3.0 * ni2 - 6.0 * ni2 * ni2 - s + 4.0 * s * ni2;
    return a * a + 32.0 * epsilon * a * d * ni + 16.0 * epsilon * (16.0 * epsilon * d * d + a * dd);
}

namespace {

/// The phase-field limit of one axis subset (bit d of `axes` set: axis d in
/// S) for a liquid with gradient direction n, given its Willmore damping.
double subset_limit_for_direction(const double n[3], int axes, const double inv_h2[3],
                                  double epsilon, double w2, double tau0, double r,
                                  double damping) {
    double sum = 0.0;
    for (int d = 0; d < 3; ++d)
        if ((axes >> d) & 1)
            sum += gradient_energy_hessian_diagonal(n, epsilon, d) * 4.0 * inv_h2[d];
    const double lam = w2 * sum + 2.0;
    if (!(lam > damping))
        return std::numeric_limits<double>::infinity();
    const double s =
        n[0] * n[0] * n[0] * n[0] + n[1] * n[1] * n[1] * n[1] + n[2] * n[2] * n[2] * n[2];
    const double a = 1.0 - 3.0 * epsilon + 4.0 * epsilon * s;
    return r * tau0 * a * a / (lam - damping);
}

} // namespace

double phase_field_dt_limit(const SimulationConfig& config) {
    const GridParams& g = config.grid;
    const PhysicsParams& p = config.physics;
    const double inv_h2[3] = {1.0 / (g.dx * g.dx), 1.0 / (g.dy * g.dy), 1.0 / (g.dz * g.dz)};
    const double w2 = p.W0 * p.W0;
    const double r = explicit_stability_interval(config.time.scheme);
    const double tau0 = p.tau0();
    const double a_iso = 1.0 - 3.0 * p.epsilon / 5.0;
    const double pi = 3.14159265358979323846;
    const double degree = pi / 180.0;
    double limit = std::numeric_limits<double>::infinity();
    for (int axes = 1; axes < 8; ++axes) {
        double mu = 0.0;
        for (int d = 0; d < 3; ++d)
            if ((axes >> d) & 1)
                mu += 4.0 * inv_h2[d];
        const double b = w2 * mu + 2.0;
        const double damping = p.willmore_beta * b * b;
        // The uniform liquid: axial face gradients (A = 1 + eps), zero cell
        // gradients (the kernel's spherical average 1 - 3 eps/5).
        const double lam_uniform = w2 * (1.0 + p.epsilon) * (1.0 + p.epsilon) * mu + 2.0;
        if (lam_uniform > damping)
            limit = std::min(limit, r * tau0 * a_iso * a_iso / (lam_uniform - damping));
        // A small smooth gradient along n (the octant suffices: H_dd and A
        // are even in every component).
        auto at = [&](double theta, double phi) {
            const double n[3] = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
                                 std::cos(theta)};
            return subset_limit_for_direction(n, axes, inv_h2, p.epsilon, w2, tau0, r, damping);
        };
        double best = std::numeric_limits<double>::infinity(), best_theta = 0.0, best_phi = 0.0;
        auto consider = [&](double theta, double phi) {
            const double v = at(theta, phi);
            if (v < best) {
                best = v;
                best_theta = theta;
                best_phi = phi;
            }
        };
        for (int it = 0; it <= 90; ++it)
            for (int ip = 0; ip <= 90; ++ip)
                consider(it * degree, ip * degree);
        consider(std::acos(1.0 / std::sqrt(3.0)), pi / 4.0); // <111>
        consider(pi / 2.0, pi / 4.0);                        // <110>
        consider(pi / 4.0, 0.0);                             // <101>
        consider(pi / 4.0, pi / 2.0);                        // <011>
        // Refine to the bottom of the best point's basin: move to the best of
        // the eight neighbours at the current step, or halve the step; stop
        // below 1e-8 rad, where the value of a smooth minimum changes by a
        // relative 1e-16, the double-precision rounding.
        if (std::isfinite(best)) {
            for (double step = degree; step > 1e-8;) {
                const double t0 = best_theta, p0 = best_phi;
                for (int i = -1; i <= 1; ++i)
                    for (int j = -1; j <= 1; ++j)
                        if (i != 0 || j != 0)
                            consider(t0 + i * step, p0 + j * step);
                if (best_theta == t0 && best_phi == p0)
                    step *= 0.5;
            }
        }
        limit = std::min(limit, best);
    }
    return limit;
}

namespace {

/// A limit with 17 significant digits (two different doubles never print
/// alike), or "none" for +infinity.
std::string limit_text(double limit) {
    if (limit == std::numeric_limits<double>::infinity())
        return "none";
    std::ostringstream os;
    os << std::setprecision(17) << limit;
    return os.str();
}

} // namespace

std::string stability_limits_report(const SimulationConfig& config) {
    const std::string scheme = config.time.scheme == TimeScheme::Euler  ? "euler"
                               : config.time.scheme == TimeScheme::Heun ? "heun"
                               : config.time.scheme == TimeScheme::RK4  ? "rk4"
                                                                        : "imex";
    const std::string stencil = config.stencil == StencilType::Isotropic27Point ? "27pt" : "7pt";
    const double thermal = thermal_dt_limit(config);
    const double phase_field = phase_field_dt_limit(config);
    std::string report = "thermal dt limit (scheme " + scheme + ", stencil " + stencil +
                         "): " + limit_text(thermal) + "\nphase-field dt limit (scheme " + scheme +
                         ", any stencil): " + limit_text(phase_field) + "\n";
    if (config.time.adaptive) {
        report += "adaptive dt cap (cfl_safety times the smaller limit): " +
                  limit_text(config.time.cfl_safety * std::min(thermal, phase_field)) + "\n";
    }
    return report;
}

} // namespace ac
