#include "core/StabilityLimits.hpp"

#include <limits>

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

double phase_field_dt_limit(const SimulationConfig& config) {
    const GridParams& g = config.grid;
    const PhysicsParams& p = config.physics;
    const double rho7 = 4.0 * (1.0 / (g.dx * g.dx) + 1.0 / (g.dy * g.dy) + 1.0 / (g.dz * g.dz));
    const double w2 = p.W0 * p.W0;
    const double a_axis = 1.0 + p.epsilon;
    const double a_iso = 1.0 - 3.0 * p.epsilon / 5.0;
    const double lam = w2 * a_axis * a_axis * rho7 + 2.0;
    const double tau_cb = p.tau0() * a_iso * a_iso;
    const double b = w2 * rho7 + 2.0;
    const double damping = p.willmore_beta * b * b;
    if (damping >= lam)
        return std::numeric_limits<double>::infinity();
    return explicit_stability_interval(config.time.scheme) * tau_cb / (lam - damping);
}

} // namespace ac
