#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
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

namespace {

/// Results of the liquid-gradient searches per physics (epsilon, beta, W0 and the three
/// spacings, compared bit for bit: the key holds their bit patterns, so NaN, -0 and +0
/// are keys like any other): validate(), the engine's adaptive cap and the report ask for
/// the same physics repeatedly. A search that throws leaves no entry.
using PhysicsKey = std::array<std::uint64_t, 6>;
struct LiquidModeCache {
    std::mutex mutex;
    std::map<PhysicsKey, LiquidModeBound> growth, damping;
};

LiquidModeCache& liquid_mode_cache() {
    static LiquidModeCache cache;
    return cache;
}

PhysicsKey physics_key(const SimulationConfig& config) {
    const PhysicsParams& p = config.physics;
    const GridParams& g = config.grid;
    return {std::bit_cast<std::uint64_t>(p.epsilon), std::bit_cast<std::uint64_t>(p.willmore_beta),
            std::bit_cast<std::uint64_t>(p.W0),      std::bit_cast<std::uint64_t>(g.dx),
            std::bit_cast<std::uint64_t>(g.dy),      std::bit_cast<std::uint64_t>(g.dz)};
}

template <class Compute>
LiquidModeBound cached(std::map<PhysicsKey, LiquidModeBound>& map, const SimulationConfig& config,
                       Compute compute) {
    LiquidModeCache& cache = liquid_mode_cache();
    const auto key = physics_key(config);
    {
        const std::lock_guard<std::mutex> lock(cache.mutex);
        const auto it = map.find(key);
        if (it != map.end())
            return it->second;
    }
    const double h[3] = {config.grid.dx, config.grid.dy, config.grid.dz};
    const LiquidModeBound result =
        compute(config.physics.epsilon, config.physics.willmore_beta, config.physics.W0, h);
    const std::lock_guard<std::mutex> lock(cache.mutex);
    map.emplace(key, result);
    return result;
}

} // namespace

double phase_field_dt_limit(const SimulationConfig& config) {
    const GridParams& g = config.grid;
    const PhysicsParams& p = config.physics;
    const double inv_h2[3] = {1.0 / (g.dx * g.dx), 1.0 / (g.dy * g.dy), 1.0 / (g.dz * g.dz)};
    const double w2 = p.W0 * p.W0;
    const double r = explicit_stability_interval(config.time.scheme);
    const double tau0 = p.tau0();
    const double beta = p.willmore_beta;
    const double axial = (1.0 + p.epsilon) * (1.0 + p.epsilon); // A^2 on an axial gradient
    const double tau_zero = tau0 * (1.0 - 3.0 * p.epsilon / 5.0) * (1.0 - 3.0 * p.epsilon / 5.0);
    const double tau_axial = tau0 * axial;
    double limit = std::numeric_limits<double>::infinity();
    // Lam - beta L of the uniform liquid at a total wave number mu: concave in mu.
    auto uniform = [&](double mu) {
        const double b = w2 * mu + 2.0;
        return w2 * axial * mu + 2.0 - beta * b * b;
    };
    // The uniform shift and the checkerboards over every subset S of the axes.
    for (int axes = 0; axes < 8; ++axes) {
        double mu = 0.0;
        for (int d = 0; d < 3; ++d)
            if ((axes >> d) & 1)
                mu += 4.0 * inv_h2[d];
        const double f = uniform(mu);
        if (f > 0.0)
            limit = std::min(limit, r * tau_zero / f);
    }
    // One axis a with any wave number (mu_a in (0, 4/h_a^2)), alternating along the axes
    // of a subset S' of the other two: the maximum of the concave f over the interval.
    for (int a = 0; a < 3; ++a)
        for (int axes = 0; axes < 8; ++axes) {
            if ((axes >> a) & 1)
                continue;
            double mu0 = 0.0;
            for (int d = 0; d < 3; ++d)
                if ((axes >> d) & 1)
                    mu0 += 4.0 * inv_h2[d];
            const double mu1 = mu0 + 4.0 * inv_h2[a];
            double mu = mu1; // f increasing when beta = 0
            if (beta > 0.0)  // vertex of f: W0^2 mu + 2 = (1 + eps)^2 / (2 beta)
                mu = std::clamp((axial / (2.0 * beta) - 2.0) / w2, mu0, mu1);
            const double f = uniform(mu);
            if (f > 0.0)
                limit = std::min(limit, r * tau_axial / f);
        }
    // Every Fourier mode of a liquid with a small gradient along any direction. At eps = 0
    // they are modes of the uniform liquid (H(n) = I, A(n) = 1: Lam, L and tau depend on
    // mu alone), already maximised above over every mu in [0, 4 sum_d h_d^-2]; the search
    // would not end there, its maximum being a level set of mu when beta > 0. Should the
    // search spend its budget, the upper end of the range it reached is still an upper
    // bound, only a looser one.
    if (p.epsilon > 0.0) {
        const LiquidModeBound growth =
            cached(liquid_mode_cache().growth, config,
                   [](double e, double b, double w, const double* spacing) {
                       try {
                           return liquid_gradient_growth(e, b, w, spacing);
                       } catch (const LiquidModeSearchExhausted& x) {
                           return LiquidModeBound{x.upper, x.lower, x.mode};
                       }
                   });
        if (growth.bound > 0.0)
            limit = std::min(limit, r * tau0 / growth.bound);
    }
    return limit;
}

std::optional<LiquidModeBound> phase_field_unstable_mode(const SimulationConfig& config) {
    const LiquidModeBound damping = cached(liquid_mode_cache().damping, config,
                                           [](double e, double b, double w, const double* spacing) {
                                               return liquid_gradient_damping(e, b, w, spacing);
                                           });
    if (damping.bound < 0.0)
        return damping;
    return std::nullopt;
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
