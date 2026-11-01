#pragma once

#include "core/LiquidModes.hpp"
#include "core/SimulationConfig.hpp"

#include <optional>
#include <string>

namespace ac {

/// Explicit stability limits of the time step (docs/THEORY.md 10.1). The
/// thermal one is exact (every accepted boundary condition); the phase-field
/// one is a necessary condition from the modes of the liquid listed below.

/// Real stability interval [-r, 0] of the explicit scheme's update
/// y_{n+1} = R(dt lambda) y_n: r = 2 for Euler and Heun (R = 1 + z,
/// 1 + z + z^2/2), r = 2.785293563405282 for RK4 (the real root of
/// z^3 + 4z^2 + 12z + 24, where R(z) = 1 + z + ... + z^4/24 returns to 1).
/// IMEX advances phi like Euler.
[[nodiscard]] double explicit_stability_interval(TimeScheme scheme);

/// Spectral radius of the thermal Laplacian for every accepted boundary
/// condition: 4 sum_d h_d^-2 (7-point), 184 / (30 h^2) (27-point).
[[nodiscard]] double thermal_spectral_radius(const SimulationConfig& config);

/// Largest dt at which the explicit thermal update (Euler, Heun, RK4) is
/// stable: r / (D rho). +infinity for IMEX (u implicit).
[[nodiscard]] double thermal_dt_limit(const SimulationConfig& config);

/// The largest dt up to which none of the modes of the liquid phi = -1 listed
/// below grows under the explicit phase-field update (docs/THEORY.md 10.1), or
/// a value below it by at most the accuracy stated below: a necessary
/// condition for stability, not a sufficient one. A mode with damping
/// Lam + beta L >= 0, Willmore operator L and relaxation time tau does not
/// grow while dt (Lam - beta L) <= r tau (r of the scheme; beta = 0 for Heun
/// and RK4). The result is the smallest r tau / (Lam - beta L) over the modes
/// with Lam - beta L > 0 among
///  * the exact modes of the uniform liquid: the uniform shift (Lam = 2,
///    L = 4) and the checkerboards (-1)^(sum_{d in S} x_d) over a subset S of
///    the axes (Lam = W0^2 (1 + eps)^2 mu_S + 2, L = (W0^2 mu_S + 2)^2,
///    mu_S = 4 sum_{d in S} h_d^-2), all with tau = tau0 (1 - 3 eps/5)^2 (the
///    zero-gradient value the kernel uses); and the modes that vary along one
///    axis a with any wave number and alternate along a subset of the other
///    two (mu between mu_S' and mu_S' + 4/h_a^2, the same Lam and L, tau =
///    tau0 (1 + eps)^2 from their axial cell gradients), maximised in closed
///    form in double precision;
///  * every Fourier mode of a liquid with a small smooth gradient along any
///    unit direction n (LiquidModes.hpp; tau = tau0 A(n)^2), through a
///    certified upper bound B of M = max (Lam - beta L) / A(n)^2: B - M is at
///    most kLiquidModeTolerance * max(1, M) when the search ends within its
///    budget (it does for the shipped and the tested physics), else B is the
///    upper end of the range it reached (LiquidModeSearchExhausted); r tau0 / B
///    is then below the exact frozen-coefficient limit by a relative
///    (B - M) / B at most. At eps = 0 these modes are the uniform liquid's
///    (H(n) = I, A(n) = 1) and add nothing.
/// +infinity when no uniform mode has Lam - beta L > 0 and B is not positive.
[[nodiscard]] double phase_field_dt_limit(const SimulationConfig& config);

/// A mode of a liquid with a small gradient that has Lam + beta L < 0 and so
/// grows at every dt (the gradient energy is not convex for eps > 1/15 and
/// beta does not regularise it at this spacing; docs/THEORY.md 10.1), or no
/// value when every such mode is proven to have Lam + beta L >= 0. The
/// uniform liquid always has Lam + beta L >= 2. Throws
/// LiquidModeSearchExhausted when the search cannot decide within its budget
/// (beta close to the smallest value that regularises eps at this spacing).
[[nodiscard]] std::optional<LiquidModeBound>
phase_field_unstable_mode(const SimulationConfig& config);

/// The lines `allen-cahn-cuda --validate-config` prints for a valid
/// configuration: the thermal limit with the scheme and stencil it is
/// computed for, the phase-field limit with the scheme (`stencil` selects
/// only the thermal Laplacian), and with time.adaptive the cap
/// cfl_safety * min(thermal, phase field) of every adaptive step. A finite
/// value is printed to 17 significant digits, as in validate()'s messages,
/// so it reads back as exactly the limit validate() compares with; +infinity
/// prints as "none".
[[nodiscard]] std::string stability_limits_report(const SimulationConfig& config);

} // namespace ac
