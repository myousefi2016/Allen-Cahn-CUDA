#pragma once

#include "core/SimulationConfig.hpp"

namespace ac {

/// Explicit stability limits of the time step (docs/THEORY.md 10.1). Both are
/// attained by a checkerboard mode: exactly on a periodic grid with an even
/// number of interior cells per axis, and approached as the grid grows
/// otherwise (a smaller grid's own limit is higher by O(1/N^2)).

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

/// Largest dt at which the explicit phase-field update does not amplify the
/// checkerboard perturbation of the liquid (phi = -1): there every face
/// gradient is axial (A = 1 + eps) and every cell gradient zero
/// (tau = tau0 (1 - 3 eps/5)^2), so one step multiplies the perturbation by
///   g = 1 - dt (Lam + beta Lcb) / (tau_cb + dt beta Lcb),
///   Lam = W0^2 (1 + eps)^2 rho7 + 2,  Lcb = (W0^2 rho7 + 2)^2,
/// rho7 = 4 sum_d h_d^-2 whatever the thermal stencil, and |g| <= 1 exactly
/// when dt (Lam - beta Lcb) <= 2 tau_cb (Heun and RK4, beta = 0: dt Lam <=
/// r tau_cb). +infinity when beta Lcb >= Lam (the Willmore term damps the
/// mode at every dt). A necessary condition: no closed form bounds the
/// anisotropic update at an interface.
[[nodiscard]] double phase_field_dt_limit(const SimulationConfig& config);

} // namespace ac
