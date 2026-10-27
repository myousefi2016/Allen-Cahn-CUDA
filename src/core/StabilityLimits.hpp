#pragma once

#include "core/SimulationConfig.hpp"

#include <string>

namespace ac {

/// Explicit stability limits of the time step (docs/THEORY.md 10.1). Both are
/// set by checkerboard modes: exactly on a periodic grid with an even number
/// of interior cells per axis, and approached as the grid grows otherwise (a
/// smaller grid's own limit is higher by O(1/N^2)).

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

/// Diagonal entry H_ii / W0^2 of the Hessian of the gradient energy
/// f(p) = 1/2 W0^2 A(p)^2 |p|^2 (Kernels.cuh anisotropic_flux) at the unit
/// vector n; f is 2-homogeneous, so H depends on the direction only. With
/// s = sum n_j^4, A = 1 - 3 eps + 4 eps s, D_i = n_i^3 - s n_i:
///   H_ii / W0^2 = A^2 + 32 eps A D_i n_i
///                 + 16 eps (16 eps D_i^2 + A (3 n_i^2 - 6 n_i^4 - s + 4 s n_i^2)).
/// Along an axis it is (1 + eps)^2 for i on that axis and
/// (1 + eps)(1 - 15 eps) = S(eps) across it (docs/THEORY.md 4.3).
[[nodiscard]] double gradient_energy_hessian_diagonal(const double n[3], double epsilon, int i);

/// A largest dt above which the explicit phase-field update provably
/// amplifies a perturbation of the liquid (docs/THEORY.md 10.1): a necessary
/// condition for stability, not a sufficient one. The perturbations are the
/// checkerboards over a non-empty subset S of the axes, (-1)^(sum_{d in S} x_d),
/// of a liquid phi = -1 that is uniform or has a small smooth gradient along
/// a direction n. Each face normal to an axis d in S then sees an axial
/// perturbation of its gradient, the face flux changes by H_dd(n) times it,
/// every cell gradient is unchanged, so tau = tau0 A(n)^2, or
/// tau0 (1 - 3 eps/5)^2 with A = 1 + eps on the faces for the uniform liquid;
/// the Willmore operator (isotropic) has L_S = (W0^2 mu_S + 2)^2,
/// mu_S = 4 sum_{d in S} h_d^-2. One step multiplies the perturbation by
///   g = 1 - dt (Lam + beta L_S) / (tau + dt beta L_S),
///   Lam = W0^2 sum_{d in S} H_dd 4/h_d^2 + 2,
/// |g| <= 1 iff dt (Lam - beta L_S) <= r tau (r of the scheme; beta = 0 for
/// Heun and RK4). The result is the smallest such limit over the seven S,
/// the uniform liquid, and directions n on a 1-degree grid of the octant,
/// the exact <110> and <111>, each refined to the bottom of its basin. Every
/// evaluated case is a valid bound on its own (exact for the uniform liquid,
/// frozen-coefficient for a gradient), so is their minimum. +infinity when
/// beta L_S >= Lam in every case.
[[nodiscard]] double phase_field_dt_limit(const SimulationConfig& config);

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
