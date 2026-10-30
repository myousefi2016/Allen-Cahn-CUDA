#pragma once

#include "core/Interval.hpp"
#include "core/LiquidModes.hpp"

// The pieces of the certified liquid-mode search (LiquidModes.cpp) that its proof rests
// on, exposed so that tests/unit/test_LiquidModes.cpp can check each one against
// binary128 references: the domain it searches, the coefficients phi_k(n) with their
// derivatives, each upper bound of F over a box (also for coefficients anywhere in given
// intervals), the test that drops a box, the enclosure of F at a mode, and the branch and
// bound without the seed search that normally starts it.

namespace ac::liquid_modes_detail {

/// The function the search maximises.
enum class Objective {
    Growth,  ///< F = (Lam - beta L) / A(n)^2
    Damping, ///< F = -(Lam + beta L)
};

/// The wave-number box the search covers: theta_x in [0, 3.2] (F is even in theta),
/// theta_y and theta_z in [-3.2, 3.2] (F is 2 pi periodic, 3.2 > pi).
inline constexpr double kThetaLo[3] = {0.0, -3.2, -3.2};
inline constexpr double kThetaHi[3] = {3.2, 3.2, 3.2};

/// phi_k(n(a, b)), k = 0..6 (LiquidModes.cpp), with their first and second (a, b)
/// derivatives enclosed over a in `a` and b in `b`, for n the direction of v = (1, a, b)
/// (face 0), (a, 1, b) (face 1) or (a, b, 1) (face 2).
void phi(int face, interval::Iv a, interval::Iv b, double epsilon, Objective objective,
         interval::D2 out[7]);

/// The three upper bounds of F the search takes the smallest of, over the box
/// n = v / |v| with v on face `face` and (a, b) in [a0, a1] x [b0, b1], theta in
/// [theta_lo, theta_hi] (within [-3.2, 3.2]^3); each is an upper bound by itself.
/// no_stationary_point: the search drops the box (its theta gradient excludes zero in
/// some component).
struct BoxBounds {
    double joint;              ///< phi over the n-box, per-axis decomposition in theta
    double second_order_n;     ///< F(n_c, theta) over the theta box + remainder in n
    double second_order_theta; ///< Taylor bound in theta at n_c + remainder in n
    bool no_stationary_point;
};
[[nodiscard]] BoxBounds box_bounds(Objective objective, double epsilon, double beta, double W0,
                                   const double h[3], int face, double a0, double a1, double b0,
                                   double b1, const double theta_lo[3], const double theta_hi[3]);

/// The bounds of sum_k phi_k psi_k(theta) over the theta box [theta_lo, theta_hi] that
/// hold for every phi in the intervals `phi` (the search passes phi's range over an n-box,
/// or its enclosure at the box's centre): the per-axis decomposition, the interval value,
/// and the second-order Taylor bound around the box's centre.
struct ThetaBounds {
    double decomposed, interval_value, taylor;
};
[[nodiscard]] ThetaBounds theta_bounds(Objective objective, double epsilon, double beta, double W0,
                                       const double h[3], const interval::Iv phi[7],
                                       const double theta_lo[3], const double theta_hi[3]);

/// Enclosure of F at the mode (the direction of n, theta): the search records the lower
/// end of it.
[[nodiscard]] interval::Iv mode_enclosure(Objective objective, double epsilon, double beta,
                                          double W0, const double h[3], const double n[3],
                                          const double theta[3]);

/// The branch and bound started from the mode `start` (a nonzero direction, finite wave
/// numbers) alone instead of the seed search's best mode, with `budget` box evaluations.
/// Results and exceptions as liquid_gradient_growth (Growth) and liquid_gradient_damping
/// (Damping).
[[nodiscard]] LiquidModeBound search_from(Objective objective, double epsilon, double beta,
                                          double W0, const double h[3], const LiquidMode& start,
                                          long budget = kLiquidModeEvaluationBudget);

} // namespace ac::liquid_modes_detail
