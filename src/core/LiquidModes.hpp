#pragma once

#include <stdexcept>
#include <string>

namespace ac {

/// Modes of a liquid phi = -1 whose small smooth gradient points along the
/// unit vector n (the tail of an interface; docs/THEORY.md 10.1). With the
/// coefficients frozen at n, every Fourier mode exp(i theta . x) of a
/// perturbation is an eigenmode of the explicit phase-field update: the face
/// fluxes respond with W0^2 H(n), H(n) the Hessian of the gradient energy per
/// W0^2 (gradient_energy_hessian), the cell gradients keep their direction
/// (tau = tau0 A(n)^2), and
///   sigma_n(theta) = sum_d H_dd(n) 2 (1 - cos theta_d) / h_d^2
///                  + sum_{d != e} H_de(n) sin theta_d sin theta_e / (h_d h_e),
///   mu(theta)      = sum_d 2 (1 - cos theta_d) / h_d^2,
///   Lam = W0^2 sigma + 2,  L = (W0^2 mu + 2)^2.
/// One step multiplies the mode by g = 1 - dt (Lam + beta L) / (tau + dt beta L).
struct LiquidMode {
    double n[3];     ///< unit gradient direction
    double theta[3]; ///< wave numbers, radians per cell
};

/// A certified bound of an extremum over every (n, theta), and a mode with a
/// certified value on the other side of the bound (`attained`; for a maximum
/// attained <= value at `mode` <= maximum <= bound). Every value is enclosed in
/// interval arithmetic rounded outward, for the given double parameters; a mode
/// is the direction of the double vector n with the double wave numbers theta.
struct LiquidModeBound {
    double bound;
    double attained;
    LiquidMode mode;
};

/// Accuracy of a maximum's LiquidModeBound: the search stops once no mode can
/// exceed `attained` by more than kLiquidModeTolerance * max(1, |attained|), and
/// bound is that limit rounded up, so the exact maximum lies in
/// [attained, bound].
inline constexpr double kLiquidModeTolerance = 1e-9;

/// H(n): the Hessian of 1/2 A(p)^2 |p|^2, the gradient energy
/// 1/2 W0^2 A(p)^2 |p|^2 per W0^2 (Kernels.cuh anisotropic_flux is W0^2 times
/// its gradient), at the unit vector n; the energy is 2-homogeneous, so H
/// depends on the direction only. With s = sum n_j^4, A = 1 - 3 eps + 4 eps s,
/// D_i = n_i (n_i^2 - s):
///   H_ii = A^2 + 32 eps A D_i n_i + 16 eps (16 eps D_i^2 + A (3 n_i^2 - 6 n_i^4 - s + 4 s n_i^2)),
///   H_ij = 256 eps^2 D_i D_j + 32 eps A n_i n_j (s - n_i^2 - n_j^2)   (i != j).
void gradient_energy_hessian(const double n[3], double epsilon, double H[3][3]);

/// Box evaluations a search may spend (a resource guard, not a tolerance; see
/// LiquidModes.cpp for the counts the shipped and tested physics need).
inline constexpr long kLiquidModeEvaluationBudget = 20'000'000;

/// Thrown when a search spends its budget, or meets a box it can no longer
/// split in double precision, without a certificate: the extremum it was after
/// is then only known to lie in [lower, upper] (the growth maximum of
/// (Lam - beta L) / A^2, or the minimum of Lam + beta L), and `mode` is the
/// best mode recorded (its certified value is the end of the range on its
/// side: lower for growth, upper for damping).
struct LiquidModeSearchExhausted : std::runtime_error {
    LiquidModeSearchExhausted(const std::string& what, double lower_, double upper_,
                              const LiquidMode& mode_)
        : std::runtime_error(what), lower(lower_), upper(upper_), mode(mode_) {}
    double lower, upper;
    LiquidMode mode;
};

/// Growth: max over (n, theta) of (Lam - beta L) / A(n)^2. A mode with this
/// value positive is stable only for dt <= r tau0 / value (r of the scheme).
/// bound is an upper bound of the maximum, attained a lower bound of the value
/// at `mode`. Throws LiquidModeSearchExhausted (above).
[[nodiscard]] LiquidModeBound liquid_gradient_growth(double epsilon, double beta, double W0,
                                                     const double h[3],
                                                     long budget = kLiquidModeEvaluationBudget);

/// Damping: the sign of min over (n, theta) of Lam + beta L, negative where
/// the gradient energy is not convex (epsilon > 1/15) and beta does not
/// regularise it: such a mode grows at every dt. Either bound = 0: every mode
/// is proven to have Lam + beta L >= 0 (attained: an upper bound of the value
/// at `mode`, the smallest the search recorded); or bound = attained < 0: an
/// upper bound of Lam + beta L at `mode`, a mode that has it negative (not
/// necessarily the minimum). Throws LiquidModeSearchExhausted when the minimum
/// is too close to 0 to decide within the budget.
[[nodiscard]] LiquidModeBound liquid_gradient_damping(double epsilon, double beta, double W0,
                                                      const double h[3],
                                                      long budget = kLiquidModeEvaluationBudget);

} // namespace ac
