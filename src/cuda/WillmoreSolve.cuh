#pragma once

#include "core/SimulationConfig.hpp"

#include <functional>
#include <vector>

namespace ac::cuda {

class CudaSolver;

/// One solver taking part in a Willmore step, with the local X planes it owns
/// that are interior planes of the global grid ([plane_begin, plane_end)) and
/// the global index of plane_begin.
struct WillmoreDomain {
    CudaSolver* solver = nullptr;
    int device_id = 0;
    int plane_begin = 0;
    int plane_end = 0;
    int global_plane_begin = 0;
};

/// Fields whose halo / periodic-ghost planes must be refreshed during a step.
enum class WillmoreBuffer {
    Omega,                  ///< omega (read by the right-hand side)
    Direction,              ///< CG direction p (read by lap p)
    DirectionLaplacian,     ///< lap p (read by lap^2 p)
    PreconditionerIterate,  ///< FP32 Chebyshev iterate (read by its Laplacian)
    PreconditionerLaplacian ///< FP32 Laplacian of that iterate (read by lap^2)
};

// ── Spectral bounds of the implicit Willmore operator ──────────────────────
//
// One Willmore step solves A dphi = rhs on the interior cells with
//   A = T/dt + beta L,   L = B^2,   B = W0^2 Lap_h - 2 I,   T = diag(tau),
// where Lap_h is the 7-point Laplacian of a field whose boundary cells hold
// the homogeneous form of phi's BCs. The preconditioner scales by
//   D = T/dt + beta d_L I,
// d_L being the interior diagonal of L. docs/THEORY.md section 4.4 derives:
//  * spec(Lap_h) lies in [lap_min, lap_max], the sums over the axes of
//    Gershgorin bounds of each axis' 1-D second-difference matrix (symmetric
//    for every BC rule, including Robin with any ghost coefficient c);
//  * spec(B) then lies in [W0^2 lap_min - 2, W0^2 lap_max - 2] and spec(L) in
//    [L_min, L_max] (the squares; L_min = 0 if that interval contains 0);
//  * with t = v.Tv / (dt v.v) in [tau_min/dt, tau_max/dt] and
//    l = v.Lv / v.v in [L_min, L_max], the Rayleigh quotient of D^-1 A is
//    g(t, l) = (t + beta l) / (t + beta d_L), monotone in t and in l, so
//    spec(D^-1 A) lies in [a, b] with a, b the extremes of g at the corners;
//  * kappa_A = (tau_max/dt + beta L_max) / (tau_min/dt + beta L_min) bounds
//    the condition number of A.
struct WillmoreSpectrum {
    double lap_min = 0.0, lap_max = 0.0; ///< bounds of spec(Lap_h)
    double L_min = 0.0, L_max = 0.0;     ///< bounds of spec(L)
    double d_L = 0.0;                    ///< diagonal constant of D (per unit beta)
    double a = 0.0, b = 0.0;             ///< spec(D^-1 A) lies in [a, b]
    double kappa_A = 0.0;                ///< bound of cond(A)
};

/// Ghost coefficient of the homogeneous form of a Robin face, as
/// apply_bc_face_kernel fills it: ghost = c * inner with
/// c = (beta/ds) / (alpha + beta/ds), from the coefficients of scaled_robin
/// (the same quotient, without under- or overflow). SimulationConfig::validate
/// accepts only alpha*beta >= 0, so c lies in [0, 1].
[[nodiscard]] double robin_ghost_coefficient(double alpha, double beta, double ds);

/// Bounds of the 1-D second-difference matrix (unit spacing) of n interior
/// cells whose two ghost cells follow `lo` and `hi` in homogeneous form:
/// Gershgorin intervals of its rows. Periodic faces must come in pairs.
void second_difference_bounds(int n, const BoundaryConfig& lo, const BoundaryConfig& hi,
                              double spacing, double& lower, double& upper);

/// Spectral bounds of the Willmore operator for the global grid and phi's
/// (effective, per-face) BCs, given dt and the range of tau over the interior
/// cells. Requires 0 < tau_min <= tau_max, beta > 0 and dt > 0.
[[nodiscard]] WillmoreSpectrum willmore_spectrum(const SimulationConfig& config,
                                                 const PerFaceBoundary& phi_faces, double dt,
                                                 double tau_min, double tau_max);

/// Chebyshev polynomial preconditioner M = p(D^-1 A) D^-1 of degree `degree`
/// on [a, b]: the preconditioned operator M A has its spectrum in
/// [1 - error_bound, 1 + error_bound], error_bound = 1 / T_degree(sigma),
/// sigma = (b + a) / (b - a). It is applied by the two-vector Chebyshev
/// iteration from zero:
///   x_1 = s0 D^-1 r,
///   x_{k+1} = x_k + step_a[k-1] (x_k - x_{k-1}) + step_b[k-1] D^-1 (r - A x_k),
/// returning x_degree (degree - 1 applications of A).
struct ChebyshevPlan {
    int degree = 1;
    double d_L = 0.0; ///< diagonal constant of D (WillmoreSpectrum::d_L)
    double s0 = 0.0;
    std::vector<double> step_a, step_b;
    double error_bound = 0.0;
    /// Outer iterations after which the stopping test ||r|| <= tol ||b|| is
    /// guaranteed in exact arithmetic (2 by construction, see chebyshev_plan).
    int guaranteed_outer_iterations = 0;
};

/// The plan for a relative residual tolerance `tol`: the lowest degree for
/// which two outer (flexible) CG iterations meet the stopping test in exact
/// arithmetic, i.e. 2 q^2 sqrt(kappa_A) <= tol with q the CG contraction
/// factor of the preconditioned condition number (1 + e) / (1 - e),
/// e = error_bound. One outer iteration would need e ~ tol / sqrt(kappa_A),
/// below the single-precision unit roundoff in which the polynomial is
/// applied; see docs/THEORY.md section 4.4 for why two minimises the work.
[[nodiscard]] ChebyshevPlan chebyshev_plan(const WillmoreSpectrum& spectrum, double tol);

struct WillmoreSolveStats {
    int iterations = 0;             ///< outer (flexible CG) iterations
    double relative_residual = 0.0; ///< ||r|| / ||b|| at exit
    int chebyshev_degree = 0;       ///< degree of the polynomial preconditioner
    WillmoreSpectrum spectrum;      ///< bounds used to build it
    double tau_min = 0.0, tau_max = 0.0;
};

/// Relative residual at which CG stops. The increment error is then below
/// cond(A) * 1e-10 of the increment itself (cond(A) is O(10^2) here): far
/// below the discretisation error of any step.
inline constexpr double kWillmoreCgTolerance = 1e-10;
/// A solve that has not met the tolerance after this many outer iterations
/// throws instead of iterating on. The number is a backstop, not derived
/// from a bound: the preconditioner is built so that two iterations suffice
/// in exact arithmetic, rounding in its single-precision application can add
/// iterations, and no a priori bound of that rounding is available; the
/// measured solves took two or three (docs/THEORY.md section 4.4). Any count
/// up to the cap is accepted; it is returned in
/// WillmoreSolveStats::iterations, which the engine does not log.
inline constexpr int kWillmoreCgMaxIterations = 1000;

/// z = M r in every domain (CudaSolver::CgVector::R -> Z), M the Chebyshev
/// preconditioner of `plan`, applied in single precision after scaling r by
/// the power of two 2^-e, e = ilogb(sqrt(rr)), rr = r.r > 0 over all domains:
/// the scaled r has norm sqrt(rr) 2^-e in [1, 2), so no component overflows
/// in single precision. The norm does not bound components from below: those
/// under 2^-126 (FLT_MIN) in magnitude are rounded to multiples of 2^-149, to
/// 0 at 2^-150 and below, an absolute error of at most 2^-150 each
/// (AC_CUDA_FAST_MATH builds flush subnormals to 0 instead). z is the same
/// bits for r and 2^m r (up to 2^m) as long as no r_c^2 of either underflows
/// in double precision.
/// Requires willmore_rhs_init() to have run in every domain (it computes
/// tau/dt in single precision). exchange() as for solve_willmore_step.
void apply_willmore_preconditioner(const std::vector<WillmoreDomain>& domains,
                                   const ChebyshevPlan& plan, double rr,
                                   const std::function<void(WillmoreBuffer)>& exchange);

/// One linearly implicit Willmore phase-field step (see WillmoreKernels.cuh):
/// solves (tau/dt + beta L) dphi = rhs over all domains by flexible
/// (Polak-Ribiere) conjugate gradients in double precision, preconditioned by
/// the Chebyshev polynomial of chebyshev_plan applied in single precision,
/// and sets phi_new = phi_old + dphi (with phi's BCs applied) in every
/// domain. `config` is the global configuration (grid and phi's BCs of the
/// whole domain); each solver's params must already hold dt.
/// exchange(buffer) must make that buffer's halo and periodic-ghost planes
/// current in every domain (a no-op for a single solver). Dot products are
/// summed per X plane and then over planes in global order, and the tau range
/// is an exact min/max, so the result is the same bits for any split of the
/// grid. Throws if CG breaks down or does not reach kWillmoreCgTolerance
/// within kWillmoreCgMaxIterations.
WillmoreSolveStats solve_willmore_step(const std::vector<WillmoreDomain>& domains,
                                       const SimulationConfig& config,
                                       const std::function<void(WillmoreBuffer)>& exchange);

} // namespace ac::cuda
