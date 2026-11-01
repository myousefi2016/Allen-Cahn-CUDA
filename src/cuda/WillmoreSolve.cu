#include "core/KernelRanges.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/WillmoreSolve.cuh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ac::cuda {

namespace {

/// v with 17 significant digits (std::to_string prints 1e-9 as 0.000000).
std::string g17(double v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

/// Why the relaxation times of a Willmore step are not positive finite doubles. tau0 or
/// the range of tau0 A(n)^2 over the orientations (relaxation_time_range) when the
/// configuration is the cause, which validate() rejects; otherwise the state: tau leaves
/// that range only where compute_An's arithmetic overflows, at a central-difference
/// gradient with an infinite component or a component whose fourth power is beyond the
/// double range (about 1.16e77). A NaN gradient takes compute_An's zero-gradient branch
/// and gives a finite A.
std::string tau_range_failure(const SimulationConfig& config, double tau_min, double tau_max) {
    const std::string range = "Willmore solve: the relaxation times tau0 A(n)^2 of the interior "
                              "cells span [" +
                              g17(tau_min) + ", " + g17(tau_max) + "], not positive finite doubles";
    const double tau0 = config.physics.tau0();
    if (!(std::isfinite(tau0) && tau0 > 0.0))
        return range + ": tau0 = " + g17(tau0) + " is not finite and positive";
    const RelaxationTimeRange r = relaxation_time_range(tau0, config.physics.epsilon);
    if (!(r.hi <= std::numeric_limits<double>::max()) || !(r.lo > 0.0)) {
        return range + ": tau0 = " + g17(tau0) + " gives tau0 A(n)^2 in [" + g17(r.lo) + ", " +
               g17(r.hi) + "] over the orientations, outside the positive doubles";
    }
    return range + ": at some interior cell the central-difference gradient of phi has an "
                   "infinite component, or one whose fourth power is beyond the double range "
                   "(about 1.16e77), where A(n) is NaN or infinite: phi or its boundary values "
                   "are out of range, or the phase field diverged (a NaN phase field gives a "
                   "finite A(n))";
}

} // namespace

// ── Spectral bounds ────────────────────────────────────────────────────────

double robin_ghost_coefficient(double alpha, double beta, double ds) {
    // apply_bc_face_kernel: ghost = (gamma + beta * inner / ds) / (alpha +
    // beta/ds), gamma = 0 in the homogeneous form, with the coefficients of
    // scaled_robin as launch_bc_face passes them.
    BoundaryConfig bc;
    bc.alpha = alpha;
    bc.beta = beta;
    const RobinCoefficients r = scaled_robin(bc, ds);
    return (r.beta / ds) / (r.alpha + r.beta / ds);
}

namespace {

/// Contribution of one face's ghost cell to the row of the adjacent interior
/// cell of the 1-D second-difference matrix (unit spacing): a diagonal term
/// (ghost = c * own value) or, for a periodic face, an off-diagonal term to
/// the cell at the opposite end.
struct GhostTerm {
    double diagonal = 0.0;
    double to_opposite_end = 0.0;
};

GhostTerm ghost_term(const BoundaryConfig& bc, double spacing) {
    switch (bc.type) {
    case BCType::Dirichlet:
        return {0.0, 0.0};
    case BCType::Neumann:
        return {1.0, 0.0};
    case BCType::Robin:
        return {robin_ghost_coefficient(bc.alpha, bc.beta, spacing), 0.0};
    case BCType::Periodic:
        return {0.0, 1.0};
    }
    throw std::logic_error("ghost_term: unknown BC type");
}

} // namespace

void second_difference_bounds(int n, const BoundaryConfig& lo, const BoundaryConfig& hi,
                              double spacing, double& lower, double& upper) {
    if (n < 1)
        throw std::invalid_argument("second_difference_bounds: n must be >= 1");
    if ((lo.type == BCType::Periodic) != (hi.type == BCType::Periodic))
        throw std::invalid_argument("second_difference_bounds: periodic faces must be paired");
    // Row i (1-based interior index) of the matrix: -2 u_i + u_{i-1} + u_{i+1},
    // where u_0 and u_{n+1} are the ghosts. Each row's Gershgorin interval is
    // [center - radius, center + radius]; the matrix is symmetric for every BC
    // (Robin only changes the end diagonals, periodic couples the two ends in
    // both directions), so its eigenvalues lie in the union of the intervals.
    const GhostTerm g_lo = ghost_term(lo, spacing);
    const GhostTerm g_hi = ghost_term(hi, spacing);
    lower = std::numeric_limits<double>::infinity();
    upper = -std::numeric_limits<double>::infinity();
    auto add_row = [&](double center, double radius) {
        lower = std::min(lower, center - radius);
        upper = std::max(upper, center + radius);
    };
    if (n == 1) {
        // Both ghosts belong to the only cell; a periodic ghost is that cell.
        add_row(-2.0 + g_lo.diagonal + g_lo.to_opposite_end + g_hi.diagonal + g_hi.to_opposite_end,
                0.0);
    } else {
        // First row: right neighbour, and the low ghost (own value or, if
        // periodic, cell n; for n == 2 that is the right neighbour again).
        // The last row mirrors it. The two couplings of a row to the same
        // column add before the absolute value; all of them are >= 0 here
        // except Robin diagonals, which do not enter the radius.
        add_row(-2.0 + g_lo.diagonal, 1.0 + g_lo.to_opposite_end);
        add_row(-2.0 + g_hi.diagonal, 1.0 + g_hi.to_opposite_end);
        if (n >= 3)
            add_row(-2.0, 2.0);
    }
    const double inv_h2 = 1.0 / (spacing * spacing);
    lower *= inv_h2;
    upper *= inv_h2;
}

WillmoreSpectrum willmore_spectrum(const SimulationConfig& config, const PerFaceBoundary& phi_faces,
                                   double dt, double tau_min, double tau_max) {
    const double beta = config.physics.willmore_beta;
    if (!(beta > 0.0) || !(dt > 0.0))
        throw std::invalid_argument("willmore_spectrum: requires willmore_beta > 0 and dt > 0");
    if (!(tau_min > 0.0) || !(tau_max >= tau_min) || !std::isfinite(tau_max))
        throw std::runtime_error(tau_range_failure(config, tau_min, tau_max));
    if (!std::isfinite(tau_min / dt)) {
        throw std::runtime_error("Willmore solve: tau/dt is beyond the double range at dt = " +
                                 g17(dt) + " (tau from " + g17(tau_min) + ")");
    }
    const int n[3] = {config.grid.Nx - 2, config.grid.Ny - 2, config.grid.Nz - 2};
    const double h[3] = {config.grid.dx, config.grid.dy, config.grid.dz};

    WillmoreSpectrum s;
    for (int axis = 0; axis < 3; ++axis) {
        double lo = 0.0, hi = 0.0;
        second_difference_bounds(n[axis], phi_faces.get(axis, 0), phi_faces.get(axis, 1), h[axis],
                                 lo, hi);
        s.lap_min += lo;
        s.lap_max += hi;
    }
    const double W2 = config.physics.W0 * config.physics.W0;
    const double B_lo = W2 * s.lap_min - 2.0, B_hi = W2 * s.lap_max - 2.0;
    s.L_max = std::max(B_lo * B_lo, B_hi * B_hi);
    s.L_min = (B_lo <= 0.0 && B_hi >= 0.0) ? 0.0 : std::min(B_lo * B_lo, B_hi * B_hi);

    // Interior diagonal of L = W0^4 lap^2 - 4 W0^2 lap + 4 for the 7-point
    // lap: diag(lap) = -s1, diag(lap^2) = s1^2 + 2 sum_d 1/h_d^4 (a row of
    // lap^2 away from the boundary; any positive constant keeps D SPD, and the
    // bounds below are derived for this one). validate() checks the same double.
    s.d_L = willmore_interior_diagonal(config.physics.W0, h);

    auto g = [&](double tau, double l) {
        return (tau / dt + beta * l) / (tau / dt + beta * s.d_L);
    };
    s.a = std::min(g(tau_min, s.L_min), g(tau_max, s.L_min));
    s.b = std::max(g(tau_min, s.L_max), g(tau_max, s.L_max));
    s.kappa_A = (tau_max / dt + beta * s.L_max) / (tau_min / dt + beta * s.L_min);
    return s;
}

// ── Chebyshev plan ─────────────────────────────────────────────────────────

namespace {

/// CG contraction factor for the preconditioned condition number
/// (1 + e) / (1 - e): (sqrt(kappa) - 1) / (sqrt(kappa) + 1) = e / (1 + sqrt(1 - e^2)).
double cg_contraction(double e) {
    return e / (1.0 + std::sqrt((1.0 - e) * (1.0 + e)));
}

/// Bound of ||r|| / ||b|| after two outer iterations (exact arithmetic).
double two_iteration_bound(double e, double kappa_A) {
    const double q = cg_contraction(e);
    return 2.0 * q * q * std::sqrt(kappa_A);
}

} // namespace

ChebyshevPlan chebyshev_plan(const WillmoreSpectrum& spectrum, double tol) {
    const double a = spectrum.a, b = spectrum.b;
    if (!(a > 0.0) || !(b >= a) || !std::isfinite(b) || !(spectrum.kappa_A >= 1.0) ||
        !(tol > 0.0 && tol < 1.0))
        throw std::invalid_argument("chebyshev_plan: needs 0 < a <= b, kappa_A >= 1, 0 < tol < 1");
    ChebyshevPlan plan;
    plan.d_L = spectrum.d_L;
    plan.guaranteed_outer_iterations = 2;
    const double theta = 0.5 * (b + a);
    plan.s0 = 1.0 / theta;
    if (b == a) {
        // D^-1 A = a I: s0 D^-1 is its exact inverse.
        plan.degree = 1;
        plan.error_bound = 0.0;
        return plan;
    }
    const double delta = 0.5 * (b - a);
    // sigma = (b + a) / (b - a) = 1 + d, d = 2a / (b - a) computed without
    // cancellation; T_k(sigma) = cosh(k acosh(sigma)) with
    // acosh(1 + d) = log1p(d + sqrt(d (2 + d))).
    const double d = 2.0 * a / (b - a);
    const double sigma = 1.0 + d;
    const double mu = std::log1p(d + std::sqrt(d * (2.0 + d)));
    auto error_bound = [&](int k) { return 1.0 / std::cosh(static_cast<double>(k) * mu); };
    auto meets = [&](int k) {
        return two_iteration_bound(error_bound(k), spectrum.kappa_A) <= tol;
    };

    // Closed form first: 2 q^2 sqrt(kappa_A) <= tol needs q <= q* =
    // sqrt(tol / (2 sqrt(kappa_A))), i.e. e <= e* = 2 q* / (1 + q*^2), i.e.
    // k >= acosh(1 / e*) / mu. Then settle rounding at the boundary exactly.
    const double q_star = std::sqrt(tol / (2.0 * std::sqrt(spectrum.kappa_A)));
    const double e_star = 2.0 * q_star / (1.0 + q_star * q_star);
    const double k_real = std::acosh(1.0 / e_star) / mu;
    if (!(k_real < static_cast<double>(std::numeric_limits<int>::max() / 2)))
        throw std::runtime_error("chebyshev_plan: the Willmore operator is too ill-conditioned "
                                 "(b/a = " +
                                 g17(b / a) + ") for a polynomial preconditioner");
    int k = std::max(1, static_cast<int>(std::ceil(k_real)));
    while (k > 1 && meets(k - 1))
        --k;
    while (!meets(k))
        ++k;
    plan.degree = k;
    plan.error_bound = error_bound(k);

    // Two-vector recurrence coefficients (Saad, Iterative Methods for Sparse
    // Linear Systems, 2nd ed., Algorithm 12.1, with x_{k+1} - x_k = d_k).
    double rho = 1.0 / sigma;
    for (int step = 1; step < k; ++step) {
        const double rho_next = 1.0 / (2.0 * sigma - rho);
        plan.step_a.push_back(rho_next * rho);
        plan.step_b.push_back(2.0 * rho_next / delta);
        rho = rho_next;
    }
    return plan;
}

// ── Solve ──────────────────────────────────────────────────────────────────

namespace {

/// Sum over all domains of the per-plane sums of a*b, added in global plane
/// order (identical for every split of the grid).
double global_dot(const std::vector<WillmoreDomain>& domains, int global_Nx, CudaSolver::CgVector a,
                  CudaSolver::CgVector b) {
    std::vector<double> planes(static_cast<std::size_t>(global_Nx), 0.0);
    for (const auto& d : domains) {
        CUDA_CHECK(cudaSetDevice(d.device_id));
        const std::vector<double> part = d.solver->cg_plane_dot(a, b, d.plane_begin, d.plane_end);
        for (std::size_t k = 0; k < part.size(); ++k)
            planes[static_cast<std::size_t>(d.global_plane_begin) + k] = part[k];
    }
    double sum = 0.0;
    for (double v : planes)
        sum += v;
    return sum;
}

template <typename F> void for_each_domain(const std::vector<WillmoreDomain>& domains, F&& f) {
    for (const auto& d : domains) {
        CUDA_CHECK(cudaSetDevice(d.device_id));
        f(*d.solver);
    }
}

/// Why the squared norm of the right-hand side r = div F - dF/dphi - beta M is not finite,
/// from r, phi and u over the interior cells the domains own (downloaded on this error path
/// only; r reads u at the cell itself): the state is not finite; lambda u, which the kernel
/// forms first in the coupling lambda u (1 - phi^2)^2, is beyond the double range; another
/// term overflows (each involves phi); or r is finite and the sum of its squares overflows.
std::string rhs_failure(const std::vector<WillmoreDomain>& domains) {
    auto nan_max = [](double a, double b) {
        return std::isnan(a) || std::isnan(b) ? std::numeric_limits<double>::quiet_NaN()
                                              : std::max(a, b);
    };
    double r_max = 0.0, coupling_max = 0.0, phi_max = 0.0, u_max = 0.0, lambda = 0.0;
    for (const auto& d : domains) {
        CUDA_CHECK(cudaSetDevice(d.device_id));
        const CudaSolver& solver = *d.solver;
        const KernelParams& p = solver.params();
        lambda = p.lambda;
        const std::vector<double> r = solver.willmore_download(CudaSolver::CgVector::R);
        const Grid grid(Dim3{p.Nx, p.Ny, p.Nz}, Spacing{p.dx, p.dy, p.dz});
        FieldData phi(grid, "phi"), u(grid, "u");
        solver.copy_phi_to_host(phi);
        solver.copy_u_to_host(u);
        for (int x = d.plane_begin; x < d.plane_end; ++x)
            for (int y = 1; y < p.Ny - 1; ++y)
                for (int z = 1; z < p.Nz - 1; ++z) {
                    const std::size_t c =
                        (static_cast<std::size_t>(x) * p.Ny + static_cast<std::size_t>(y)) * p.Nz +
                        static_cast<std::size_t>(z);
                    r_max = nan_max(r_max, std::fabs(r[c]));
                    coupling_max = nan_max(coupling_max, std::fabs(p.lambda * u(x, y, z)));
                    phi_max = nan_max(phi_max, std::fabs(phi(x, y, z)));
                    u_max = nan_max(u_max, std::fabs(u(x, y, z)));
                }
    }
    const std::string what = "Willmore solve: the right-hand side r = div F - dF/dphi - beta M "
                             "is not finite in norm";
    if (!std::isfinite(phi_max) || !std::isfinite(u_max)) {
        return what + ": the state itself is not finite (max |phi| = " + g17(phi_max) +
               ", max |u| = " + g17(u_max) + " over the interior cells): the simulation diverged";
    }
    if (!std::isfinite(r_max)) {
        if (!std::isfinite(coupling_max)) {
            return what +
                   ": lambda u, the coupling's factor, is beyond the double range at an "
                   "interior cell (max |u| = " +
                   g17(u_max) + ", lambda = " + g17(lambda) +
                   "): the temperature, from physics.delta or the u boundary data, is too large";
        }
        return what +
               ": a phase-field term of r (div F, the double well, the coupling lambda u "
               "(1 - phi^2)^2 or beta M) is beyond the double range at an interior cell, "
               "with lambda u finite (max |phi| = " +
               g17(phi_max) + ", max |u| = " + g17(u_max) + "): the phase field is too large";
    }
    return what + ": every component is finite (max |r| = " + g17(r_max) +
           ") but the sum of their squares is beyond the double range (max |phi| = " +
           g17(phi_max) + ", max |u| = " + g17(u_max) + " over the interior cells)";
}

/// Effective (per-face) phi BCs of the global configuration.
PerFaceBoundary global_phi_faces(const SimulationConfig& config) {
    return config.boundary.per_face ? config.boundary.phi_faces
                                    : PerFaceBoundary::uniform(config.boundary.phi_bc);
}

} // namespace

void apply_willmore_preconditioner(const std::vector<WillmoreDomain>& domains,
                                   const ChebyshevPlan& plan, double rr,
                                   const std::function<void(WillmoreBuffer)>& exchange) {
    if (!(rr > 0.0) || !std::isfinite(rr))
        throw std::invalid_argument("apply_willmore_preconditioner: needs 0 < r.r < inf");
    // Power-of-two scaling into single precision: sqrt(rr) 2^-e lies in
    // [1, 2), so no component overflows; components below 2^-126 in
    // magnitude are rounded to multiples of 2^-149 (see the header). The
    // scaling adds no rounding of its own, so the result is the same bits for
    // r and 2^m r, up to 2^m, as long as no r_c^2 underflows in double
    // precision.
    const int e = std::ilogb(std::sqrt(rr));
    const double scale = std::ldexp(1.0, -e), inv_scale = std::ldexp(1.0, e);
    const int steps = plan.degree - 1;
    for_each_domain(domains, [&](CudaSolver& s) {
        s.willmore_precondition_first(plan, scale, inv_scale, steps == 0);
    });
    for (int k = 1; k <= steps; ++k) {
        exchange(WillmoreBuffer::PreconditionerIterate);
        for_each_domain(domains, [](CudaSolver& s) { s.willmore_precondition_laplacian(); });
        exchange(WillmoreBuffer::PreconditionerLaplacian);
        for_each_domain(domains, [&](CudaSolver& s) {
            s.willmore_precondition_step(plan, k, inv_scale, k == steps);
        });
    }
}

WillmoreSolveStats solve_willmore_step(const std::vector<WillmoreDomain>& domains,
                                       const SimulationConfig& config,
                                       const std::function<void(WillmoreBuffer)>& exchange) {
    using V = CudaSolver::CgVector;
    const int global_Nx = config.grid.Nx;

    for_each_domain(domains, [](CudaSolver& s) { s.willmore_omega(); });
    exchange(WillmoreBuffer::Omega);
    for_each_domain(domains, [](CudaSolver& s) { s.willmore_rhs_init(); });

    WillmoreSolveStats stats;
    // tau range over the interior cells of every domain (exact min / max).
    stats.tau_min = std::numeric_limits<double>::infinity();
    stats.tau_max = -std::numeric_limits<double>::infinity();
    double dt = 0.0;
    for (const auto& d : domains) {
        CUDA_CHECK(cudaSetDevice(d.device_id));
        const auto [lo, hi] = d.solver->willmore_tau_range();
        // NaN propagates: a NaN tau fails the checks in willmore_spectrum.
        stats.tau_min = std::isnan(lo) ? lo : std::min(stats.tau_min, lo);
        stats.tau_max = std::isnan(hi) ? hi : std::max(stats.tau_max, hi);
        dt = d.solver->params().dt;
    }
    stats.spectrum =
        willmore_spectrum(config, global_phi_faces(config), dt, stats.tau_min, stats.tau_max);
    const ChebyshevPlan plan = chebyshev_plan(stats.spectrum, kWillmoreCgTolerance);
    stats.chebyshev_degree = plan.degree;

    const double bb = global_dot(domains, global_Nx, V::R, V::R);
    if (!std::isfinite(bb))
        throw std::runtime_error(rhs_failure(domains));
    if (bb > 0.0) {
        const double target = kWillmoreCgTolerance * std::sqrt(bb);
        auto require_positive = [](double v, const char* what, int k) {
            if (!(v > 0.0))
                throw std::runtime_error(std::string("Willmore CG breakdown: ") + what + " = " +
                                         g17(v) + " at iteration " + std::to_string(k) +
                                         " (operator or preconditioner not positive "
                                         "definite, or non-finite data)");
        };
        apply_willmore_preconditioner(domains, plan, bb, exchange);
        double rz = global_dot(domains, global_Nx, V::R, V::Z);
        require_positive(rz, "r.z", 0);
        // p = z (+ 0 * p: willmore_rhs_init zeroed p).
        for_each_domain(domains, [](CudaSolver& s) { s.willmore_update_direction(0.0); });

        bool converged = false;
        double rr = bb;
        for (int k = 1; k <= kWillmoreCgMaxIterations; ++k) {
            for_each_domain(domains, [](CudaSolver& s) { s.willmore_direction_bc(); });
            exchange(WillmoreBuffer::Direction);
            for_each_domain(domains, [](CudaSolver& s) { s.willmore_direction_laplacian(); });
            exchange(WillmoreBuffer::DirectionLaplacian);
            for_each_domain(domains, [](CudaSolver& s) { s.willmore_apply(); });

            const double pAp = global_dot(domains, global_Nx, V::P, V::AP);
            require_positive(pAp, "p.Ap", k);
            const double alpha = rz / pAp;
            for_each_domain(domains, [alpha](CudaSolver& s) { s.willmore_update_xr(alpha); });

            rr = global_dot(domains, global_Nx, V::R, V::R);
            stats.iterations = k;
            if (std::sqrt(rr) <= target) {
                converged = true;
                break;
            }
            apply_willmore_preconditioner(domains, plan, rr, exchange);
            const double rz_next = global_dot(domains, global_Nx, V::R, V::Z);
            require_positive(rz_next, "r.z", k);
            // Flexible (Polak-Ribiere) beta = z_{k+1}.(r_{k+1} - r_k) / (z_k.r_k)
            // = -alpha z_{k+1}.Ap / (z_k.r_k), since r_{k+1} - r_k = -alpha Ap.
            const double zAp = global_dot(domains, global_Nx, V::Z, V::AP);
            const double beta = -alpha * zAp / rz;
            rz = rz_next;
            for_each_domain(domains, [beta](CudaSolver& s) { s.willmore_update_direction(beta); });
        }
        stats.relative_residual = std::sqrt(rr / bb);
        if (!converged) {
            throw std::runtime_error(
                "Willmore CG did not converge in " + std::to_string(kWillmoreCgMaxIterations) +
                " iterations (relative residual " + g17(stats.relative_residual) + ")");
        }
    }
    for_each_domain(domains, [](CudaSolver& s) { s.willmore_finish(); });
    return stats;
}

} // namespace ac::cuda
