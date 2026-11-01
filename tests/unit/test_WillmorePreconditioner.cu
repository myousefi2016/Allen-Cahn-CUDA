// Tests of the Willmore solve's spectral bounds, Chebyshev plan and
// single-precision polynomial preconditioner (docs/THEORY.md section 4.4).
#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/WillmoreSolve.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

namespace {

using V = CudaSolver::CgVector;
constexpr double kU64 = std::numeric_limits<double>::epsilon() / 2; // unit roundoff
constexpr double kU32 = std::numeric_limits<float>::epsilon() / 2;

/// Eigenvalues of the symmetric n x n matrix `a` (row-major) by the cyclic
/// Jacobi method (Golub & Van Loan, Matrix Computations, 2nd ed., Section 8.5,
/// Jacobi Methods). It stops when the off-diagonal Frobenius norm
/// is below the unit roundoff times the matrix norm, so every
/// eigenvalue is then that close (Weyl). Fails the test if 100 sweeps (the
/// method converges quadratically) do not get there.
std::vector<double> symmetric_eigenvalues(std::vector<double> a, int n) {
    auto at = [&](int i, int j) -> double& { return a[static_cast<std::size_t>(i * n + j)]; };
    double norm2 = 0;
    for (double v : a)
        norm2 += v * v;
    const double tol2 =
        std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon() * norm2;
    bool converged = false;
    for (int sweep = 0; sweep < 100 && !converged; ++sweep) {
        double off2 = 0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                off2 += 2 * at(i, j) * at(i, j);
        if (off2 <= tol2) {
            converged = true;
            break;
        }
        for (int p = 0; p < n; ++p)
            for (int q = p + 1; q < n; ++q) {
                if (at(p, q) == 0)
                    continue;
                const double theta = (at(q, q) - at(p, p)) / (2 * at(p, q));
                const double t =
                    (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(1 + theta * theta));
                const double c = 1 / std::sqrt(1 + t * t), s = t * c;
                for (int k = 0; k < n; ++k) {
                    const double akp = at(k, p), akq = at(k, q);
                    at(k, p) = c * akp - s * akq;
                    at(k, q) = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    const double apk = at(p, k), aqk = at(q, k);
                    at(p, k) = c * apk - s * aqk;
                    at(q, k) = s * apk + c * aqk;
                }
            }
    }
    EXPECT_TRUE(converged) << "Jacobi eigenvalue iteration did not converge";
    std::vector<double> ev(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        ev[static_cast<std::size_t>(i)] = at(i, i);
    std::sort(ev.begin(), ev.end());
    return ev;
}

BoundaryConfig bc_of(BCType type, double alpha = 0.0, double beta = 0.0) {
    BoundaryConfig bc;
    bc.type = type;
    bc.alpha = alpha;
    bc.beta = beta;
    return bc;
}

/// The 1-D second-difference matrix (unit spacing) of n cells with the
/// homogeneous ghost rules, built from the rules' definitions: Dirichlet
/// ghost 0, Neumann ghost = neighbour, Robin ghost = (beta/h)/(alpha+beta/h)
/// * neighbour, periodic ghost = cell at the other end.
std::vector<double> second_difference_matrix(int n, const BoundaryConfig& lo,
                                             const BoundaryConfig& hi, double h) {
    std::vector<double> m(static_cast<std::size_t>(n * n), 0);
    auto at = [&](int i, int j) -> double& { return m[static_cast<std::size_t>(i * n + j)]; };
    auto ghost = [&](const BoundaryConfig& bc, int self, int other_end) {
        switch (bc.type) {
        case BCType::Dirichlet:
            break;
        case BCType::Neumann:
            at(self, self) += 1;
            break;
        case BCType::Robin:
            at(self, self) += (bc.beta / h) / (bc.alpha + bc.beta / h);
            break;
        case BCType::Periodic:
            at(self, other_end) += 1;
            break;
        }
    };
    for (int i = 0; i < n; ++i) {
        at(i, i) -= 2;
        if (i > 0)
            at(i, i - 1) += 1;
        else
            ghost(lo, 0, n - 1);
        if (i < n - 1)
            at(i, i + 1) += 1;
        else
            ghost(hi, n - 1, 0);
    }
    return m;
}

class WillmorePreconditionerTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// A small anisotropic grid with the given phi faces, eps = 0.12, beta = 1.
SimulationConfig small_config(int Nx, int Ny, int Nz, const PerFaceBoundary& phi_faces) {
    SimulationConfig cfg;
    cfg.grid.Nx = Nx;
    cfg.grid.Ny = Ny;
    cfg.grid.Nz = Nz;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.5;
    cfg.grid.dz = 0.3;
    cfg.physics.epsilon = 0.12;
    cfg.physics.W0 = 1.0;
    cfg.physics.willmore_beta = 1.0;
    cfg.time.dt = 0.004;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.stencil = StencilType::Standard7Point;
    cfg.boundary.per_face = true;
    cfg.boundary.phi_faces = phi_faces;
    cfg.boundary.u_faces = PerFaceBoundary::uniform(bc_of(BCType::Neumann));
    cfg.validate();
    return cfg;
}

/// phi random in [-1, 1] (so the gradient directions, and tau, vary from cell
/// to cell), u random in [-0.8, 0].
void random_state(FieldData& phi, FieldData& u, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> P(-1.0, 1.0), U(-0.8, 0.0);
    for (std::size_t i = 0; i < phi.size(); ++i) {
        phi.data()[i] = P(rng);
        u.data()[i] = U(rng);
    }
}

std::vector<int> interior_cells(const SimulationConfig& cfg) {
    std::vector<int> cells;
    for (int x = 1; x < cfg.grid.Nx - 1; ++x)
        for (int y = 1; y < cfg.grid.Ny - 1; ++y)
            for (int z = 1; z < cfg.grid.Nz - 1; ++z)
                cells.push_back((x * cfg.grid.Ny + y) * cfg.grid.Nz + z);
    return cells;
}

/// The production double-precision operator A as a dense matrix over the
/// interior cells: column j is A e_j computed by the solver's building blocks
/// (homogeneous BCs on p, t = lap p with BCs, Ap).
std::vector<double> assemble_operator(CudaSolver& solver, const std::vector<int>& cells) {
    const int n = static_cast<int>(cells.size());
    std::vector<double> A(static_cast<std::size_t>(n * n));
    std::vector<double> e(solver.total_points(), 0.0);
    for (int j = 0; j < n; ++j) {
        e[static_cast<std::size_t>(cells[static_cast<std::size_t>(j)])] = 1.0;
        solver.willmore_upload(V::P, e);
        e[static_cast<std::size_t>(cells[static_cast<std::size_t>(j)])] = 0.0;
        solver.willmore_direction_bc();
        solver.willmore_direction_laplacian();
        solver.willmore_apply();
        const std::vector<double> Ae = solver.willmore_download(V::AP);
        for (int i = 0; i < n; ++i)
            A[static_cast<std::size_t>(i * n + j)] =
                Ae[static_cast<std::size_t>(cells[static_cast<std::size_t>(i)])];
    }
    return A;
}

/// The 7-point Laplacian of v (a full-grid vector) with the exact spacings over
/// the interior cells of cfg's grid, each face's ghost following the homogeneous
/// form of phi's BC there, as second_difference_matrix builds it: Dirichlet 0,
/// Neumann the adjacent interior cell's value, Robin (beta/h)/(alpha + beta/h)
/// times it, periodic the interior cell at the other end. With `magnitude`, the
/// sum of the terms' magnitudes instead, sum_d (|v_-| + |v_+| + 2 |v_c|) / h_d^2:
/// what the roundings of evaluating the sum scale with. 0 outside the interior.
std::vector<double> host_laplacian(const SimulationConfig& cfg, const std::vector<double>& v,
                                   bool magnitude) {
    const PerFaceBoundary faces = cfg.boundary.per_face
                                      ? cfg.boundary.phi_faces
                                      : PerFaceBoundary::uniform(cfg.boundary.phi_bc);
    const int n[3] = {cfg.grid.Nx, cfg.grid.Ny, cfg.grid.Nz};
    const double h[3] = {cfg.grid.dx, cfg.grid.dy, cfg.grid.dz};
    auto index = [&](const int(&q)[3]) {
        return static_cast<std::size_t>((q[0] * n[1] + q[1]) * n[2] + q[2]);
    };
    std::vector<double> out(v.size(), 0.0);
    for (int x = 1; x < n[0] - 1; ++x)
        for (int y = 1; y < n[1] - 1; ++y)
            for (int z = 1; z < n[2] - 1; ++z) {
                const int c[3] = {x, y, z};
                const double vc = v[index(c)];
                double sum = 0.0;
                for (int d = 0; d < 3; ++d) {
                    double side[2] = {0.0, 0.0};
                    for (int s = 0; s < 2; ++s) {
                        const BoundaryConfig& bc = faces.faces[static_cast<std::size_t>(2 * d + s)];
                        int q[3] = {x, y, z};
                        q[d] += s == 0 ? -1 : 1;
                        if (q[d] >= 1 && q[d] <= n[d] - 2) {
                            side[s] = v[index(q)];
                        } else if (bc.type == BCType::Periodic) {
                            q[d] = s == 0 ? n[d] - 2 : 1;
                            side[s] = v[index(q)];
                        } else if (bc.type == BCType::Neumann) {
                            side[s] = vc;
                        } else if (bc.type == BCType::Robin) {
                            side[s] = (bc.beta / h[d]) / (bc.alpha + bc.beta / h[d]) * vc;
                        } // Dirichlet: 0
                    }
                    const double term =
                        magnitude ? std::fabs(side[0]) + std::fabs(side[1]) + 2 * std::fabs(vc)
                                  : side[0] + side[1] - 2 * vc;
                    sum += term / (h[d] * h[d]);
                }
                out[index(c)] = sum;
            }
    return out;
}

struct Case {
    std::string name;
    int Nx, Ny, Nz;
    PerFaceBoundary faces;
};

/// Grids and phi BCs covering every accepted ghost rule: zero flux,
/// Dirichlet, periodic (paired), Robin with c in (0, 1) (c = 0.56 at h = 0.4),
/// near 1 (0.98) and near 0 (0.012), and with alpha, beta both negative,
/// per-face mixes, and axes of 1 and 2 interior cells. (Robin c outside
/// [0, 1] is rejected by SimulationConfig::validate; the per-axis bounds for
/// it are tested on the host below.)
std::vector<Case> spectrum_cases() {
    const BoundaryConfig N = bc_of(BCType::Neumann), D = bc_of(BCType::Dirichlet),
                         P = bc_of(BCType::Periodic), R = bc_of(BCType::Robin, 1.0, 0.5),
                         R_near_1 = bc_of(BCType::Robin, 0.05, 1.0),
                         R_near_0 = bc_of(BCType::Robin, 20.0, 0.1),
                         R_negative = bc_of(BCType::Robin, -1.0, -0.5);
    auto faces = [](BoundaryConfig xl, BoundaryConfig xh, BoundaryConfig yl, BoundaryConfig yh,
                    BoundaryConfig zl, BoundaryConfig zh) {
        PerFaceBoundary f;
        f.faces = {xl, xh, yl, yh, zl, zh};
        return f;
    };
    return {
        {"neumann", 7, 6, 8, PerFaceBoundary::uniform(N)},
        {"dirichlet", 7, 6, 8, PerFaceBoundary::uniform(D)},
        {"periodic", 7, 6, 8, PerFaceBoundary::uniform(P)},
        {"mixed", 7, 6, 8, faces(P, P, D, N, R, N)},
        {"robin_c_near_1", 7, 6, 8, faces(R_near_1, N, R_near_1, R_near_1, N, R_near_1)},
        {"robin_c_near_0", 7, 6, 8, faces(R_near_0, N, N, R_near_0, R_near_0, D)},
        {"robin_negative", 7, 6, 8, faces(R_negative, R, N, R_negative, D, R_negative)},
        {"thin_axes", 3, 4, 9, faces(N, R, P, P, D, R_near_1)},
    };
}

} // namespace

// ── Host-only: the per-axis bounds ─────────────────────────────────────────

// The Gershgorin bounds of the 1-D second-difference matrix contain its
// exact eigenvalues for every ghost rule and every size down to one cell.
// Robin(1e-31, 0) is a Dirichlet-like face (c = 0) whatever the size of its
// denominator.
TEST(WillmoreSpectrumBounds, SecondDifferenceBoundsContainTheExactSpectrum) {
    const double h = 0.4;
    const std::vector<BoundaryConfig> rules = {
        bc_of(BCType::Neumann),          bc_of(BCType::Dirichlet),
        bc_of(BCType::Robin, 1.0, 0.5),  bc_of(BCType::Robin, -1.0, 1.0),
        bc_of(BCType::Robin, -4.0, 1.0), bc_of(BCType::Robin, 1e-31, 0.0)};
    for (int n = 1; n <= 6; ++n) {
        std::vector<std::pair<BoundaryConfig, BoundaryConfig>> pairs;
        for (const auto& lo : rules)
            for (const auto& hi : rules)
                pairs.emplace_back(lo, hi);
        pairs.emplace_back(bc_of(BCType::Periodic), bc_of(BCType::Periodic));
        for (const auto& [lo, hi] : pairs) {
            double lower = 0, upper = 0;
            second_difference_bounds(n, lo, hi, h, lower, upper);
            const auto ev = symmetric_eigenvalues(second_difference_matrix(n, lo, hi, h), n);
            // Bounds and eigenvalues agree to the few (< 8) roundings of the
            // double-precision bound arithmetic, relative to their magnitude.
            const double s = 1.0 / (h * h);
            const double slack = 8 * kU64 * std::max(std::fabs(lower), std::fabs(upper));
            EXPECT_GE(ev.front() * s, lower - slack) << "n=" << n;
            EXPECT_LE(ev.back() * s, upper + slack) << "n=" << n;
        }
    }
    double lo = 0, hi = 0;
    EXPECT_THROW(
        second_difference_bounds(4, bc_of(BCType::Periodic), bc_of(BCType::Neumann), h, lo, hi),
        std::invalid_argument);
}

// ── Host-only: the Chebyshev plan ──────────────────────────────────────────

// The plan's degree is the lowest meeting the two-iteration bound
// 2 q^2 sqrt(kappa_A) <= tol, its error bound is 1/T_k(sigma) (T_k from the
// three-term recurrence), and its recurrence coefficients reproduce the
// Chebyshev residual polynomial r(lambda) = T_k((theta - lambda)/delta) /
// T_k(theta/delta): with D = 1 and r = 1, x_k(lambda) of the two-vector
// iteration satisfies 1 - lambda x_k(lambda) = r(lambda).
TEST(WillmoreChebyshevPlan, LowestDegreeMeetingTheBoundAndExactPolynomial) {
    const double tol = kWillmoreCgTolerance;
    struct S {
        double a, b, kappa_A;
    };
    for (const S& sp :
         {S{0.0409, 3.2144, 79.5}, S{0.00728, 3.379, 465.0}, S{0.5, 1.5, 3.0}, S{0.9, 1.1, 1.3}}) {
        WillmoreSpectrum s;
        s.a = sp.a;
        s.b = sp.b;
        s.kappa_A = sp.kappa_A;
        const ChebyshevPlan plan = chebyshev_plan(s, tol);
        const double sigma = (sp.b + sp.a) / (sp.b - sp.a);
        auto T = [](int k, double x) { // T_k(x), x >= 1, by the recurrence
            double t0 = 1, t1 = x;
            for (int i = 1; i < k; ++i) {
                const double t2 = 2 * x * t1 - t0;
                t0 = t1;
                t1 = t2;
            }
            return k == 0 ? 1.0 : t1;
        };
        auto bound = [&](int k) {
            const double e = 1 / T(k, sigma);
            const double q = e / (1 + std::sqrt(1 - e * e));
            return 2 * q * q * std::sqrt(sp.kappa_A);
        };
        const int k = plan.degree;
        EXPECT_LE(bound(k), tol) << "a=" << sp.a;
        if (k > 1)
            EXPECT_GT(bound(k - 1), tol) << "a=" << sp.a;
        EXPECT_NEAR(plan.error_bound * T(k, sigma), 1.0, 64 * k * kU64);
        ASSERT_EQ(plan.step_a.size(), static_cast<std::size_t>(k - 1));

        // The recurrences are stable, so rounding grows at most linearly per
        // step: compare within k^2 u relative to the polynomial's range.
        const double theta = 0.5 * (sp.b + sp.a), delta = 0.5 * (sp.b - sp.a);
        for (int i = 0; i <= 20; ++i) {
            const double lambda = sp.a + (sp.b - sp.a) * i / 20.0;
            double x_prev = 0, x = plan.s0;
            for (int j = 0; j + 1 < k; ++j) {
                const double xn = x + plan.step_a[static_cast<std::size_t>(j)] * (x - x_prev) +
                                  plan.step_b[static_cast<std::size_t>(j)] * (1 - lambda * x);
                x_prev = x;
                x = xn;
            }
            const double arg = (theta - lambda) / delta; // in [-1, 1]
            const double exact = std::cos(k * std::acos(std::clamp(arg, -1.0, 1.0))) / T(k, sigma);
            EXPECT_NEAR(1 - lambda * x, exact, static_cast<double>(k) * k * kU64 * 16)
                << "a=" << sp.a << " lambda=" << lambda;
        }
    }
    WillmoreSpectrum flat;
    flat.a = flat.b = 2.0;
    flat.kappa_A = 1.0;
    const ChebyshevPlan exact = chebyshev_plan(flat, tol);
    EXPECT_EQ(exact.degree, 1);
    EXPECT_DOUBLE_EQ(exact.s0, 0.5);
}

// ── GPU: the bounds against the production operator ───────────────────────

// For every ghost rule, anisotropic spacing and thin axes, the eigenvalues of
// D^-1/2 A D^-1/2, with A assembled column by column from the production
// double-precision operator and tau from the solver, lie in [a, b] of
// willmore_spectrum. A is also symmetric. Tolerances are rounding bounds:
// every entry passes through at most 16 double roundings, so the assembled
// matrix is within 16 u of the exact one entrywise, and by Weyl its
// eigenvalues within 16 u ||S||_F.
TEST_F(WillmorePreconditionerTest, SpectrumBoundsContainEveryEigenvalue) {
    for (const Case& cs : spectrum_cases()) {
        SCOPED_TRACE(cs.name);
        const SimulationConfig cfg = small_config(cs.Nx, cs.Ny, cs.Nz, cs.faces);
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 7);
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        solver.mutable_params().dt = cfg.time.dt;
        solver.willmore_omega();
        solver.willmore_rhs_init();
        const auto [tau_min, tau_max] = solver.willmore_tau_range();
        const WillmoreSpectrum sp =
            willmore_spectrum(cfg, cfg.boundary.phi_faces, cfg.time.dt, tau_min, tau_max);
        const std::vector<double> tau = solver.willmore_download(V::Tau);
        const std::vector<int> cells = interior_cells(cfg);
        const int n = static_cast<int>(cells.size());
        std::vector<double> A = assemble_operator(solver, cells);

        double amax = 0, asym = 0, snorm2 = 0;
        for (double v : A)
            amax = std::max(amax, std::fabs(v));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                asym = std::max(asym, std::fabs(A[static_cast<std::size_t>(i * n + j)] -
                                                A[static_cast<std::size_t>(j * n + i)]));
        EXPECT_LE(asym, 2 * 16 * kU64 * amax);

        const double beta = cfg.physics.willmore_beta;
        std::vector<double> S(A.size());
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                const double Di =
                    tau[static_cast<std::size_t>(cells[static_cast<std::size_t>(i)])] /
                        cfg.time.dt +
                    beta * sp.d_L;
                const double Dj =
                    tau[static_cast<std::size_t>(cells[static_cast<std::size_t>(j)])] /
                        cfg.time.dt +
                    beta * sp.d_L;
                // Symmetric part: the rounding asymmetry is covered above.
                const double aij = 0.5 * (A[static_cast<std::size_t>(i * n + j)] +
                                          A[static_cast<std::size_t>(j * n + i)]);
                S[static_cast<std::size_t>(i * n + j)] = aij / std::sqrt(Di * Dj);
                snorm2 +=
                    S[static_cast<std::size_t>(i * n + j)] * S[static_cast<std::size_t>(i * n + j)];
            }
        const auto ev = symmetric_eigenvalues(S, n);
        const double slack = 16 * kU64 * std::sqrt(snorm2);
        std::printf("%-16s n=%3d  eig [%.6f, %.6f]  bounds [%.6f, %.6f]\n", cs.name.c_str(), n,
                    ev.front(), ev.back(), sp.a, sp.b);
        EXPECT_GE(ev.front(), sp.a - slack);
        EXPECT_LE(ev.back(), sp.b + slack);
        EXPECT_GT(ev.front(), 0.0);
    }
}

// ── GPU: the single-precision preconditioner ───────────────────────────────

// z = M r from the device (single precision) agrees with the same Chebyshev
// polynomial applied in double precision to the assembled operator. The
// device computes the residual r - A x_k explicitly at every step, so there
// is no gap between a recursively updated and the true residual, and for a
// real spectrum the recurrence multiplies earlier local errors by factors of
// modulus below 1 (Gutknecht & Roellin, Parallel Comput. 28 (2002) 263-283,
// Algorithm 2 and Theorem 2). Each of the k steps therefore contributes its
// own rounding once; its leading part is the cancellation in the explicit
// residual, a relative (b/a) u_32 for the D-scaled operator. k (b/a) u_32 is
// that first-order size of the accumulated difference. It leaves out the
// number of roundings per step, so it is a tolerance, not a proven bound: the
// measured differences (printed, 6.1e-8 to 8.0e-8 on an RTX 4090) lie more
// than 100 times below it, and an error in the operator (a wrong ghost rule,
// stencil or step size) shows as O(1). An error in the polynomial does not:
// with explicit residuals any nearby polynomial also approximates A^-1 r to
// about its own error bound 1/T_k(sigma) (4.5e-6 to 6.3e-6 here), within this
// tolerance, so one step fewer or another step's coefficients pass;
// EveryChebyshevStepIsThePlansStep tests the polynomial step by step. The
// solve uses a dt other than the configuration's, as adaptive stepping does.
TEST_F(WillmorePreconditionerTest, SinglePrecisionMatchesExactPolynomial) {
    for (const Case& cs : spectrum_cases()) {
        SCOPED_TRACE(cs.name);
        const SimulationConfig cfg = small_config(cs.Nx, cs.Ny, cs.Nz, cs.faces);
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 11);
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        // A step size other than the configuration's: the single-precision
        // operator must use the step's own dt (tau/dt), as A does.
        const double dt = cfg.time.dt / 4.0;
        solver.mutable_params().dt = dt;
        solver.willmore_omega();
        solver.willmore_rhs_init();
        const auto [tau_min, tau_max] = solver.willmore_tau_range();
        const WillmoreSpectrum sp =
            willmore_spectrum(cfg, cfg.boundary.phi_faces, dt, tau_min, tau_max);
        const ChebyshevPlan plan = chebyshev_plan(sp, kWillmoreCgTolerance);
        const std::vector<double> tau = solver.willmore_download(V::Tau);
        const std::vector<int> cells = interior_cells(cfg);
        const int n = static_cast<int>(cells.size());
        const std::vector<double> A = assemble_operator(solver, cells);

        std::mt19937_64 rng(3);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        std::vector<double> r(solver.total_points(), 0.0);
        std::vector<double> rv(static_cast<std::size_t>(n)), Dinv(static_cast<std::size_t>(n));
        double rr = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto c = static_cast<std::size_t>(cells[static_cast<std::size_t>(i)]);
            r[c] = U(rng);
            rv[static_cast<std::size_t>(i)] = r[c];
            rr += r[c] * r[c];
            Dinv[static_cast<std::size_t>(i)] =
                1.0 / (tau[c] / dt + cfg.physics.willmore_beta * sp.d_L);
        }
        solver.willmore_upload(V::R, r);
        apply_willmore_preconditioner({WillmoreDomain{&solver, 0, 1, cfg.grid.Nx - 1, 1}}, plan, rr,
                                      [](WillmoreBuffer) {});
        const std::vector<double> z = solver.willmore_download(V::Z);

        // Reference: x_1 = s0 D^-1 r, x_{k+1} = x_k + a_k (x_k - x_{k-1}) + b_k D^-1 (r - A x_k).
        std::vector<double> x(static_cast<std::size_t>(n)), xp(static_cast<std::size_t>(n), 0);
        for (int i = 0; i < n; ++i)
            x[static_cast<std::size_t>(i)] =
                plan.s0 * Dinv[static_cast<std::size_t>(i)] * rv[static_cast<std::size_t>(i)];
        for (std::size_t k = 0; k < plan.step_a.size(); ++k) {
            std::vector<double> xn(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                double Ax = 0;
                for (int j = 0; j < n; ++j)
                    Ax += A[static_cast<std::size_t>(i * n + j)] * x[static_cast<std::size_t>(j)];
                const auto ii = static_cast<std::size_t>(i);
                xn[ii] = x[ii] + plan.step_a[k] * (x[ii] - xp[ii]) +
                         plan.step_b[k] * Dinv[ii] * (rv[ii] - Ax);
            }
            xp = x;
            x = xn;
        }
        double diff2 = 0, ref2 = 0;
        for (int i = 0; i < n; ++i) {
            const double d = z[static_cast<std::size_t>(cells[static_cast<std::size_t>(i)])] -
                             x[static_cast<std::size_t>(i)];
            diff2 += d * d;
            ref2 += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        }
        const double rel = static_cast<double>(std::sqrt(diff2 / ref2));
        const double allowed = plan.degree * (sp.b / sp.a) * kU32;
        std::printf("%-16s degree %2d  |z32 - z_exact| / |z_exact| = %.2e  (allowed %.2e)\n",
                    cs.name.c_str(), plan.degree, rel, allowed);
        EXPECT_LE(rel, allowed);
    }
}

// The device applies the plan's polynomial, step by step. For m = 1 .. k the
// plan P_m with the plan's s0, d_L and first m - 1 recurrence coefficients
// gives z_m = x_m of
//   x_1 = s0 D^-1 r,  x_{j+1} = x_j + a_j (x_j - x_{j-1}) + b_j D^-1 (r - A x_j)
// (x_0 = 0; the Chebyshev coefficients do not depend on the degree, so P_m is
// the degree-m plan on [a, b]). Each kernel computes every cell in one thread
// without atomics, so the device's iterates while applying P_m are the bits of
// z_1 .. z_{m-1}: z_1 must be s0 D^-1 r and every z_m (m >= 2) one step from
// z_{m-1} and z_{m-2}. Both are checked cell by cell against the step computed
// in double on the host from those device values, with the Laplacian written
// out here (host_laplacian) and exact constants (tau/dt, 1/h^2, d_L, s0, a_j,
// b_j). The plans are applied in order on one solver, so from P_4 on every
// application finds the previous one's iterates in its buffers, as in a CG
// solve: the first step must not read them. SinglePrecisionMatchesExactPolynomial
// cannot see a wrong polynomial (see its comment). Here a missing step, another
// step's coefficients, or a wrong s0, d_L or x_0 changes z_m by a fraction of
// z_m itself within the first four steps.
//
// Bound: the standard model (Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., Section 2.2, Lemmas 3.1 and 3.3). A single-precision
// operation returns the exact result times (1 + delta), |delta| <= u = 2^-24.
// A fused multiply-add only removes roundings. A quotient is counted 4 times,
// which also covers the 2-ulp division of AC_CUDA_FAST_MATH builds. Every
// operation is a sum or difference, or a product or quotient of data and a
// constant, so the computed value is the exact expression with each term
// multiplied by at most N factors (1 + delta)^(+-1). It is therefore within
// gamma_N = N u / (1 - N u) times the expression evaluated with magnitudes
// (each minus a plus). N on the longest path of a term:
//  - constants as formed from doubles: 1/h^2 3 (h h, 1/., float), W0^4 4,
//    4 W0^2 2, beta 1, beta d_L 2, tau/dt 3 (1/dt, tau times it, float),
//    Robin c 5, s0, a_j, b_j and f = float(2^-e r) 1 each;
//    D = tau/dt + beta d_L 4; a ghost c v of an iterate or Laplacian 6;
//  - lap32: the ghost, two sums, 1/h^2, its product, two outer sums: 14.
//    So t = lap x is within gamma_14 |lap||x|, and lap t, 14 more on t's own
//    error, within gamma_28 |lap|^2 |x|;
//  - A x: W0^4, its product, two sums, beta, its product, the last sum: 10,
//    on lap t's 28: 38 (4 W0^2 t: 8 on t's 14; tau/dt x 5; 4 x 4);
//  - the step: x 2, a (x - x_prev) 5, b (f - A x) / D: the difference, b,
//    its product, D, the quotient, the last sum 12, on A x's 38: 50 (f 13);
//  - the first step s0 f / D: s0, f, the product, D, the quotient: 11.
// One more u covers the host's double-precision evaluation (fewer than 100
// roundings of 2^-53, relative to the same magnitudes). Underflow: an
// operation's result may also be off by up to 2^-126 (2^-150 with gradual
// underflow; fast math flushes). One cell's step takes fewer than 256
// operations and reads. Each such error reaches the result with a gain below
// the step's magnitude expression at unit magnitudes, with every constant c
// replaced by max(1, |c|) and |lap| by its largest row sum 4 sum_d 1/h_d^2,
// times (1 + u)^50 < 2. All of this is in units of the scaled iterate
// (z = 2^e x).
TEST_F(WillmorePreconditionerTest, EveryChebyshevStepIsThePlansStep) {
    constexpr int kFirstStepRoundings = 11;
    constexpr int kStepRoundings = 50;
    constexpr double kOperations = 256;
    auto gamma_n = [](int roundings) { return roundings * kU32 / (1 - roundings * kU32); };
    auto at_least_1 = [](double v) { return std::max(1.0, std::fabs(v)); };
    for (const Case& cs : spectrum_cases()) {
        SCOPED_TRACE(cs.name);
        const SimulationConfig cfg = small_config(cs.Nx, cs.Ny, cs.Nz, cs.faces);
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 11);
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        const double dt = cfg.time.dt / 4.0;
        solver.mutable_params().dt = dt;
        solver.willmore_omega();
        solver.willmore_rhs_init();
        const auto [tau_min, tau_max] = solver.willmore_tau_range();
        const WillmoreSpectrum sp =
            willmore_spectrum(cfg, cfg.boundary.phi_faces, dt, tau_min, tau_max);
        const ChebyshevPlan plan = chebyshev_plan(sp, kWillmoreCgTolerance);
        ASSERT_GE(plan.degree, 4); // P_4 is the first application that finds stale iterates
        const std::vector<double> tau = solver.willmore_download(V::Tau);
        const std::vector<int> cells = interior_cells(cfg);

        std::mt19937_64 rng(3);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        std::vector<double> r(solver.total_points(), 0.0);
        double rr = 0.0;
        for (int c : cells) {
            const auto i = static_cast<std::size_t>(c);
            r[i] = U(rng);
            rr += r[i] * r[i];
        }
        solver.willmore_upload(V::R, r);

        const double beta = cfg.physics.willmore_beta;
        const double W2 = cfg.physics.W0 * cfg.physics.W0;
        double tdt_max = 0.0, D_min = std::numeric_limits<double>::infinity();
        for (int c : cells) {
            const double tdt = tau[static_cast<std::size_t>(c)] / dt;
            tdt_max = std::max(tdt_max, tdt);
            D_min = std::min(D_min, tdt + beta * plan.d_L);
        }
        const double lap_row = 4.0 * (at_least_1(1.0 / (cfg.grid.dx * cfg.grid.dx)) +
                                      at_least_1(1.0 / (cfg.grid.dy * cfg.grid.dy)) +
                                      at_least_1(1.0 / (cfg.grid.dz * cfg.grid.dz)));
        const double a_max = *std::max_element(plan.step_a.begin(), plan.step_a.end());
        const double b_max = *std::max_element(plan.step_b.begin(), plan.step_b.end());
        const double step_gain = 1.0 + 2.0 * at_least_1(a_max) +
                                 at_least_1(b_max) * at_least_1(2.0 / D_min) *
                                     (1.0 + at_least_1(tdt_max) +
                                      at_least_1(beta) * (at_least_1(W2 * W2) * lap_row * lap_row +
                                                          at_least_1(4.0 * W2) * lap_row + 4.0));
        const double gain =
            2.0 * std::max(at_least_1(plan.s0) * at_least_1(2.0 / D_min), step_gain);
        // z = 2^e x, e = ilogb(sqrt(rr)) (apply_willmore_preconditioner).
        const double underflow =
            kOperations * gain * std::ldexp(1.0, -126) * std::ldexp(1.0, std::ilogb(std::sqrt(rr)));

        // zs[m] = z of P_m, zs[0] = x_0 = 0.
        std::vector<std::vector<double>> zs(static_cast<std::size_t>(plan.degree) + 1,
                                            std::vector<double>(solver.total_points(), 0.0));
        double worst = 0.0;
        int worst_step = 0;
        for (int m = 1; m <= plan.degree; ++m) {
            const auto mi = static_cast<std::size_t>(m);
            ChebyshevPlan pm = plan;
            pm.degree = m;
            pm.step_a.resize(mi - 1);
            pm.step_b.resize(mi - 1);
            apply_willmore_preconditioner({WillmoreDomain{&solver, 0, 1, cfg.grid.Nx - 1, 1}}, pm,
                                          rr, [](WillmoreBuffer) {});
            zs[mi] = solver.willmore_download(V::Z);
            const std::vector<double>& x = zs[mi - 1];
            const std::vector<double>& x_prev = zs[m >= 2 ? mi - 2 : 0];
            double a = 0.0, b = 0.0;
            std::vector<double> t, lt, t_mag, lt_mag;
            if (m >= 2) {
                a = plan.step_a[mi - 2];
                b = plan.step_b[mi - 2];
                t = host_laplacian(cfg, x, false);
                lt = host_laplacian(cfg, t, false);
                t_mag = host_laplacian(cfg, x, true);
                lt_mag = host_laplacian(cfg, t_mag, true);
            }
            int over = 0;
            double step_worst = 0.0;
            for (int c : cells) {
                const auto i = static_cast<std::size_t>(c);
                const double tdt = tau[i] / dt, D = tdt + beta * plan.d_L;
                double expected = 0.0, bound = 0.0;
                if (m == 1) {
                    expected = plan.s0 * r[i] / D;
                    bound = gamma_n(kFirstStepRoundings + 1) * std::fabs(plan.s0 * r[i]) / D;
                } else {
                    const double Ax =
                        tdt * x[i] + beta * (W2 * W2 * lt[i] - 4.0 * W2 * t[i] + 4.0 * x[i]);
                    expected = x[i] + a * (x[i] - x_prev[i]) + b * (r[i] - Ax) / D;
                    const double Ax_mag =
                        tdt * std::fabs(x[i]) +
                        beta * (W2 * W2 * lt_mag[i] + 4.0 * W2 * t_mag[i] + 4.0 * std::fabs(x[i]));
                    bound =
                        gamma_n(kStepRoundings + 1) *
                        (std::fabs(x[i]) + std::fabs(a) * (std::fabs(x[i]) + std::fabs(x_prev[i])) +
                         std::fabs(b) * (std::fabs(r[i]) + Ax_mag) / D);
                }
                bound += underflow;
                const double ratio = std::fabs(zs[mi][i] - expected) / bound;
                if (!(ratio <= 1.0)) // NaN counts
                    ++over;
                step_worst = std::max(step_worst, ratio);
            }
            if (step_worst > worst) {
                worst = step_worst;
                worst_step = m;
            }
            if (over > 0) {
                ADD_FAILURE() << "step " << m << " of degree " << plan.degree << ": " << over
                              << " of " << cells.size()
                              << " cells beyond the rounding bound, largest |z - step| / bound "
                              << step_worst;
                break;
            }
        }
        std::printf("%-16s degree %2d  max over steps and cells of |z_m - step| / bound = %.3f "
                    "(step %d)\n",
                    cs.name.c_str(), plan.degree, worst, worst_step);
    }
}

// A right-hand side far below single precision's range must not underflow:
// the preconditioner scales r by a power of two first. With eps = 0, u = 0
// and |phi| <= 2^-40 the step is exactly linear in phi in double precision
// (phi^3 and every other nonlinear term are below one ulp of the linear
// ones), so the increments for phi = 2^-150 psi and 2^-40 psi must be the
// same bits up to 2^-110. Without the scaling, the right-hand side for
// 2^-150 psi (components 3.4e-47 to 8.1e-43) rounds to zero or to
// subnormals in single precision and the solve breaks down.
TEST_F(WillmorePreconditionerTest, TinyRightHandSideGivesTheScaledIncrementBitwise) {
    const int N = 14;
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = N;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
    cfg.physics.epsilon = 0.0;
    cfg.physics.willmore_beta = 1.0;
    cfg.time.dt = 0.01;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.stencil = StencilType::Standard7Point;
    cfg.boundary.phi_bc = bc_of(BCType::Periodic);
    cfg.boundary.u_bc = bc_of(BCType::Periodic);
    cfg.validate();
    const Grid grid = cfg.make_grid();
    auto run = [&](int exponent) {
        FieldData phi(grid, "phi"), u(grid, "u");
        u.fill(0.0);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z)
                    phi(x, y, z) = std::ldexp(
                        std::cos(0.7 * x) * std::sin(0.4 * y + 0.3 * z) + 0.25, exponent);
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        solver.step(cfg.time.dt);
        solver.copy_phi_to_host(phi);
        return std::vector<double>(phi.data(), phi.data() + phi.size());
    };
    std::vector<double> tiny, normal;
    ASSERT_NO_THROW(tiny = run(-150));
    ASSERT_NO_THROW(normal = run(-40));
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < tiny.size(); ++i)
        mismatches += std::ldexp(tiny[i], 110) != normal[i] ? 1 : 0;
    EXPECT_EQ(mismatches, 0U);
}

// After a solve the true residual ||b - A x|| / ||b||, recomputed from the
// state, meets the tolerance. The recursive residual CG tests drifts from the
// true one by rounding: at most 16 u kappa_A ||b|| per iteration (x and r
// are updated with the same alpha and Ap), far below the tolerance.
TEST_F(WillmorePreconditionerTest, TrueResidualMeetsTheTolerance) {
    std::vector<Case> cases = spectrum_cases();
    // Near the liquid bulk (phi = -1 + 1e-6 * noise) ||b|| is about 0.02.
    // Below ||b|| = 1 a stopping test on ||r||^2 instead of ||r|| is met as
    // soon as ||r|| <= sqrt(1e-10 ||b||), i.e. after the first iteration.
    cases.push_back({"near_equilibrium", 7, 6, 8, cases.front().faces});
    for (const Case& cs : cases) {
        SCOPED_TRACE(cs.name);
        const SimulationConfig cfg = small_config(cs.Nx, cs.Ny, cs.Nz, cs.faces);
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 5);
        if (cs.name == "near_equilibrium") {
            for (std::size_t i = 0; i < phi.size(); ++i) {
                phi.data()[i] = -1.0 + 1e-6 * phi.data()[i];
                u.data()[i] = 0.0;
            }
        }
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        solver.mutable_params().dt = cfg.time.dt;
        const WillmoreSolveStats st = solve_willmore_step(
            {WillmoreDomain{&solver, 0, 1, cfg.grid.Nx - 1, 1}}, cfg, [](WillmoreBuffer) {});
        const std::vector<double> x = solver.willmore_download(V::X);
        // b again, as the solve builds it (omega first: the solve reused its
        // buffer for lap p; rhs_init zeroes x, so x was saved first), then A x.
        solver.willmore_omega();
        solver.willmore_rhs_init();
        const std::vector<double> b = solver.willmore_download(V::R);
        solver.willmore_upload(V::P, x);
        solver.willmore_direction_bc();
        solver.willmore_direction_laplacian();
        solver.willmore_apply();
        const std::vector<double> Ax = solver.willmore_download(V::AP);
        double rr = 0, bb = 0;
        for (int c : interior_cells(cfg)) {
            const auto i = static_cast<std::size_t>(c);
            rr += (static_cast<double>(b[i]) - Ax[i]) * (static_cast<double>(b[i]) - Ax[i]);
            bb += static_cast<double>(b[i]) * b[i];
        }
        const double true_rel = static_cast<double>(std::sqrt(rr / bb));
        std::printf("%-16s iterations %d  degree %d  ||b|| %.3e  true relative residual %.3e\n",
                    cs.name.c_str(), st.iterations, st.chebyshev_degree,
                    static_cast<double>(std::sqrt(bb)), true_rel);
        EXPECT_LE(true_rel, kWillmoreCgTolerance + st.iterations * 16 * kU64 * st.spectrum.kappa_A);
    }
}

// The bounds and the polynomial depend on dt, which adaptive stepping changes
// from step to step, and on the tau range of the state: every solve must
// rebuild them for its own dt and state (a plan kept from an earlier step
// would be a polynomial for another operator). Consecutive steps with
// dt = 0.004, 0.001, 0.004 report exactly the minimum and maximum of tau over
// the interior cells of the state each step starts from (computed here on the
// host from the downloaded tau), and the degree and bounds chebyshev_plan
// gives for that range and their dt.
TEST_F(WillmorePreconditionerTest, EveryStepUsesThePlanOfItsOwnTimeStep) {
    const SimulationConfig cfg =
        small_config(7, 6, 8, PerFaceBoundary::uniform(bc_of(BCType::Neumann)));
    const Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    random_state(phi, u, 9);
    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    const std::vector<int> cells = interior_cells(cfg);
    int previous_degree = 0;
    for (double dt : {0.004, 0.001, 0.004}) {
        // tau of the state this step starts from (the step recomputes it).
        solver.mutable_params().dt = dt;
        solver.willmore_omega();
        solver.willmore_rhs_init();
        const std::vector<double> tau = solver.willmore_download(V::Tau);
        double tau_min = std::numeric_limits<double>::infinity(), tau_max = 0.0;
        for (int c : cells) {
            tau_min = std::min(tau_min, tau[static_cast<std::size_t>(c)]);
            tau_max = std::max(tau_max, tau[static_cast<std::size_t>(c)]);
        }
        solver.step(dt);
        const WillmoreSolveStats& st = solver.last_willmore_stats();
        EXPECT_EQ(st.tau_min, tau_min) << "dt=" << dt;
        EXPECT_EQ(st.tau_max, tau_max) << "dt=" << dt;
        const WillmoreSpectrum expected =
            willmore_spectrum(cfg, cfg.boundary.phi_faces, dt, tau_min, tau_max);
        EXPECT_EQ(st.chebyshev_degree, chebyshev_plan(expected, kWillmoreCgTolerance).degree)
            << "dt=" << dt;
        EXPECT_EQ(st.spectrum.a, expected.a) << "dt=" << dt;
        EXPECT_EQ(st.spectrum.b, expected.b) << "dt=" << dt;
        EXPECT_NE(st.chebyshev_degree, previous_degree) << "dt=" << dt;
        previous_degree = st.chebyshev_degree;
    }
}

// A non-finite phase field must make the solve throw, never loop or return.
// A NaN cell reaches the right-hand side (its gradient is NaN, and
// compute_An treats a NaN gradient like a zero one, so tau stays finite); an
// infinite cell makes the gradient infinite and tau NaN (inf/inf in
// compute_An), which the bounds reject before any degree is sought.
TEST_F(WillmorePreconditionerTest, NonFiniteStateThrows) {
    const SimulationConfig cfg =
        small_config(7, 6, 8, PerFaceBoundary::uniform(bc_of(BCType::Neumann)));
    const Grid grid = cfg.make_grid();
    for (double bad :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        SCOPED_TRACE(bad);
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 1);
        phi(3, 3, 3) = bad;
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        EXPECT_THROW(solver.step(cfg.time.dt), std::runtime_error);
    }
    const PerFaceBoundary faces = PerFaceBoundary::uniform(bc_of(BCType::Neumann));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW((void)willmore_spectrum(cfg, faces, cfg.time.dt, 0.3, nan), std::runtime_error);
    EXPECT_THROW((void)willmore_spectrum(cfg, faces, cfg.time.dt, nan, 0.5), std::runtime_error);
    EXPECT_THROW((void)willmore_spectrum(cfg, faces, cfg.time.dt, 0.0, 0.5), std::runtime_error);
}

// A step that cannot proceed names its cause exactly. The right-hand side's
// norm: the state not finite, lambda u beyond the double range, another
// (phase-field) term beyond it, or finite components whose squares overflow
// (before, every case read "right-hand side is not finite"). The relaxation
// times: tau0 itself, or, for a valid configuration, a gradient whose fourth
// powers overflow A(n) (before: "non-finite phase field?", although a NaN
// phase field gives a finite A(n)). A dt for which tau/dt overflows is named.
TEST_F(WillmorePreconditionerTest, FailedStepNamesItsCause) {
    const SimulationConfig cfg =
        small_config(7, 6, 8, PerFaceBoundary::uniform(bc_of(BCType::Neumann)));
    const Grid grid = cfg.make_grid();
    struct Case {
        const char* what;
        double phi, u;
        const char* needle;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Case cases[] = {
        {"phi NaN", nan, 0.0, "the state itself is not finite"},
        {"u DBL_MAX", 0.0, std::numeric_limits<double>::max(),
         "lambda u, the coupling's factor, is beyond the double range"},
        {"phi 4e61", 4e61, 0.0, "a phase-field term of r"},
        {"phi 1e40", 1e40, 0.0, "every component is finite"},
        {"u 1e300", 0.0, 1e300, "every component is finite"},
        {"phi inf", std::numeric_limits<double>::infinity(), 0.0,
         "at some interior cell the central-difference gradient of phi has an infinite component"},
    };
    for (const Case& c : cases) {
        SCOPED_TRACE(c.what);
        FieldData phi(grid, "phi"), u(grid, "u");
        random_state(phi, u, 1);
        if (c.phi != 0.0 || std::isnan(c.phi))
            phi(3, 3, 3) = c.phi;
        if (c.u != 0.0)
            u(3, 3, 3) = c.u;
        CudaSolver solver(cfg);
        solver.initialize(phi, u);
        try {
            solver.step(cfg.time.dt);
            ADD_FAILURE() << "the step succeeded";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(c.needle), std::string::npos) << e.what();
        }
    }
    const PerFaceBoundary faces = PerFaceBoundary::uniform(bc_of(BCType::Neumann));
    auto message = [&](const SimulationConfig& config, double dt, double lo, double hi) {
        try {
            (void)willmore_spectrum(config, faces, dt, lo, hi);
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    EXPECT_NE(message(cfg, cfg.time.dt, 0.3, nan).find("the central-difference gradient"),
              std::string::npos);
    auto infinite_tau0 = cfg;
    infinite_tau0.physics.d0 = 5e-324;
    EXPECT_NE(message(infinite_tau0, cfg.time.dt, 0.3, nan).find("tau0 = inf is not finite"),
              std::string::npos);
    EXPECT_NE(message(cfg, 5e-324, 0.3, 0.5)
                  .find("tau/dt is beyond the double range at dt = 4.9406564584124654e-324"),
              std::string::npos);
}

// The single-precision kernels cover the interior with a (z, y) thread grid.
// One block row covers 8 Y rows and gridDim.y is limited to 65535, so above
// Ny - 2 = 8 * 65535 interior rows the Y rows are covered by a grid-stride
// loop. A pencil of 1 x (8 * 65535 + 1) x 1 interior cells along Y must give
// the same bits as its transpose along Z (gridDim.x covers z without that
// limit): with zero-flux faces every stencil term across the pencil is an
// exact zero, so both orientations do the same arithmetic cell by cell.
// One full step must agree within what the two solves' residuals allow:
// ||x_y - x_z|| <= (||r_y|| + ||r_z||) / lambda_min(A) <= 2 tol ||b|| /
// (tau_min/dt + beta L_min), the denominator a lower bound of A's spectrum.
TEST_F(WillmorePreconditionerTest, PencilBeyondTheGridYLimitMatchesItsTranspose) {
    const int L = 8 * 65535 + 3; // Ny - 2 = 524281 interior rows > 8 * 65535
    auto make = [&](bool along_y) {
        SimulationConfig cfg;
        cfg.grid.Nx = 3;
        cfg.grid.Ny = along_y ? L : 3;
        cfg.grid.Nz = along_y ? 3 : L;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.epsilon = 0.12;
        cfg.physics.W0 = 1.0;
        cfg.physics.willmore_beta = 0.25;
        cfg.time.dt = 0.005;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.phi_bc = bc_of(BCType::Neumann);
        cfg.boundary.u_bc = bc_of(BCType::Neumann);
        cfg.validate();
        return cfg;
    };
    const SimulationConfig cy = make(true), cz = make(false);
    // A front pair along the pencil (a solid slab of half-width 40 h centred
    // in the pencil), u = -0.55.
    auto value = [&](int i) {
        const double s = 0.4 * (i - 0.5 * (L - 1));
        return -std::tanh((std::fabs(s) - 16.0) / std::sqrt(2.0));
    };
    const Grid gy = cy.make_grid(), gz = cz.make_grid();
    FieldData phy(gy, "phi"), uy(gy, "u"), phz(gz, "phi"), uz(gz, "u");
    for (int i = 0; i < L; ++i)
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                phy(a, i, b) = value(i);
                phz(a, b, i) = value(i);
                uy(a, i, b) = -0.55;
                uz(a, b, i) = -0.55;
            }
    CudaSolver sy(cy), sz(cz);
    sy.initialize(phy, uy);
    sz.initialize(phz, uz);
    // Index of interior cell i of the pencil in each layout.
    auto iy = [&](int i) { return static_cast<std::size_t>((1 * L + i) * 3 + 1); };
    auto iz = [&](int i) { return static_cast<std::size_t>((1 * 3 + 1) * L + i); };

    std::vector<double> tau_y, tau_z, r_y, r_z;
    for (CudaSolver* s : {&sy, &sz}) {
        s->mutable_params().dt = cy.time.dt;
        s->willmore_omega();
        s->willmore_rhs_init();
    }
    tau_y = sy.willmore_download(V::Tau);
    tau_z = sz.willmore_download(V::Tau);
    r_y = sy.willmore_download(V::R);
    r_z = sz.willmore_download(V::R);
    double rr = 0.0;
    int tau_mismatch = 0, r_mismatch = 0;
    for (int i = 1; i < L - 1; ++i) {
        tau_mismatch += tau_y[iy(i)] != tau_z[iz(i)];
        r_mismatch += r_y[iy(i)] != r_z[iz(i)];
        rr += r_y[iy(i)] * r_y[iy(i)];
    }
    EXPECT_EQ(tau_mismatch, 0);
    EXPECT_EQ(r_mismatch, 0);
    ASSERT_GT(rr, 0.0);
    EXPECT_EQ(sy.willmore_tau_range(), sz.willmore_tau_range());

    const auto [tau_min, tau_max] = sy.willmore_tau_range();
    const WillmoreSpectrum sp =
        willmore_spectrum(cy, cy.boundary.phi_faces, cy.time.dt, tau_min, tau_max);
    const ChebyshevPlan plan = chebyshev_plan(sp, kWillmoreCgTolerance);
    apply_willmore_preconditioner({WillmoreDomain{&sy, 0, 1, cy.grid.Nx - 1, 1}}, plan, rr,
                                  [](WillmoreBuffer) {});
    apply_willmore_preconditioner({WillmoreDomain{&sz, 0, 1, cz.grid.Nx - 1, 1}}, plan, rr,
                                  [](WillmoreBuffer) {});
    const std::vector<double> z_y = sy.willmore_download(V::Z), z_z = sz.willmore_download(V::Z);
    int z_mismatch = 0;
    for (int i = 1; i < L - 1; ++i)
        z_mismatch += z_y[iy(i)] != z_z[iz(i)];
    EXPECT_EQ(z_mismatch, 0) << "degree " << plan.degree;

    sy.step(cy.time.dt);
    sz.step(cz.time.dt);
    const WillmoreSolveStats st_y = sy.last_willmore_stats(), st_z = sz.last_willmore_stats();
    EXPECT_LE(st_y.relative_residual, kWillmoreCgTolerance);
    EXPECT_LE(st_z.relative_residual, kWillmoreCgTolerance);
    const std::vector<double> x_y = sy.willmore_download(V::X), x_z = sz.willmore_download(V::X);
    double d2 = 0.0;
    for (int i = 1; i < L - 1; ++i)
        d2 += (x_y[iy(i)] - x_z[iz(i)]) * (x_y[iy(i)] - x_z[iz(i)]);
    const double lambda_min =
        st_y.tau_min / cy.time.dt + cy.physics.willmore_beta * st_y.spectrum.L_min;
    const double allowed = 2.0 * kWillmoreCgTolerance * std::sqrt(rr) / lambda_min;
    std::printf("pencil L=%d: degree %d, its %d/%d, |x_y - x_z| = %.3e (allowed %.3e)\n", L,
                plan.degree, st_y.iterations, st_z.iterations, std::sqrt(d2), allowed);
    EXPECT_LE(std::sqrt(d2), allowed);
}
