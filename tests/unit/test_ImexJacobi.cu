// The IMEX thermal solve (docs/THEORY.md 10.2): Jacobi sweeps on
// (I - dt D Lap_h) u = rhs that stop on a certified bound of the max-norm
// error, and fail with an error when the bound cannot be certified.
#include "common/Gpu.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/JacobiStopping.hpp"
#include "cuda/MultiGPUSolver.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ac;
using namespace ac::cuda;

namespace {

BoundaryConfig face(BCType type, double value = 0.0, double flux = 0.0, double alpha = 0.0,
                    double beta = 0.0, double gamma = 0.0) {
    BoundaryConfig b;
    b.type = type;
    b.value = value;
    b.flux = flux;
    b.alpha = alpha;
    b.beta = beta;
    b.gamma = gamma;
    return b;
}

/// u faces with every accepted ghost rule: Dirichlet, Robin (c = 0.38 at
/// h = 0.4), a periodic pair, Neumann with and without flux.
PerFaceBoundary u_faces() {
    PerFaceBoundary f;
    f.faces = {face(BCType::Dirichlet, -0.5),   face(BCType::Robin, 0.0, 0.0, 2.0, 0.5, -1.6),
               face(BCType::Periodic),          face(BCType::Periodic),
               face(BCType::Neumann, 0.0, 0.3), face(BCType::Neumann)};
    return f;
}

/// phi = -1 (liquid) with zero-flux phi faces is a fixed point of the phase
/// field update (no reaction, coupling or gradient term, and a zero Willmore
/// right-hand side, whose solve returns zero), so an IMEX step is exactly the
/// backward-Euler heat solve (I - dt D Lap_h) u_new = u_old. The Willmore
/// term (beta = 0.25, as shipped at eps = 0.12) raises the explicit
/// phase-field limit to 0.7147 on the anisotropic grid and 0.7161 on the
/// cubic one (docs/THEORY.md 10.1), above the dt of these tests; at
/// beta = 0.6 no mode of the liquid grows at any dt (limit +infinity).
SimulationConfig imex_config(StencilType stencil, double dt, int domains = 1, double beta = 0.25) {
    SimulationConfig cfg;
    cfg.grid.Nx = 20;
    cfg.grid.Ny = 18;
    cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4;
    // anisotropic spacing for the 7-point stencil; the 27-point needs cubic
    cfg.grid.dy = stencil == StencilType::Standard7Point ? 0.5 : 0.4;
    cfg.grid.dz = stencil == StencilType::Standard7Point ? 0.3 : 0.4;
    cfg.physics.epsilon = 0.05;
    cfg.physics.willmore_beta = beta;
    cfg.physics.D = 2.0;
    cfg.time.dt = dt;
    cfg.time.scheme = TimeScheme::IMEX;
    cfg.stencil = stencil;
    cfg.boundary.per_face = true;
    cfg.boundary.phi_faces = PerFaceBoundary::uniform(face(BCType::Neumann));
    cfg.boundary.u_faces = u_faces();
    cfg.gpu.multi_gpu = domains > 1;
    cfg.gpu.device_ids = std::vector<int>(static_cast<std::size_t>(domains), 0);
    cfg.validate();
    return cfg;
}

std::unique_ptr<ISolver> make_solver(const SimulationConfig& cfg) {
    if (cfg.gpu.multi_gpu)
        return std::make_unique<MultiGPUSolver>(cfg);
    return std::make_unique<CudaSolver>(cfg);
}

/// u_old: random in [-0.8, 0.2], so every mode of the error, the slowest
/// included, starts at a size of about 0.1.
void initial_state(FieldData& phi, FieldData& u) {
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> U(-0.8, 0.2);
    phi.fill(-1.0);
    for (std::size_t i = 0; i < u.size(); ++i)
        u.data()[i] = U(rng);
}

using LD = long double;

/// The exact discrete system in long double: the ghost fill of
/// apply_bc_face_kernel (faces Z, Y, X; Y faces skip the X walls, Z faces
/// skip the X and Y walls) and the stencil of jacobi_step_kernel.
struct HostSystem {
    int N[3];
    double h[3];
    PerFaceBoundary faces;
    bool iso27;
    LD aD;

    [[nodiscard]] std::size_t at(int x, int y, int z) const {
        return (static_cast<std::size_t>(x) * N[1] + y) * N[2] + z;
    }

    void fill(std::vector<LD>& f) const {
        for (int axis = 2; axis >= 0; --axis) {
            const int a1 = axis == 0 ? 1 : 0, a2 = axis == 2 ? 1 : 2;
            for (int side = 0; side < 2; ++side) {
                const BoundaryConfig& bc = faces.get(axis, side);
                for (int i = 0; i < N[a1]; ++i)
                    for (int j = 0; j < N[a2]; ++j) {
                        int c[3];
                        c[axis] = side ? N[axis] - 1 : 0;
                        c[a1] = i;
                        c[a2] = j;
                        if (axis >= 1 && (c[0] == 0 || c[0] == N[0] - 1))
                            continue;
                        if (axis == 2 && (c[1] == 0 || c[1] == N[1] - 1))
                            continue;
                        int in[3] = {c[0], c[1], c[2]};
                        in[axis] = side ? N[axis] - 2 : 1;
                        const LD inner = f[at(in[0], in[1], in[2])];
                        const LD ds = h[axis];
                        LD& g = f[at(c[0], c[1], c[2])];
                        switch (bc.type) {
                        case BCType::Dirichlet:
                            g = bc.value;
                            break;
                        case BCType::Neumann:
                            g = inner + LD(bc.flux) * ds;
                            break;
                        case BCType::Periodic: {
                            int src[3] = {c[0], c[1], c[2]};
                            src[axis] = side ? 1 : N[axis] - 2;
                            g = f[at(src[0], src[1], src[2])];
                            break;
                        }
                        case BCType::Robin:
                            g = (LD(bc.gamma) + LD(bc.beta) * inner / ds) /
                                (LD(bc.alpha) + LD(bc.beta) / ds);
                            break;
                        }
                    }
            }
        }
    }

    /// s: the diagonal of the sweep is 1 + s.
    [[nodiscard]] LD s() const {
        if (iso27)
            return 128 * aD / (30 * LD(h[0]) * LD(h[0]));
        return 2 * aD * (1 / (LD(h[0]) * h[0]) + 1 / (LD(h[1]) * h[1]) + 1 / (LD(h[2]) * h[2]));
    }

    void sweep(const std::vector<LD>& v, std::vector<LD>& out, const std::vector<LD>& rhs) const {
        const LD diag = 1 + s();
        for (int x = 1; x < N[0] - 1; ++x)
            for (int y = 1; y < N[1] - 1; ++y)
                for (int z = 1; z < N[2] - 1; ++z) {
                    LD off = 0;
                    if (iso27) {
                        const LD coeff = aD / (30 * LD(h[0]) * h[0]);
                        for (int dx = -1; dx <= 1; ++dx)
                            for (int dy = -1; dy <= 1; ++dy)
                                for (int dz = -1; dz <= 1; ++dz) {
                                    const int k = std::abs(dx) + std::abs(dy) + std::abs(dz);
                                    if (k == 0)
                                        continue;
                                    off += coeff *
                                           (k == 1   ? 14
                                            : k == 2 ? 3
                                                     : 1) *
                                           v[at(x + dx, y + dy, z + dz)];
                                }
                    } else {
                        off = aD * ((v[at(x + 1, y, z)] + v[at(x - 1, y, z)]) / (LD(h[0]) * h[0]) +
                                    (v[at(x, y + 1, z)] + v[at(x, y - 1, z)]) / (LD(h[1]) * h[1]) +
                                    (v[at(x, y, z + 1)] + v[at(x, y, z - 1)]) / (LD(h[2]) * h[2]));
                    }
                    out[at(x, y, z)] = (rhs[at(x, y, z)] + off) / diag;
                }
        fill(out);
    }
};

struct Reference {
    std::vector<LD> u;
    LD error_bound; ///< of |u - u*| (u* the exact solution)
};

/// Solves the system to long-double accuracy by Jacobi sweeps from u_old,
/// enough of them that q^k (q = s/(1+s), the contraction of a sweep in the
/// max norm) is below 1e-24, then bounds the error a posteriori like the
/// solver does (s times the last change, plus rounding of the same form
/// with the long-double unit roundoff).
Reference reference_solution(const SimulationConfig& cfg, const FieldData& u_old) {
    HostSystem sys{{cfg.grid.Nx, cfg.grid.Ny, cfg.grid.Nz},
                   {cfg.grid.dx, cfg.grid.dy, cfg.grid.dz},
                   cfg.boundary.u_faces,
                   cfg.stencil == StencilType::Isotropic27Point,
                   LD(cfg.time.dt) * LD(cfg.physics.D)};
    std::vector<LD> rhs(u_old.data(), u_old.data() + u_old.size());
    sys.fill(rhs); // u_old's ghosts as initialize() leaves them (unused by the sweep)
    std::vector<LD> a = rhs, b = rhs;
    const LD s = sys.s(), q = s / (1 + s);
    const int sweeps = static_cast<int>(std::ceil(std::log(1e-24L) / std::log(q))) + 1;
    LD change = 0, magnitude = 0;
    for (int k = 0; k < sweeps; ++k) {
        sys.sweep(a, b, rhs);
        change = 0;
        magnitude = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            change = std::max(change, std::fabs(b[i] - a[i]));
            magnitude = std::max({magnitude, std::fabs(a[i]), std::fabs(b[i])});
        }
        std::swap(a, b);
    }
    const LD u_ld = LDBL_EPSILON / 2;
    const LD gamma = 80 * u_ld / (1 - 80 * u_ld);
    const LD bound = (s * change + gamma * (1 + 3 * s) * magnitude) / (1 - gamma * (1 + 2 * s));
    return {a, bound};
}

class ImexJacobiTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// One IMEX step at dt*D/h^2 far beyond the explicit limit (s = 17.1 for the
/// 7-point and 16 for the 27-point stencil): the result must lie within
/// kJacobiTol of the exact solution in every interior cell, on one GPU and
/// on three domains.
void expect_certified_error(StencilType stencil, double dt, int domains) {
    const SimulationConfig cfg = imex_config(stencil, dt, domains);
    const Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    initial_state(phi, u);
    const Reference ref = reference_solution(cfg, u);

    auto solver = make_solver(cfg);
    solver->initialize(phi, u);
    solver->step(dt);
    solver->copy_u_to_host(u);
    solver->copy_phi_to_host(phi);

    double err = 0.0;
    for (int x = 1; x < cfg.grid.Nx - 1; ++x)
        for (int y = 1; y < cfg.grid.Ny - 1; ++y)
            for (int z = 1; z < cfg.grid.Nz - 1; ++z) {
                const std::size_t i =
                    (static_cast<std::size_t>(x) * cfg.grid.Ny + y) * cfg.grid.Nz + z;
                err = std::max(err, static_cast<double>(std::fabs(u.data()[i] - ref.u[i])));
                ASSERT_EQ(phi.data()[i], -1.0) << "phi must stay the liquid fixed point";
            }
    const double allowed = CudaSolver::kJacobiTol + static_cast<double>(ref.error_bound);
    std::printf("%s dt %.2f domains %d: max |u - u_ref| = %.3e (allowed %.3e, reference error "
                "<= %.1e)\n",
                stencil == StencilType::Standard7Point ? "7-point " : "27-point", dt, domains, err,
                allowed, static_cast<double>(ref.error_bound));
    EXPECT_LE(err, allowed);
}

/// The step must throw a runtime_error whose message contains `needle`.
void expect_step_throws(const SimulationConfig& cfg, FieldData& phi, FieldData& u,
                        const std::string& needle) {
    auto solver = make_solver(cfg);
    solver->initialize(phi, u);
    try {
        solver->step(cfg.time.dt);
        solver->synchronize();
        ADD_FAILURE() << "the IMEX step returned";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

} // namespace

TEST_F(ImexJacobiTest, ErrorIsWithinTheToleranceAtLargeDt) {
    for (int domains : {1, 3}) {
        expect_certified_error(StencilType::Standard7Point, 0.2, domains);
        expect_certified_error(StencilType::Isotropic27Point, 0.3, domains);
    }
}

// A NaN in the temperature is not a converged solve: the sweeps carry it,
// and the step must fail instead of returning the iterate. Without the
// Willmore term (beta = 0, at dt 0.005 below the explicit phase-field limit
// 0.0093 of this grid), so that the Jacobi rule, not the Willmore solve,
// meets the NaN first.
TEST_F(ImexJacobiTest, NonFiniteTemperatureThrows) {
    for (int domains : {1, 3}) {
        SimulationConfig cfg = imex_config(StencilType::Standard7Point, 0.005, domains);
        cfg.physics.willmore_beta = 0.0;
        cfg.validate();
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        initial_state(phi, u);
        u(9, 9, 8) = std::numeric_limits<double>::quiet_NaN();
        expect_step_throws(cfg, phi, u, "IMEX thermal solve: the Jacobi iterate is not finite");
    }
}

// At dt = 1e4 (s = 8.5e5) the rounding of one sweep, amplified by the
// conditioning of the system, exceeds kJacobiTol: no number of sweeps can
// certify the tolerance, and the step must say so (beta = 0.6: no
// phase-field limit, so this dt is accepted).
TEST_F(ImexJacobiTest, ToleranceBeyondRoundingThrows) {
    for (int domains : {1, 3}) {
        const SimulationConfig cfg = imex_config(StencilType::Standard7Point, 1e4, domains, 0.6);
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        initial_state(phi, u);
        expect_step_throws(cfg, phi, u, "rounding bound");
    }
}

// The multi-GPU solver must stop where one GPU stops, on the maxima over
// every domain. A perturbation in the first domain (x <= 4 of 20, three
// domains) with zero-flux faces and a Dirichlet face at x_hi: the slowest
// error mode peaks at x_lo and vanishes at x_hi, so the last domain alone
// sees a smaller change than the whole grid. Both results must be within
// kJacobiTol of the exact solution and bitwise equal.
TEST_F(ImexJacobiTest, DomainsStopWhereOneGpuStops) {
    auto config = [](int domains) {
        SimulationConfig cfg = imex_config(StencilType::Standard7Point, 0.2, domains);
        cfg.boundary.u_faces = PerFaceBoundary::uniform(face(BCType::Neumann));
        cfg.boundary.u_faces[Face::XHi] = face(BCType::Dirichlet, -0.8);
        cfg.validate();
        return cfg;
    };
    const SimulationConfig one = config(1), three = config(3);
    const Grid grid = one.make_grid();
    FieldData phi(grid, "phi"), u0(grid, "u");
    phi.fill(-1.0);
    u0.fill(-0.8);
    for (int x = 1; x <= 4; ++x)
        for (int y = 0; y < one.grid.Ny; ++y)
            for (int z = 0; z < one.grid.Nz; ++z)
                u0(x, y, z) = -0.3;
    const Reference ref = reference_solution(one, u0);
    std::vector<double> results[2];
    int k = 0;
    for (const SimulationConfig* cfg : {&one, &three}) {
        FieldData u = u0, p = phi;
        auto solver = make_solver(*cfg);
        solver->initialize(p, u);
        solver->step(cfg->time.dt);
        solver->copy_u_to_host(u);
        double err = 0.0;
        for (int x = 1; x < cfg->grid.Nx - 1; ++x)
            for (int y = 1; y < cfg->grid.Ny - 1; ++y)
                for (int z = 1; z < cfg->grid.Nz - 1; ++z) {
                    const std::size_t i =
                        (static_cast<std::size_t>(x) * cfg->grid.Ny + y) * cfg->grid.Nz + z;
                    err = std::max(err, static_cast<double>(std::fabs(u.data()[i] - ref.u[i])));
                }
        EXPECT_LE(err, CudaSolver::kJacobiTol + static_cast<double>(ref.error_bound))
            << (k ? "three domains" : "one GPU");
        results[k++].assign(u.data(), u.data() + u.size());
    }
    std::size_t mismatches = 0;
    for (std::size_t i = 0; i < results[0].size(); ++i)
        mismatches += results[0][i] != results[1][i] ? 1 : 0;
    EXPECT_EQ(mismatches, 0U);
}

// ── Host: the stopping rule ────────────────────────────────────────────────

namespace {

double gamma_of(int k) {
    const double ku = k * (std::numeric_limits<double>::epsilon() / 2.0);
    return ku / (1.0 - ku);
}

/// rule.converged must throw a runtime_error whose message contains `needle`.
void expect_rule_throws(JacobiStoppingRule& rule, double change, double magnitude,
                        const std::string& needle) {
    try {
        (void)rule.converged(10, {change, magnitude});
        ADD_FAILURE() << "accepted change " << change << ", magnitude " << magnitude;
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

} // namespace

// bound(d, m) = (s d + gamma_K (1 + 3s) m) / (1 - gamma_K (1 + 2s)), and the
// rule accepts exactly when it is <= the tolerance.
TEST(JacobiStoppingRule, AcceptsExactlyWhenTheBoundMeetsTheTolerance) {
    const double s = 3.0, tol = 1e-10, m = 0.9;
    JacobiStoppingRule rule(s, 35, tol);
    const double g = gamma_of(35);
    for (double d : {0.0, 1e-12, 2e-11, 1e-6}) {
        const double expected = (s * d + g * (1 + 3 * s) * m) / (1 - g * (1 + 2 * s));
        EXPECT_EQ(rule.bound(d, m), expected) << "d = " << d;
    }
    // The change at which the bound equals the tolerance, and a change one
    // part in 1e9 on either side of it.
    const double d_star = (tol * (1 - g * (1 + 2 * s)) - g * (1 + 3 * s) * m) / s;
    JacobiStoppingRule above(s, 35, tol), below(s, 35, tol);
    EXPECT_FALSE(above.converged(10, {d_star * (1 + 1e-9), m}));
    EXPECT_TRUE(below.converged(10, {d_star * (1 - 1e-9), m}));
    // The 27-point rounding count enters through gamma_80.
    JacobiStoppingRule rule27(s, 80, tol);
    EXPECT_EQ(rule27.bound(0.0, m),
              gamma_of(80) * (1 + 3 * s) * m / (1 - gamma_of(80) * (1 + 2 * s)));
}

TEST(JacobiStoppingRule, NonFiniteChangeOrMagnitudeThrows) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto& [d, m] :
         {std::pair{nan, 1.0}, std::pair{inf, 1.0}, std::pair{1e-3, nan}, std::pair{1e-3, inf}}) {
        JacobiStoppingRule rule(3.0, 35, 1e-10);
        expect_rule_throws(rule, d, m, "not finite");
    }
}

// When the rounding term alone reaches the tolerance, even a zero change is
// not accepted; also when 1 - gamma_K (1 + 2s) <= 0 (the bound is infinite).
TEST(JacobiStoppingRule, RoundingAtTheToleranceThrows) {
    const double tol = 1e-10, m = 1.0, g = gamma_of(35);
    // s with gamma (1 + 3s) m / (1 - gamma (1 + 2s)) = 2 tol
    const double s = (2 * tol - g * m - 2 * tol * g) / (3 * g * m + 4 * tol * g);
    JacobiStoppingRule rule(s, 35, tol);
    EXPECT_GE(rule.bound(0.0, m), tol);
    expect_rule_throws(rule, 0.0, m, "rounding bound");
    JacobiStoppingRule huge(1e17, 35, tol);
    EXPECT_EQ(huge.bound(0.0, 0.0), std::numeric_limits<double>::infinity());
    expect_rule_throws(huge, 0.0, 0.0, "rounding bound");
}

// When the rounding term exceeds the tolerance, the message tells whether a
// smaller dt can certify the solve: as dt -> 0 the term falls to gamma m /
// (1 - gamma), which at m above tol (1 - gamma) / gamma (25735 for 7 points)
// exceeds the tolerance at every dt, and the old advice "reduce dt" was wrong
// (record 346 of the fuzz campaign, max |u| = 8.47e15 from delta 1e16).
// Below it the message gives the factor s / s_max that dt must at least shrink by.
TEST(JacobiStoppingRule, RoundingMessageSaysWhetherASmallerStepCanCertify) {
    const double tol = 1e-10, g = gamma_of(35);
    const double m_edge = tol * (1 - g) / g;
    JacobiStoppingRule hopeless(0.43, 35, tol);
    try {
        (void)hopeless.converged(10, {0.0, 8471519747172330.0});
        ADD_FAILURE() << "accepted";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("no time step certifies it"), std::string::npos) << what;
        EXPECT_NE(what.find("max |u| must stay below 2.573e+04"), std::string::npos) << what;
        EXPECT_EQ(what.find("reduce dt"), std::string::npos) << what;
        EXPECT_EQ(what.find("smaller by a factor"), std::string::npos) << what;
    }
    // m below the edge: a smaller s certifies; s_max solves bound(0, m) = tol.
    const double m = 0.5 * m_edge;
    const double s_max = (tol * (1 - g) - g * m) / (g * (3 * m + 2 * tol));
    JacobiStoppingRule at_this_dt(4 * s_max, 35, tol);
    try {
        (void)at_this_dt.converged(10, {0.0, m});
        ADD_FAILURE() << "accepted";
    } catch (const std::runtime_error& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("smaller by a factor above s / s_max = 4.000e+00"), std::string::npos)
            << what;
        EXPECT_EQ(what.find("no time step certifies it"), std::string::npos) << what;
    }
    JacobiStoppingRule fine(0.99 * s_max, 35, tol);
    EXPECT_LT(fine.bound(0.0, m), tol);
}

// In exact arithmetic the change shrinks by at least the factor s/(1+s) per
// sweep; a change that is not smaller than at the previous check ends the
// solve with an error.
TEST(JacobiStoppingRule, ChangeThatStopsDecreasingThrows) {
    JacobiStoppingRule rule(3.0, 35, 1e-10);
    EXPECT_FALSE(rule.converged(10, {1e-3, 1.0}));
    EXPECT_FALSE(rule.converged(20, {5e-4, 1.0}));
    expect_rule_throws(rule, 5e-4, 1.0, "stopped decreasing");
    JacobiStoppingRule growing(3.0, 35, 1e-10);
    EXPECT_FALSE(growing.converged(10, {1e-3, 1.0}));
    expect_rule_throws(growing, 2e-3, 1.0, "stopped decreasing");
}

TEST(JacobiStoppingRule, RejectsInvalidParameters) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(JacobiStoppingRule(-1.0, 35, 1e-10), std::invalid_argument);
    EXPECT_THROW(JacobiStoppingRule(nan, 35, 1e-10), std::invalid_argument);
    EXPECT_THROW(JacobiStoppingRule(std::numeric_limits<double>::infinity(), 35, 1e-10),
                 std::invalid_argument);
    EXPECT_THROW(JacobiStoppingRule(3.0, 0, 1e-10), std::invalid_argument);
    EXPECT_THROW(JacobiStoppingRule(3.0, 35, 0.0), std::invalid_argument);
}

// The multi-GPU solver merges the domains' change and magnitude with max_of:
// the larger of each, and NaN from either side.
TEST(JacobiStoppingRule, MaxOfTakesTheLargerOfEachAndPropagatesNaN) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const JacobiChange a{1e-3, 2.0}, b{2e-3, 1.0};
    const JacobiChange ab = max_of(a, b), ba = max_of(b, a);
    EXPECT_EQ(ab.change, 2e-3);
    EXPECT_EQ(ab.magnitude, 2.0);
    EXPECT_EQ(ba.change, 2e-3);
    EXPECT_EQ(ba.magnitude, 2.0);
    for (const auto& [x, y] :
         {std::pair{JacobiChange{nan, 1.0}, b}, std::pair{b, JacobiChange{nan, 1.0}}}) {
        EXPECT_TRUE(std::isnan(max_of(x, y).change));
        EXPECT_EQ(max_of(x, y).magnitude, 1.0);
    }
    for (const auto& [x, y] :
         {std::pair{JacobiChange{1.0, nan}, b}, std::pair{b, JacobiChange{1.0, nan}}}) {
        EXPECT_EQ(max_of(x, y).change, 1.0);
        EXPECT_TRUE(std::isnan(max_of(x, y).magnitude));
    }
}
