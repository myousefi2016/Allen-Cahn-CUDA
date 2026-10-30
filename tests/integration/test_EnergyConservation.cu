// Energy budget of the coupled solver.
//
// The thermal equation du/dt = D lap u + (1/2) dphi/dt (docs/THEORY.md 3.2) conserves the
// enthalpy u - phi/2 up to the diffusive flux through the walls, and so do the discrete
// updates. Summed over the interior cells, the Euler update
//   u1 = u0 + (phi1 - phi0)/2 + dt D lap_h u0
// changes the sum of u - phi/2 by dt D sum(lap_h u0), and the 7-point Laplacian telescopes:
// along each line of interior cells only (ghost - adjacent interior cell) / h^2 at its two
// walls remains. A zero-flux wall copies the adjacent cell into the ghost (no flux); a
// Dirichlet wall conducts. Every step is checked against that budget within a bound on its
// rounding: a product of at most k factors (1 + delta)^(+-1), |delta| <= u (the unit
// roundoff), lies within gamma_k = k u / (1 - k u) of 1 (Higham, Accuracy and Stability of
// Numerical Algorithms, 2nd ed., Lemma 3.1), so a computed sum of terms that each pass
// through at most k roundings is within gamma_k times the sum of their magnitudes.
//
// The free energy of docs/THEORY.md 2 is not asserted to decrease: for the coupling
// g = u G(phi), G' = (1 - phi^2)^2, and the latent heat (1/2) dphi/dt, along the equations
// dF/dt = -int tau (dphi/dt)^2 + lambda int G(phi) du/dt, whose second term has no sign.
#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <utility>

using namespace ac;
using namespace ac::cuda;

namespace {

using LD = long double;

/// Unit roundoffs of the device's double arithmetic and of the host's long-double sums
/// (the same where long double is double).
constexpr LD kUnitDevice = static_cast<LD>(DBL_EPSILON) / 2;
constexpr LD kUnitHost = LDBL_EPSILON / 2;

/// gamma_k = k u / (1 - k u).
LD gamma_k(LD k, LD unit) {
    return k * unit / (1 - k * unit);
}

BoundaryConfig dirichlet(double value) {
    return {BCType::Dirichlet, value, 0.0, 0.0, 0.0, 0.0};
}

BoundaryConfig zero_flux() {
    return {BCType::Neumann, 0.0, 0.0, 0.0, 0.0, 0.0};
}

/// 16^3 cells of h = 0.4 with the default physics (eps 0.05, W0 1, D 2, d0 0.5,
/// delta 0.8), the 7-point stencil, phi held at the liquid -1 on every wall, and
/// dt = 0.005, inside the explicit limits validate() checks (docs/THEORY.md 10.1).
SimulationConfig seed_config(TimeScheme scheme, const BoundaryConfig& u_bc) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 16;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
    cfg.time.dt = 0.005;
    cfg.time.scheme = scheme;
    cfg.stencil = StencilType::Standard7Point;
    cfg.physics.delta = 0.8;
    cfg.boundary.phi_bc = dirichlet(-1.0);
    cfg.boundary.u_bc = u_bc;
    cfg.validate();
    return cfg;
}

/// A solid ball of radius 3 cells at the centre (phi = 1, u = 0, 93 cells) in the
/// undercooled melt (phi = -1, u = -delta).
void seed(const SimulationConfig& cfg, FieldData& phi, FieldData& u) {
    const double cx = 0.5 * cfg.grid.Nx, cy = 0.5 * cfg.grid.Ny, cz = 0.5 * cfg.grid.Nz;
    for (int x = 0; x < cfg.grid.Nx; ++x)
        for (int y = 0; y < cfg.grid.Ny; ++y)
            for (int z = 0; z < cfg.grid.Nz; ++z) {
                const double r =
                    std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
                phi(x, y, z) = r < 3.0 ? 1.0 : -1.0;
                u(x, y, z) = r < 3.0 ? 0.0 : -cfg.physics.delta;
            }
}

/// Calls f(ghost, adjacent) for every wall cell the 7-point stencil of an interior cell
/// reads, with the interior cell next to it: the two ends of every line of interior cells.
template <typename F> void for_each_wall_pair(const FieldData& u, F&& f) {
    const int Nx = u.Nx(), Ny = u.Ny(), Nz = u.Nz();
    for (int y = 1; y < Ny - 1; ++y)
        for (int z = 1; z < Nz - 1; ++z) {
            f(u(0, y, z), u(1, y, z));
            f(u(Nx - 1, y, z), u(Nx - 2, y, z));
        }
    for (int x = 1; x < Nx - 1; ++x)
        for (int z = 1; z < Nz - 1; ++z) {
            f(u(x, 0, z), u(x, 1, z));
            f(u(x, Ny - 1, z), u(x, Ny - 2, z));
        }
    for (int x = 1; x < Nx - 1; ++x)
        for (int y = 1; y < Ny - 1; ++y) {
            f(u(x, y, 0), u(x, y, 1));
            f(u(x, y, Nz - 1), u(x, y, Nz - 2));
        }
}

/// Wall cells read by the stencil that do not hold u's rule exactly: the Dirichlet value,
/// or, zero flux, the adjacent cell (the fill inner + 0 * h is the inner value exactly).
int wall_mismatches(const BoundaryConfig& bc, const FieldData& u) {
    int bad = 0;
    for_each_wall_pair(u, [&](double ghost, double adjacent) {
        if (ghost != (bc.type == BCType::Dirichlet ? bc.value : adjacent))
            ++bad;
    });
    return bad;
}

bool all_finite(const FieldData& f) {
    return std::all_of(f.data(), f.data() + f.size(), [](double v) { return std::isfinite(v); });
}

/// Interior sums of one step (phi0, u0) -> (phi1, u1), accumulated in long double, with
/// c = dt D / h^2 (cubic grid) and s = 6 c (the 7-point Jacobi sweep's diagonal minus 1).
struct StepSums {
    LD change = 0;     ///< sum of (u1 - u0) - (phi1 - phi0)/2: the enthalpy change
    LD latent = 0;     ///< sum of (phi1 - phi0)/2: the latent heat released
    LD flux = 0;       ///< dt D sum(lap_h u0) = c * sum over the walls of (ghost - adjacent)
    LD change_mag = 0; ///< sum of |u1 - u0| + |phi1 - phi0|/2
    LD flux_mag = 0;   ///< c * sum over the walls of |ghost - adjacent|
    LD rhs_mag = 0;    ///< sum of a = |u0| + |phi1 - phi0|/2
    LD euler_mag = 0;  ///< sum of a + c M, M = sum of |u0| over the 6 neighbours + 6 |u0|
    LD imex_mag = 0;   ///< sum of a + c (sum of |u1| over the 6 neighbours) + (1 + s) |u1|
    std::size_t cells = 0, wall_pairs = 0;
};

StepSums step_sums(const SimulationConfig& cfg, const FieldData& phi0, const FieldData& u0,
                   const FieldData& phi1, const FieldData& u1) {
    const LD h = cfg.grid.dx;
    const LD c = LD(cfg.time.dt) * LD(cfg.physics.D) / (h * h);
    const LD s = 6 * c;
    auto mag = [](const FieldData& f, int x, int y, int z) { return std::fabs(LD(f(x, y, z))); };
    auto neighbours = [&](const FieldData& f, int x, int y, int z) {
        return mag(f, x + 1, y, z) + mag(f, x - 1, y, z) + mag(f, x, y + 1, z) +
               mag(f, x, y - 1, z) + mag(f, x, y, z + 1) + mag(f, x, y, z - 1);
    };
    StepSums r;
    for (int x = 1; x < cfg.grid.Nx - 1; ++x)
        for (int y = 1; y < cfg.grid.Ny - 1; ++y)
            for (int z = 1; z < cfg.grid.Nz - 1; ++z) {
                const LD du = LD(u1(x, y, z)) - LD(u0(x, y, z));
                const LD half_dphi = (LD(phi1(x, y, z)) - LD(phi0(x, y, z))) / 2;
                const LD a = mag(u0, x, y, z) + std::fabs(half_dphi);
                r.change += du - half_dphi;
                r.latent += half_dphi;
                r.change_mag += std::fabs(du) + std::fabs(half_dphi);
                r.rhs_mag += a;
                r.euler_mag += a + c * (neighbours(u0, x, y, z) + 6 * mag(u0, x, y, z));
                r.imex_mag += a + c * neighbours(u1, x, y, z) + (1 + s) * mag(u1, x, y, z);
                ++r.cells;
            }
    LD walls = 0, walls_mag = 0;
    for_each_wall_pair(u0, [&](double ghost, double adjacent) {
        const LD d = LD(ghost) - LD(adjacent);
        walls += d;
        walls_mag += std::fabs(d);
        ++r.wall_pairs;
    });
    r.flux = c * walls;
    r.flux_mag = c * walls_mag;
    return r;
}

/// The host's share of a bound: the enthalpy change is a long-double sum of `cells`
/// terms of at most 3 roundings, so each passes through at most cells + 2; the flux a
/// sum of `wall_pairs` differences times c (3 roundings), at most wall_pairs + 4. The
/// bound itself is a long-double sum of non-negative terms (at most cells + 32
/// roundings), so the computed value may fall short of the exact one by the relative
/// gamma_{cells+32}; the factor 1 + 4 gamma_{cells+32} covers that and its own rounding.
LD with_host_rounding(const StepSums& s, LD device) {
    const LD n = static_cast<LD>(s.cells), p = static_cast<LD>(s.wall_pairs);
    return (device + gamma_k(n + 2, kUnitHost) * s.change_mag +
            gamma_k(p + 4, kUnitHost) * s.flux_mag) *
           (1 + 4 * gamma_k(n + 32, kUnitHost));
}

/// Euler (thermal_equation_kernel, 7-point): 23 floating-point operations per cell
/// (laplacian_7pt 17: per axis an add, a doubling, a subtraction, dx * dx and a
/// division, then two adds; the latent heat 2; the update 4: two adds, dt * D and its
/// product with the Laplacian), so each term of u0 + (phi1 - phi0)/2 + dt D lap_h u0
/// passes through at most 23 roundings: the computed u1 is within gamma_23 (a + c M) of
/// the exact update of the stored u0, phi0, phi1. Zero-flux and Dirichlet ghosts are
/// stored exactly, so sum(lap_h u0) is the telescoped flux exactly.
LD euler_bound(const StepSums& s) {
    return with_host_rounding(s, gamma_k(23, kUnitDevice) * s.euler_mag);
}

/// IMEX with zero-flux u walls (jacobi_step_kernel, 7-point). rhs = u0 + 0.5 (phi1 - phi0)
/// (add_latent_heat_kernel, 2 roundings) is within gamma_2 a of its exact value. The last
/// sweep, u_K from u_{K-1}, is in exact arithmetic
///   u_K = rhs + dt D lap_h u_{K-1} + s (u_{K-1} - u_K) + r,
/// and sum(lap_h u_{K-1}) = 0 (zero-flux ghosts on every iterate). The step stops when the
/// computed JacobiStoppingRule::bound(d~, m) <= kJacobiTol, with d~ the reduced max
/// |u_K - u_{K-1}| (one rounding) and s~ the rule's s (at least s (1 - gamma_7)). That
/// bound's other term is not negative, so the sum is at least fl(s~ d~), and its rounded
/// denominator 1 - gamma_35 (1 + 2 s~) is at least 34 u below 1, more than the roundings of
/// fl(s~ d~) and of the division take away: s~ d~ <= kJacobiTol, and s d <= kJacobiTol /
/// ((1 - gamma_7) (1 - u)) <= kJacobiTol (1 + gamma_9) per cell. The sweep performs 24
/// operations per cell, so |r| <= gamma_24 (|rhs| + c sum_nb |u_{K-1}| + (1 + s) |u_K|),
/// with |rhs| <= (1 + gamma_2) a and |u_{K-1}| <= |u_K| + d: in all at most
/// gamma_26 (imex_mag + N kJacobiTol (1 + gamma_9)).
LD imex_bound(const StepSums& s) {
    const LD iteration =
        static_cast<LD>(s.cells) * CudaSolver::kJacobiTol * (1 + gamma_k(9, kUnitDevice));
    return with_host_rounding(s, iteration + gamma_k(26, kUnitDevice) * (s.imex_mag + iteration) +
                                     gamma_k(2, kUnitDevice) * s.rhs_mag);
}

/// The largest |change - flux|, |latent| and |flux| over the steps of a run, each divided
/// by its step's bound.
struct BudgetRun {
    LD error_over_bound = 0;
    LD latent_over_bound = 0;
    LD flux_over_bound = 0;
};

/// Runs `steps` steps from the seed and checks each against its budget: u's walls hold
/// their rule exactly, and |change - flux| <= bound(sums).
template <typename Bound>
void check_enthalpy_budget(const char* what, const SimulationConfig& cfg, int steps, Bound bound,
                           BudgetRun& run) {
    const BoundaryConfig& bc = cfg.boundary.u_bc;
    ASSERT_TRUE(bc.type == BCType::Dirichlet || (bc.type == BCType::Neumann && bc.flux == 0.0))
        << "the wall check knows Dirichlet and zero-flux walls";
    const Grid grid = cfg.make_grid();
    FieldData phi0(grid, "phi0"), u0(grid, "u0"), phi1(grid, "phi1"), u1(grid, "u1");
    seed(cfg, phi0, u0);
    CudaSolver solver(cfg);
    solver.initialize(phi0, u0);
    // The first step starts from the solver's state: initialize() applies the BCs.
    solver.copy_phi_to_host(phi0);
    solver.copy_u_to_host(u0);
    ASSERT_EQ(wall_mismatches(bc, u0), 0) << "after initialize()";
    for (int n = 1; n <= steps; ++n) {
        solver.step(cfg.time.dt);
        solver.copy_phi_to_host(phi1);
        solver.copy_u_to_host(u1);
        ASSERT_TRUE(all_finite(phi1) && all_finite(u1)) << "step " << n;
        ASSERT_EQ(wall_mismatches(bc, u1), 0) << "step " << n;
        const StepSums s = step_sums(cfg, phi0, u0, phi1, u1);
        const LD b = bound(s);
        EXPECT_LE(std::fabs(s.change - s.flux), b)
            << "step " << n << ": enthalpy change " << static_cast<double>(s.change)
            << ", wall flux " << static_cast<double>(s.flux) << ", latent heat "
            << static_cast<double>(s.latent) << ", bound " << static_cast<double>(b);
        run.error_over_bound = std::max(run.error_over_bound, std::fabs(s.change - s.flux) / b);
        run.latent_over_bound = std::max(run.latent_over_bound, std::fabs(s.latent) / b);
        run.flux_over_bound = std::max(run.flux_over_bound, std::fabs(s.flux) / b);
        std::swap(phi0, phi1);
        std::swap(u0, u1);
    }
    std::printf("%s: max |change - flux| / bound %.3e, |latent| / bound %.3e, |flux| / bound "
                "%.3e over %d steps\n",
                what, static_cast<double>(run.error_over_bound),
                static_cast<double>(run.latent_over_bound),
                static_cast<double>(run.flux_over_bound), steps);
}

} // namespace

class EnergyConservationTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

/// phi = -1, u = -0.8 with Dirichlet walls at those values is an exact fixed point of the
/// Euler step: every difference of a uniform field is exactly 0 (a + b - 2c with
/// a = b = c), 1 - phi^2 = 0 makes dF/dphi exactly 0 whatever u is, and so is the latent
/// heat 0.5 * (-1 - (-1)). 100 steps must leave both fields bit for bit unchanged, walls
/// included. phi alone cannot show a fault of u: dF/dphi(-1, u) = 0 for every u.
TEST_F(EnergyConservationTest, UniformFieldStaysUniform) {
    SimulationConfig cfg;
    cfg.grid.Nx = 16;
    cfg.grid.Ny = 16;
    cfg.grid.Nz = 16;
    cfg.grid.dx = 0.4;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.4;
    cfg.time.dt = 0.01;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.physics.delta = 0.8;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -0.8, 0.0, 0.0, 0.0, 0.0};
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-cfg.physics.delta);

    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    for (int step = 0; step < 100; ++step)
        solver.step(cfg.time.dt);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    int phi_changed = 0, u_changed = 0;
    for (std::size_t i = 0; i < phi.size(); ++i) {
        if (phi.data()[i] != -1.0)
            ++phi_changed;
        if (u.data()[i] != -cfg.physics.delta)
            ++u_changed;
    }
    EXPECT_EQ(phi_changed, 0);
    EXPECT_EQ(u_changed, 0);
}

/// 20 Euler steps from the seed, u's walls zero-flux and then Dirichlet at -delta: every
/// step changes the enthalpy by the wall flux within euler_bound (about 1e-11 here).
/// The latent heat of a step is up to about 0.6; a u update that dropped it would miss
/// the budget by |latent| (a sign flip by 2 |latent|), more than a bound of this size
/// whenever |latent| > 2 bound. With Dirichlet walls the flux reaches the same scale.
TEST_F(EnergyConservationTest, EulerStepChangesEnthalpyByTheWallFluxOnly) {
    for (const BoundaryConfig& u_bc : {zero_flux(), dirichlet(-0.8)}) {
        const char* what =
            u_bc.type == BCType::Dirichlet ? "Euler, u Dirichlet -0.8" : "Euler, u zero flux";
        SCOPED_TRACE(what);
        const SimulationConfig cfg = seed_config(TimeScheme::Euler, u_bc);
        BudgetRun run;
        ASSERT_NO_FATAL_FAILURE(check_enthalpy_budget(what, cfg, 20, euler_bound, run));
        EXPECT_GT(run.latent_over_bound, 2);
        if (u_bc.type == BCType::Dirichlet)
            EXPECT_GT(run.flux_over_bound, 2);
    }
}

/// 20 IMEX steps from the seed with zero-flux u walls conserve the enthalpy within
/// imex_bound (N kJacobiTol = 2.7e-7 for the 14^3 interior cells, plus rounding), while
/// the latent heat of a step is up to about 0.6.
TEST_F(EnergyConservationTest, ImexStepConservesEnthalpyWithinTheJacobiTolerance) {
    const SimulationConfig cfg = seed_config(TimeScheme::IMEX, zero_flux());
    BudgetRun run;
    ASSERT_NO_FATAL_FAILURE(check_enthalpy_budget("IMEX, u zero flux", cfg, 20, imex_bound, run));
    EXPECT_GT(run.latent_over_bound, 2);
}
