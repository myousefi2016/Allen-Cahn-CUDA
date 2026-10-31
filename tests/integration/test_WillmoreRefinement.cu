#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <vector>

using namespace ac;
using namespace ac::cuda;

namespace {

/// One level of the refinement ladder: N^3 points at spacing h over the same
/// box (N - 1) h = 18.4, with dt = 0.03 (h / 0.8)^2.
struct Level {
    int N;
    double h;
    double dt;
};
constexpr Level kLadder[] = {{24, 0.8, 0.03}, {47, 0.4, 0.0075}, {93, 0.2, 0.001875}};
constexpr double kEndTime = 7.5;

/// phi at kEndTime from a tanh seed of radius 4 in the centre of the box,
/// undercooling 0.55 (u = -0.55 initially and on the walls), zero-flux phi,
/// eps = 0.12. beta = 0 is the unregularised equation, which validate()
/// rejects at this eps: the solver is built from the config directly.
std::vector<double> final_phi(double beta, const Level& level) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = level.N;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = level.h;
    cfg.physics.delta = 0.55;
    cfg.physics.epsilon = 0.12;
    cfg.physics.W0 = 1.0;
    cfg.physics.D = 2.0;
    cfg.physics.d0 = 0.5;
    cfg.physics.willmore_beta = beta;
    cfg.time.dt = level.dt;
    cfg.time.scheme = TimeScheme::Euler;
    cfg.stencil = StencilType::Isotropic27Point;
    cfg.boundary.phi_bc = {BCType::Neumann, 0.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, -0.55, 0.0, 0.0, 0.0, 0.0};
    if (beta > 0.0)
        cfg.validate();

    const Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    const double c = 0.5 * (level.N - 1);
    for (int x = 0; x < level.N; ++x)
        for (int y = 0; y < level.N; ++y)
            for (int z = 0; z < level.N; ++z) {
                const double r =
                    level.h * std::sqrt((x - c) * (x - c) + (y - c) * (y - c) + (z - c) * (z - c));
                phi(x, y, z) = -std::tanh((r - 4.0) / std::sqrt(2.0));
                u(x, y, z) = -0.55;
            }
    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    const int steps = static_cast<int>(std::lround(kEndTime / level.dt));
    for (int s = 0; s < steps; ++s)
        solver.step(level.dt);
    solver.copy_phi_to_host(phi);
    return std::vector<double>(phi.data(), phi.data() + phi.size());
}

struct Difference {
    double max_norm = 0.0;
    double mean_abs = 0.0;
};

/// |phi_a - phi_b| on the interior nodes of the coarsest grid (Nc points per
/// axis), which are nodes of every level: node i of that grid is node
/// stride_a * i of grid a (Na points) and stride_b * i of grid b. Both
/// differences of a ladder are taken on these same nodes, as a three-grid
/// convergence study requires.
Difference difference(const std::vector<double>& a, int Na, int stride_a,
                      const std::vector<double>& b, int Nb, int stride_b, int Nc) {
    auto at = [](const std::vector<double>& v, int N, int s, int i, int j, int k) {
        return v[static_cast<std::size_t>(((s * i) * N + s * j) * N + s * k)];
    };
    Difference d;
    long count = 0;
    for (int i = 1; i < Nc - 1; ++i)
        for (int j = 1; j < Nc - 1; ++j)
            for (int k = 1; k < Nc - 1; ++k) {
                const double diff =
                    std::fabs(at(a, Na, stride_a, i, j, k) - at(b, Nb, stride_b, i, j, k));
                d.max_norm = std::max(d.max_norm, diff);
                d.mean_abs += diff;
                ++count;
            }
    d.mean_abs /= static_cast<double>(count);
    return d;
}

class WillmoreRefinementTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

} // namespace

// At eps = 0.12 > 1/15 the gradient energy is not convex and the Allen-Cahn
// equation is ill-posed (docs/THEORY.md section 4.3): refinement does not
// converge, and the mean difference between successive levels grows
// (ratio d(0.8, 0.4) / d(0.4, 0.2) of the mean |.| below 1). With the
// Willmore term at the shipped strength for this eps (beta = 0.25, corner
// length 0.528 W0) the ladder converges by the criterion of section 4.3: a
// ratio above 2 in both norms, i.e. each halving of h removes more than half
// of the difference. The unregularised max-norm ratio is not asserted: on the
// common nodes it depends on where the grid-scale structure falls (0.783
// here, 2.25 in the larger box of section 4.3).
// Both differences are taken on the interior nodes of the h = 0.8 grid.
// Measured on an RTX 4090: unregularised 0.783 (max norm) and 0.633 (mean),
// regularised 3.49 and 3.87.
TEST_F(WillmoreRefinementTest, RegularisedConvergesUnregularisedDoesNot) {
    for (double beta : {0.0, 0.25}) {
        std::vector<std::vector<double>> phi;
        for (const Level& level : kLadder)
            phi.push_back(final_phi(beta, level));
        const int Nc = kLadder[0].N;
        const Difference d1 = difference(phi[0], Nc, 1, phi[1], kLadder[1].N, 2, Nc);
        const Difference d2 = difference(phi[1], kLadder[1].N, 2, phi[2], kLadder[2].N, 4, Nc);
        const double r_max = d1.max_norm / d2.max_norm, r_mean = d1.mean_abs / d2.mean_abs;
        std::printf("beta %.2f: max |dphi| %.4f -> %.4f (ratio %.3f), mean %.3e -> %.3e "
                    "(ratio %.3f)\n",
                    beta, d1.max_norm, d2.max_norm, r_max, d1.mean_abs, d2.mean_abs, r_mean);
        if (beta > 0.0) {
            EXPECT_GT(r_max, 2.0);
            EXPECT_GT(r_mean, 2.0);
        } else {
            EXPECT_LT(r_mean, 1.0);
        }
    }
}
