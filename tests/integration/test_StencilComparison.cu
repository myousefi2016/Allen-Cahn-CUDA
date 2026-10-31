// The stencil option selects the Laplacian of the thermal update and nothing
// else: phi's operator is the face-flux divergence (anisotropic_divergence in
// Kernels.cuh) for both settings. This file checks, end to end through
// SimulationConfig, KernelParams and every time scheme, that u is advanced by
// the discrete heat equation of the selected stencil.
#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>

using namespace ac;
using namespace ac::cuda;

namespace {

using LD = long double;

constexpr int kN = 17;     // interior cells 1..15, Dirichlet cells 0 and 16
constexpr double kH = 0.4; // cubic, as the 27-point stencil requires
constexpr double kD = 2.0;
constexpr double kDt = 0.005; // dt D / h^2 = 1/16
constexpr int kSteps = 20;
constexpr double kAmplitude = 0.5;

/// gamma_k = k u / (1 - k u) (Higham, Lemma 3.1) with u the unit roundoff of
/// double, or of long double.
LD gamma_double(LD k) {
    const LD u = static_cast<LD>(std::numeric_limits<double>::epsilon()) / 2;
    return k * u / (1 - k * u);
}
LD gamma_long(LD k) {
    const LD u = LDBL_EPSILON / 2;
    return k * u / (1 - k * u);
}

/// The liquid (phi = -1, Dirichlet -1) with u = 0 on the faces, at the physics
/// of the other integration tests. validate() accepts dt = 0.005 for every
/// scheme and stencil: the thermal limits are 0.0133 (Euler and Heun, 7-point)
/// and above; the phase-field limit is 0.0104.
SimulationConfig mode_config(TimeScheme scheme, StencilType stencil) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = kN;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = kH;
    cfg.physics.delta = 0.8;
    cfg.physics.epsilon = 0.05;
    cfg.physics.W0 = 1.0;
    cfg.physics.D = kD;
    cfg.physics.d0 = 0.5;
    cfg.time.dt = kDt;
    cfg.time.scheme = scheme;
    cfg.stencil = stencil;
    cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
    cfg.boundary.u_bc = {BCType::Dirichlet, 0.0, 0.0, 0.0, 0.0, 0.0};
    cfg.output.frequency = 10000;
    cfg.checkpoint.frequency = 0;
    cfg.validate();
    return cfg;
}

/// lambda h^2 of the stencil on prod_a sin(theta_a x_a), from c_a = cos theta_a.
/// A +-1 pair along axis a gives 2 c_a, so with the weights of docs/DESIGN.md:
///   7-point:  sum_a 2 (c_a - 1)
///   27-point: (28 sum c_a + 12 sum_{a<b} c_a c_b + 8 c_x c_y c_z - 128) / 30
/// (faces 14 * 2 c_a, edges 3 * 4 c_a c_b, corners 1 * 8 c_x c_y c_z).
LD symbol_h2(StencilType stencil, const LD c[3]) {
    if (stencil == StencilType::Standard7Point)
        return 2 * (c[0] - 1) + 2 * (c[1] - 1) + 2 * (c[2] - 1);
    return (28 * (c[0] + c[1] + c[2]) + 12 * (c[0] * c[1] + c[1] * c[2] + c[0] * c[2]) +
            8 * c[0] * c[1] * c[2] - 128) /
           30;
}

/// h^2 sum_e |w_e|: 7-point 6 + 6 = 12; 27-point (6*14 + 12*3 + 8 + 128)/30 = 256/30.
LD sum_abs_weights_h2(StencilType stencil) {
    return stencil == StencilType::Standard7Point ? LD(12) : LD(256) / 30;
}

/// R(z): the factor by which one step multiplies an eigenvector of dt D Lap_h,
/// with eigenvalue z, while phi is the liquid fixed point.
/// - Euler: 1 + z.
/// - Heun: 0.5 (1 + (1 + z)^2).
/// - RK4: the degree-4 Taylor polynomial of e^z.
/// - IMEX: 1 / (1 - z).
/// Written as monomials: at most 8 roundings per evaluation.
LD step_factor(TimeScheme scheme, LD z) {
    switch (scheme) {
    case TimeScheme::Euler:
        return 1 + z;
    case TimeScheme::Heun:
        return 1 + z + z * z / 2;
    case TimeScheme::RK4:
        return 1 + z + z * z / 2 + z * z * z / 6 + z * z * z * z / 24;
    case TimeScheme::IMEX:
        return 1 / (1 - z);
    }
    return std::numeric_limits<LD>::quiet_NaN();
}

/// The most roundings any term of one GPU step passes through, counted on the
/// 27-point path; the 7-point path is shorter (6 per Laplacian).
/// - laplacian_27pt: 11 (edge sum) + 1 (x3) + 3 (combination) + 2 (30 h h)
///   + 1 (/) = 18.
/// - Euler (thermal_equation_kernel): + dt*D, *lap, + u gives 21.
/// - Heun: two Euler stages (42), then the average's + and *0.5 gives 44.
/// - RK4: a stage's rhs is D*lap and + 0.5 k_phi (20), its axpy c, c*k and +
///   (3). k4 comes after three stages and its own rhs: 3 * 23 + 20 = 89.
///   The combination's + k4, dt/6, * and + a give 93.
int roundings_per_step(TimeScheme scheme) {
    switch (scheme) {
    case TimeScheme::Euler:
        return 21;
    case TimeScheme::Heun:
        return 44;
    case TimeScheme::RK4:
        return 93;
    case TimeScheme::IMEX:
        return 0; // the Jacobi stopping rule certifies the step instead
    }
    return 0;
}

const char* scheme_name(TimeScheme s) {
    switch (s) {
    case TimeScheme::Euler:
        return "Euler";
    case TimeScheme::Heun:
        return "Heun";
    case TimeScheme::RK4:
        return "RK4";
    case TimeScheme::IMEX:
        return "IMEX";
    }
    return "?";
}

} // namespace

class StencilComparisonTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

// The initial u is u0 = A prod_a sin(theta_a x_a), theta_a = pi m_a / (N - 1).
// It vanishes on the Dirichlet faces (x_a = 0 and N - 1) and is an exact
// eigenvector of either stencil: sum_e w_e prod_a sin(theta_a (x_a + e_a)) =
// lambda prod_a sin(theta_a x_a). phi = -1 is a fixed point of every
// phase-field update: there is no gradient and dF/dphi(-1, u) = 0, so the rate
// is 0 and phi stays -1 bitwise. The latent heat is then 0, and each scheme
// maps u to G u with G = R(dt D Lap_h). After n steps, u = R(z)^n u0 with
// z = dt D lambda.
//
// The four modes all have |m|^2 = 27: three close to the axes and one along
// <111>. Their 27-point lambda h^2 are -0.9541 (three times) and -0.9557,
// agreeing to O(theta^6). Their 7-point values are -0.9657 and -1.0112.
//
// The bound on max |u - R^n u0| over every cell is the sum of three terms.
// (1) 2 E0, where E0 = A gamma(4 pi sum m + 6) bounds how far the computed u0
//     is from the exact mode. theta takes 3 roundings and theta x one more, so
//     the argument is off by at most gamma_4 pi m_a; sin adds 1 ulp; there are
//     three products. G and R(z)^n both have max norm <= 1 (next paragraph),
//     so this part of u0 contributes at most 2 E0.
// (2) The GPU's rounding. One step evaluates a polynomial in the cell values
//     whose absolute-value form is R(zeta) max|u|, with zeta = dt D sum|w_e|.
//     Each term passes through at most K roundings (roundings_per_step), so a
//     step adds at most gamma_K R(zeta) max|u| (Higham, sec. 3.3). G
//     propagates these errors without growth, so n steps add at most n times
//     that.
//     Why ||G||_inf <= 1: with a = zeta/2 <= 1 (a is dt D times |w_centre|),
//     write Z = dt D Lap_h = -a I + P with P >= 0 and P 1 <= a 1. Every
//     derivative of the Taylor polynomials R is >= 0 on [-1, 0], so
//     G = sum_j R^(j)(-a) P^j / j! >= 0 and G 1 <= R(0) 1 = 1. The same
//     condition gives |R| <= 1 and |R'| <= 1 on [-zeta, 0].
//     IMEX: (I - Z)^-1 is the inverse of an M-matrix with row sums >= 1, and
//     each step's Jacobi solve is certified to kJacobiTol in the max norm
//     (docs/THEORY.md 10.2), which gives n kJacobiTol.
// (3) The long-double reference's own rounding.
//     - c_a is within gamma_L(12) of cos theta_a (theta takes 3 roundings and
//       theta <= pi; cosl is allowed 2 ulp).
//     - The symbol's slopes sum to at most 6. Its evaluation takes at most 9
//       roundings with absolute form <= 12. So lambda h^2 is within
//       gamma_L(180).
//     - z takes 4 more roundings, which gives
//       dz = (dt D/h^2) gamma_L(184) + zeta gamma_L(4).
//     - Then n dz, plus R's own evaluation (n R(zeta) gamma_L(8)), the n
//       products and the comparison (gamma_L(n + 2)).
TEST_F(StencilComparisonTest, ThermalModeDecaysByTheStencilEigenvalue) {
    const int modes[][3] = {{5, 1, 1}, {1, 5, 1}, {1, 1, 5}, {3, 3, 3}};
    const double pi = 3.14159265358979323846;
    const LD pi_l = 3.141592653589793238462643383279502884L;
    const StencilType stencils[2] = {StencilType::Standard7Point, StencilType::Isotropic27Point};
    for (TimeScheme scheme :
         {TimeScheme::Euler, TimeScheme::Heun, TimeScheme::RK4, TimeScheme::IMEX}) {
        for (const auto& m : modes) {
            const Grid grid = mode_config(scheme, stencils[0]).make_grid();
            FieldData phi0(grid, "phi0"), u0(grid, "u0");
            const double theta[3] = {pi * m[0] / (kN - 1), pi * m[1] / (kN - 1),
                                     pi * m[2] / (kN - 1)};
            double s_max = 0.0;
            for (int x = 0; x < kN; ++x)
                for (int y = 0; y < kN; ++y)
                    for (int z = 0; z < kN; ++z) {
                        const bool wall =
                            x == 0 || y == 0 || z == 0 || x == kN - 1 || y == kN - 1 || z == kN - 1;
                        phi0(x, y, z) = -1.0;
                        u0(x, y, z) = wall ? 0.0
                                           : kAmplitude * std::sin(theta[0] * x) *
                                                 std::sin(theta[1] * y) * std::sin(theta[2] * z);
                        s_max = std::max(s_max, std::fabs(u0(x, y, z)));
                    }
            LD factor[2] = {0, 0}; // R(z)^n, 7- and 27-point
            LD bound[2] = {0, 0};
            for (int k = 0; k < 2; ++k) {
                const SimulationConfig cfg = mode_config(scheme, stencils[k]);
                CudaSolver solver(cfg);
                solver.initialize(phi0, u0);
                for (int s = 0; s < kSteps; ++s)
                    solver.step(kDt);
                FieldData phi(grid, "phi"), u(grid, "u");
                solver.copy_phi_to_host(phi);
                solver.copy_u_to_host(u);

                LD c[3];
                for (int a = 0; a < 3; ++a)
                    c[a] = std::cos(pi_l * m[a] / (kN - 1));
                const LD dtD_h2 = LD(kDt) * LD(kD) / (LD(kH) * LD(kH));
                const LD lambda_h2 = symbol_h2(stencils[k], c);
                const LD r = step_factor(scheme, dtD_h2 * lambda_h2);
                LD rn = 1;
                for (int s = 0; s < kSteps; ++s)
                    rn *= r;
                factor[k] = rn;

                const bool imex = scheme == TimeScheme::IMEX;
                const LD zeta = dtD_h2 * sum_abs_weights_h2(stencils[k]);
                if (!imex)
                    ASSERT_LE(zeta / 2, 1) << "dt D |w_centre| > 1: ||G||_inf <= 1 not shown";
                const LD abs_factor = imex ? LD(1) : step_factor(scheme, zeta);
                const LD e0 = LD(kAmplitude) * gamma_double(4 * pi_l * (m[0] + m[1] + m[2]) + 6);
                LD rounding = kSteps * LD(CudaSolver::kJacobiTol);
                if (!imex) {
                    const LD per_step = gamma_double(roundings_per_step(scheme)) * abs_factor;
                    rounding = kSteps * per_step * s_max / (1 - kSteps * per_step);
                }
                const LD dz = dtD_h2 * gamma_long(184) + zeta * gamma_long(4);
                const LD reference = s_max * (kSteps * dz + kSteps * abs_factor * gamma_long(8) +
                                              gamma_long(kSteps + 2));
                bound[k] = 2 * e0 + rounding + reference;

                LD err = 0;
                for (int x = 0; x < kN; ++x)
                    for (int y = 0; y < kN; ++y)
                        for (int z = 0; z < kN; ++z) {
                            ASSERT_EQ(phi(x, y, z), -1.0)
                                << "phi must stay the liquid fixed point at (" << x << "," << y
                                << "," << z << ")";
                            const LD diff = std::fabs(LD(u(x, y, z)) - rn * LD(u0(x, y, z)));
                            if (!(diff <= err)) // keeps a NaN
                                err = diff;
                        }
                std::printf("%-5s %s mode (%d,%d,%d): lambda h^2 %.6Lf, R^n %.9Lf, "
                            "max |u - R^n u0| %.3Le (bound %.3Le)\n",
                            scheme_name(scheme), k == 0 ? "7-point " : "27-point", m[0], m[1], m[2],
                            lambda_h2, rn, err, bound[k]);
                EXPECT_LE(err, bound[k])
                    << scheme_name(scheme) << (k == 0 ? " 7" : " 27") << "-point, mode (" << m[0]
                    << "," << m[1] << "," << m[2] << ")";
            }
            // Suppose a run applied the other stencil. It would lie within that
            // stencil's bound of that stencil's reference, so at the cell of
            // max |u0| it would miss its own reference by more than its bound.
            EXPECT_GT(std::fabs(factor[1] - factor[0]) * s_max, bound[0] + bound[1])
                << "the two stencils are not told apart at this dt and n";
        }
    }
}
