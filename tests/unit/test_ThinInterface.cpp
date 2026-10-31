// The thin-interface constants a1 and a2 (docs/THEORY.md 5.2), checked against the
// model's own equations rather than the asymptotic formulas: a sphere of the phase-field
// model must be in equilibrium at the Gibbs-Thomson undercooling of the configured d0, and
// the kinetic coefficient of a planar front must be the configured beta0.
#include "core/SimulationConfig.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace ac;

namespace {

/// x of sub[i] x[i-1] + diag[i] x[i] + sup[i] x[i+1] = rhs[i] (Thomas algorithm, no
/// pivoting; sub[0] and sup[n-1] unused). The systems below are the halves of a
/// problem split where the interface is pinned, and every pivot is checked to be
/// negative: no half has an eigenvalue near 0, unlike the whole, which has the
/// interface's displacement as an almost exact null vector.
std::vector<double> solve_tridiagonal(const std::vector<double>& sub,
                                      const std::vector<double>& diag,
                                      const std::vector<double>& sup,
                                      const std::vector<double>& rhs) {
    const std::size_t n = diag.size();
    std::vector<double> c(n), d(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double pivot = diag[i] - (i > 0 ? sub[i] * c[i - 1] : 0.0);
        EXPECT_LT(pivot, 0.0) << "row " << i;
        c[i] = sup[i] / pivot;
        d[i] = (rhs[i] - (i > 0 ? sub[i] * d[i - 1] : 0.0)) / pivot;
    }
    for (std::size_t i = n; i-- > 0;)
        x[i] = d[i] - (i + 1 < n ? c[i] * x[i + 1] : 0.0);
    return x;
}

/// The kinetic coefficient of a planar front of the model the kernels discretise,
/// on a 1D grid of spacing h = W0 / k. In the frame of a front moving at V into
/// liquid at u = -Delta,
///     W0^2 phi'' + tau0 V phi' + phi - phi^3 - lambda u (1 - phi^2)^2 = 0,
///     D u' = -V (u - phi/2 + Delta - 1/2)
/// (the thermal equation integrated once; u stays bounded in the solid only if it
/// is 1 - Delta there), with phi = 1 (solid) at x = -L, -1 (liquid) at x = L and
/// the front pinned at phi(0) = 0. A planar sharp interface has Delta - 1 = beta V
/// exactly, so the kinetic coefficient is dDelta/dV at V = 0: the derivative of the
/// discrete travelling wave at rest. phi'' and phi' are central differences and u'
/// the trapezoid rule, so it tends to the continuum's value as h^2.
double planar_kinetic_coefficient(const PhysicsParams& p, std::size_t k) {
    const double w = p.W0, h = w / static_cast<double>(k), s = w * w / (h * h);
    const double lambda = p.lambda(), tau0 = p.tau0();
    // L = 25 W0: there 1 - |phi0| = 2 exp(-25 sqrt 2) < 9e-16, a few ulps of 1.
    const std::size_t half = 25 * k, n = 2 * half; // nodes 0..n at x = (j - half) h

    // The front at rest: phi(-x) = -phi(x), so phi(0) = 0, and Newton's method on
    // the left half, whose Jacobian (the linearisation with phi = 0 imposed at the
    // centre) is negative definite, unlike the whole domain's, which has the
    // front's translation as an almost exact null vector.
    std::vector<double> phi(n + 1);
    for (std::size_t j = 0; j <= n; ++j)
        phi[j] = -std::tanh((static_cast<double>(j) - static_cast<double>(half)) * h /
                            (std::sqrt(2.0) * w));
    phi[0] = 1.0;
    phi[half] = 0.0;
    double last = std::numeric_limits<double>::infinity();
    for (int it = 0;; ++it) {
        if (it == 50) {
            ADD_FAILURE() << "Newton's method did not converge, k = " << k;
            return std::numeric_limits<double>::quiet_NaN();
        }
        std::vector<double> diag(half - 1), res(half - 1);
        for (std::size_t j = 1; j < half; ++j) {
            diag[j - 1] = -2.0 * s + 1.0 - 3.0 * phi[j] * phi[j];
            res[j - 1] =
                s * (phi[j + 1] - 2.0 * phi[j] + phi[j - 1]) + phi[j] - phi[j] * phi[j] * phi[j];
        }
        const std::vector<double> off(half - 1, s);
        const std::vector<double> step = solve_tridiagonal(off, diag, off, res);
        double largest = 0.0;
        for (std::size_t j = 1; j < half; ++j) {
            phi[j] -= step[j - 1];
            largest = std::max(largest, std::abs(step[j - 1]));
        }
        // Converged once a step no longer shrinks: the rounding floor.
        if (largest == 0.0 || largest >= last)
            break;
        last = largest;
    }
    for (std::size_t j = half + 1; j <= n; ++j)
        phi[j] = -phi[n - j];

    // d/dV at V = 0 (u = 0, Delta = 1 there): u_V(x_j) = -Delta_V + U_j from the
    // trapezoid rule, and phi_V solves
    //   W0^2 D2 phi_V + (1 - 3 phi^2) phi_V = -tau0 D1 phi + lambda (1 - phi^2)^2 u_V
    // with phi_V = 0 at -L, L and at the pinned centre. On each half it is a
    // negative definite tridiagonal system: phi_V = a - Delta_V b there, and the
    // centre's equation gives Delta_V.
    std::vector<double> g(n + 1), u(n + 1, 0.0), rhs(n + 1, 0.0);
    for (std::size_t j = 0; j <= n; ++j)
        g[j] = (1.0 - phi[j] * phi[j]) * (1.0 - phi[j] * phi[j]);
    for (std::size_t j = 1; j <= n; ++j)
        u[j] = u[j - 1] - (h / p.D) * (0.5 - (phi[j] + phi[j - 1]) / 4.0);
    for (std::size_t j = 1; j < n; ++j)
        rhs[j] = -tau0 * (phi[j + 1] - phi[j - 1]) / (2.0 * h) + lambda * g[j] * u[j];
    auto solve_half = [&](std::size_t first) {
        std::vector<double> diag(half - 1), r(half - 1), coupling(half - 1);
        for (std::size_t i = 0; i + 1 < half; ++i) {
            const std::size_t j = first + i;
            diag[i] = -2.0 * s + 1.0 - 3.0 * phi[j] * phi[j];
            r[i] = rhs[j];
            coupling[i] = lambda * g[j];
        }
        const std::vector<double> off(half - 1, s);
        return std::array<std::vector<double>, 2>{solve_tridiagonal(off, diag, off, r),
                                                  solve_tridiagonal(off, diag, off, coupling)};
    };
    const auto [a_left, b_left] = solve_half(1);
    const auto [a_right, b_right] = solve_half(half + 1);
    return (rhs[half] - s * (a_left.back() + a_right.front())) /
           (lambda * g[half] - s * (b_left.back() + b_right.front()));
}

/// The uniform u at which a sphere of radius R = m W0 is in equilibrium in the model:
/// the critical nucleus, the radial solution of
///     W0^2 (phi'' + 2 phi'/r) + phi - phi^3 - lambda u (1 - phi^2)^2 = 0
/// with phi'(0) = 0, phi(R) = 0 and phi = -1 at r = R + 30 W0 (where 1 + phi0 =
/// 2 exp(-30 sqrt 2) < 1e-18), on a grid of spacing h = W0 / k: central differences, and
/// 6 (phi_1 - phi_0) / h^2 for the Laplacian at r = 0, where phi is even. Newton's method
/// on (phi, u), each step split at the pinned node r = R into the two radial halves.
double critical_undercooling(const PhysicsParams& p, std::size_t m, std::size_t k) {
    const double w = p.W0, h = w / static_cast<double>(k), s = w * w / (h * h);
    const double lambda = p.lambda(), radius = static_cast<double>(m) * w;
    const std::size_t pin = m * k, n = pin + 30 * k; // nodes 0..n at r = j h
    std::vector<double> phi(n + 1);
    for (std::size_t j = 0; j <= n; ++j)
        phi[j] = -std::tanh((static_cast<double>(j) * h - radius) / (std::sqrt(2.0) * w));
    phi[pin] = 0.0;
    phi[n] = -1.0;
    double u = -2.0 * p.d0 / radius; // the sharp-interface value as the first guess
    double last = std::numeric_limits<double>::infinity();
    for (int it = 0;; ++it) {
        if (it == 50) {
            ADD_FAILURE() << "Newton's method did not converge, R = " << m << " W0, k = " << k;
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Rows 0..n-1: the residual, its Jacobian in phi (tridiagonal) and in u.
        std::vector<double> sub(n, 0.0), diag(n), sup(n, 0.0), res(n), coupling(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double r = static_cast<double>(j) * h, q = phi[j];
            const double g = (1.0 - q * q) * (1.0 - q * q);
            double lap;
            if (j == 0) {
                lap = 6.0 * (phi[1] - phi[0]) / (h * h);
                diag[0] = -6.0 * s;
                sup[0] = 6.0 * s;
            } else {
                lap = (phi[j + 1] - 2.0 * q + phi[j - 1]) / (h * h) +
                      (phi[j + 1] - phi[j - 1]) / (r * h);
                sub[j] = s * (1.0 - h / r);
                diag[j] = -2.0 * s;
                sup[j] = s * (1.0 + h / r);
            }
            res[j] = w * w * lap + q - q * q * q - lambda * u * g;
            diag[j] += 1.0 - 3.0 * q * q + lambda * u * 4.0 * q * (1.0 - q * q);
            coupling[j] = -lambda * g;
        }
        // J dphi + coupling du = res with dphi = 0 at the pin and at r_n: on each half
        // dphi = a - du b, and the pinned row gives du.
        auto solve_half = [&](std::size_t first, std::size_t count) {
            auto part = [&](const std::vector<double>& v) {
                return std::vector<double>(v.begin() + static_cast<std::ptrdiff_t>(first),
                                           v.begin() + static_cast<std::ptrdiff_t>(first + count));
            };
            return std::array<std::vector<double>, 2>{
                solve_tridiagonal(part(sub), part(diag), part(sup), part(res)),
                solve_tridiagonal(part(sub), part(diag), part(sup), part(coupling))};
        };
        const auto [a_in, b_in] = solve_half(0, pin);
        const auto [a_out, b_out] = solve_half(pin + 1, n - pin - 1);
        const double du = (res[pin] - sub[pin] * a_in.back() - sup[pin] * a_out.front()) /
                          (coupling[pin] - sub[pin] * b_in.back() - sup[pin] * b_out.front());
        double largest = std::abs(du);
        for (std::size_t i = 0; i < pin; ++i) {
            const double d = a_in[i] - du * b_in[i];
            phi[i] -= d;
            largest = std::max(largest, std::abs(d));
        }
        for (std::size_t i = 0; i + pin + 1 < n; ++i) {
            const double d = a_out[i] - du * b_out[i];
            phi[pin + 1 + i] -= d;
            largest = std::max(largest, std::abs(d));
        }
        u -= du;
        // Converged once a step no longer shrinks: the rounding floor.
        if (largest == 0.0 || largest >= last)
            return u;
        last = largest;
    }
}

} // namespace

// A sphere of radius R is in equilibrium at u = -2 d0 / R in the sharp-interface limit
// (Gibbs-Thomson, curvature 2/R), so with lambda() = a1 W0 / d0 and a1 = 5 sqrt(2)/8 the
// critical nucleus of the model must give e = -u R / (2 d0) - 1 -> 0 as W0 / R -> 0. It
// does at second order, e = c (W0/R)^2 + ... with c from -0.30457 to -0.30471 for R = 10
// to 80 W0, so doubling R divides e by 4: the observed order must lie within 1/2 of 2, as
// for the kinetic coefficient below.
// Each e is extrapolated in h from W0/10 and W0/20 (second-order differences; what remains,
// 2.6e-8 at R = 80 W0 where e = -4.8e-5, measured against W0/40). Measured: orders 1.9996,
// 2.0001 and 2.0006 for R = 10, 20, 40 and 80 W0; a1 = 0.88 gives -1.43, -0.21 and -0.05.
// u enters only as lambda u, so e depends on R / W0 alone; the second case (W0 = 1.5,
// d0 = 0.2) checks how lambda() scales with them.
TEST(ThinInterface, CriticalNucleusIsInGibbsThomsonEquilibrium) {
    struct Case {
        double W0, d0;
    };
    for (const Case c : {Case{1.0, 0.5}, Case{1.5, 0.2}}) {
        PhysicsParams p; // a1 as shipped
        p.W0 = c.W0;
        p.d0 = c.d0;
        std::array<double, 4> err{};
        for (std::size_t i = 0; i < err.size(); ++i) {
            const std::size_t m = std::size_t{10} << i; // R = m W0
            const double radius = static_cast<double>(m) * c.W0;
            auto e = [&](std::size_t k) {
                return -critical_undercooling(p, m, k) * radius / (2.0 * c.d0) - 1.0;
            };
            err[i] = (4.0 * e(20) - e(10)) / 3.0;
        }
        for (std::size_t i = 0; i + 1 < err.size(); ++i) {
            const double ratio = err[i] / err[i + 1];
            EXPECT_TRUE(ratio > std::exp2(1.5) && ratio < std::exp2(2.5))
                << "W0 " << c.W0 << ", d0 " << c.d0 << ": -u R/(2 d0) - 1 = " << err[i]
                << " at R = " << (std::size_t{10} << i) << " W0 and " << err[i + 1]
                << " at R = " << (std::size_t{10} << (i + 1)) << " W0, order " << std::log2(ratio);
        }
    }
}

// With the shipped a1 and a2, lambda() and tau0() make the planar kinetic
// coefficient of the model the configured beta0 in the thin-interface limit. On
// the grid, beta(h) - beta0 = c h^2 + O(h^4), so halving h divides it by 4: the
// observed order must lie within 1/2 of 2, the bound the scheme-order tests use
// (test_SchemeComparison.cu), which tells order 2 from orders 1 and 3. Measured:
// orders 2.00001 to 2.0004 (h = W0/10 to W0/80) in every case. With any other a2
// the difference tends to a1 (a2 - 47/75) W0 / D instead and the ratio to 1:
// a2 = 0.64 gave orders -0.022 to -0.0008, a limit of 0.0059 at the shipped W0 = 1
// and D = 2. a2 off by 2e-6 either way fails too (+1e-6 fails, -1e-6 passes), and
// so does the literature's four-digit 0.6267.
TEST(ThinInterface, PlanarFrontKineticCoefficientIsBeta0) {
    struct Case {
        double W0, D, d0, beta0;
    };
    // The shipped physics (W0 = 1, D = 2, d0 = 0.5, beta0 = 0), the same with a
    // kinetic coefficient, and another scale.
    for (const Case c :
         {Case{1.0, 2.0, 0.5, 0.0}, Case{1.0, 2.0, 0.5, 0.3}, Case{1.5, 0.7, 0.2, 0.05}}) {
        PhysicsParams p; // a1 and a2 as shipped
        p.W0 = c.W0;
        p.D = c.D;
        p.d0 = c.d0;
        p.beta0 = c.beta0;
        std::array<double, 4> err{};
        for (std::size_t i = 0; i < err.size(); ++i)
            err[i] = planar_kinetic_coefficient(p, std::size_t{10} << i) - c.beta0;
        for (std::size_t i = 0; i + 1 < err.size(); ++i) {
            const double ratio = err[i] / err[i + 1];
            EXPECT_TRUE(ratio > std::exp2(1.5) && ratio < std::exp2(2.5))
                << "W0 " << c.W0 << ", D " << c.D << ", d0 " << c.d0 << ", beta0 " << c.beta0
                << ": beta - beta0 = " << err[i] << " at h = W0/" << (std::size_t{10} << i)
                << " and " << err[i + 1] << " at h = W0/" << (std::size_t{10} << (i + 1))
                << ", order " << std::log2(ratio) << "; the limit is near beta0 + "
                << (4.0 * err[i + 1] - err[i]) / 3.0;
        }
    }
}
