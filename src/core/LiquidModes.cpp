#include "core/LiquidModes.hpp"

#include "core/Interval.hpp"
#include "core/LiquidModesDetail.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

// Certified extrema over the liquid-gradient modes (docs/THEORY.md 10.1).
//
// The objective is bilinear in phi(n) and psi(theta):
//   F(n, theta) = sum_k phi_k(n) psi_k(theta),
//   phi = (H_xx, H_yy, H_zz, H_xy, H_xz, H_yz, 1) / A(n)^2       (growth)
//         (-H_xx, ..., -H_yz, 1)                                  (damping, F = -(Lam + beta L))
//   psi = (W c_x, W c_y, W c_z, 2W q_x q_y, 2W q_x q_z, 2W q_y q_z, kappa - beta (W mu + 2)^2),
//   c_d = 2 (1 - cos theta_d) / h_d^2, q_d = sin theta_d / h_d, mu = sum c_d, W = W0^2,
//   kappa = 2 (growth) or -2 (damping).
// Its maximum is found by a two-level branch and bound:
//  * outer boxes in the direction n = v / |v|, v on the faces (1, a, b), (a, 1, b),
//    (a, b, 1) of the unit cube (a, b in [0, 1] cover the octant; a sign change of n_d
//    together with theta_d leaves F unchanged, since it flips the sign of H_de and of
//    sin theta_d for every e != d, so the octant suffices), or, when the three spacings
//    are equal, the triangle 0 <= b <= a <= 1 of the first face (axis permutations of n
//    and theta together leave F unchanged);
//  * inner boxes in theta over kThetaLo..kThetaHi (LiquidModesDetail.hpp): theta_x in
//    [0, 3.2], theta_y, theta_z in [-3.2, 3.2] (F is even in theta and 2 pi periodic;
//    3.2 > pi covers the period).
// The upper bound of F over (n-box x theta-box) is the smallest of three bounds, each an
// upper bound by itself:
//  * joint: phi over the whole n-box, with each axis's terms of F maximised exactly over
//    the theta box (a concave parabola in c_d) and the pair terms by interval extension;
//  * second order in n: F(n_c, theta) over the theta box (interval extension, and the same
//    per-axis decomposition) plus a second-order Taylor remainder in n around the box
//    centre n_c, with the Hessian of phi enclosed over the box;
//  * second order in theta as well: a second-order Taylor bound of F(n_c, theta) around
//    the theta box's centre, with the theta Hessian enclosed over the box, plus the same
//    remainder in n.
// Every bound is a composition of the outward-rounded operations of core/Interval.hpp
// (the coefficients of the formula too: 1 - 3 eps and 256 eps^2 are not doubles), so it
// encloses the exact real value of the formula for the given double parameters. A theta
// box whose interval theta gradient excludes zero in some component holds no stationary
// point and so not the maximum of the periodic F, and is dropped. Every value the search
// records is the lower end of an outward-rounded enclosure of F at a mode, so the value it
// reports is attained. For the growth objective it stops when no box can exceed the best
// value by more than kLiquidModeTolerance * max(1, |best|), and that limit, rounded up, is
// the bound. For the damping objective it decides a sign: it stops when no box can exceed
// 0 (the bound is then 0) or at the first mode whose enclosure lies above 0 (a witness).
// A search that spends its evaluation budget (it may exceed it by the inner searches of the
// direction box in hand), or meets a box it cannot split, keeps every undecided box and
// reports the range the extremum is known to lie in (LiquidModeSearchExhausted).

namespace ac {

void gradient_energy_hessian(const double n[3], double epsilon, double H[3][3]) {
    const double s =
        n[0] * n[0] * n[0] * n[0] + n[1] * n[1] * n[1] * n[1] + n[2] * n[2] * n[2] * n[2];
    const double a = 1.0 - 3.0 * epsilon + 4.0 * epsilon * s;
    double D[3];
    for (int i = 0; i < 3; ++i)
        D[i] = n[i] * (n[i] * n[i] - s);
    for (int i = 0; i < 3; ++i) {
        const double ni2 = n[i] * n[i];
        H[i][i] = a * a + 32.0 * epsilon * a * D[i] * n[i] +
                  16.0 * epsilon *
                      (16.0 * epsilon * D[i] * D[i] +
                       a * (3.0 * ni2 - 6.0 * ni2 * ni2 - s + 4.0 * s * ni2));
        for (int j = i + 1; j < 3; ++j) // H is symmetric: one evaluation per pair
            H[i][j] = H[j][i] = 256.0 * epsilon * epsilon * D[i] * D[j] +
                                32.0 * epsilon * a * n[i] * n[j] * (s - ni2 - n[j] * n[j]);
    }
}

namespace {

using interval::add_up;
using interval::cst;
using interval::D2;
using interval::iabs;
using interval::icos;
using interval::isin;
using interval::Iv;
using interval::kPi;
using interval::mag;
using interval::mul_up;
using interval::sq;
using interval::up;
using liquid_modes_detail::Objective;

// ── The coefficients phi_k(n) ───────────────────────────────────────────────

/// The epsilon-dependent coefficients of H(n), enclosed (1 - 3 eps and 256 eps^2 round).
struct EpsCoefficients {
    Iv one_minus_3, four, sixteen, thirty_two, two_fifty_six_sq;
};
EpsCoefficients eps_coefficients(double eps) {
    const Iv e(eps);
    return {Iv(1.0) - Iv(3.0) * e, Iv(4.0) * e, Iv(16.0) * e, Iv(32.0) * e, Iv(256.0) * sq(e)};
}

template <class T> T constant(double c);
template <> Iv constant<Iv>(double c) {
    return Iv(c);
}
template <> D2 constant<D2>(double c) {
    return cst(c);
}

/// phi_k(n), k = 0..6, n = v / |v|; T = Iv encloses the values, T = D2 also the (a, b)
/// derivatives when v depends on (a, b).
template <class T>
void phis(const T v[3], const EpsCoefficients& k, Objective objective, T out[7]) {
    const T r = inv_sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    T n[3], n2[3], n4[3];
    for (int i = 0; i < 3; ++i) {
        n[i] = v[i] * r;
        n2[i] = n[i] * n[i];
        n4[i] = n2[i] * n2[i];
    }
    const T s = n4[0] + n4[1] + n4[2];
    const T A = k.one_minus_3 + k.four * s;
    T Dv[3];
    for (int i = 0; i < 3; ++i)
        Dv[i] = n[i] * (n2[i] - s);
    const bool growth = objective == Objective::Growth;
    const T w = growth ? inv(A * A) : constant<T>(1.0);
    for (int i = 0; i < 3; ++i) {
        const T inner = 3.0 * n2[i] - 6.0 * n4[i] - s + 4.0 * (s * n2[i]);
        const T Hii = A * A + k.thirty_two * (A * Dv[i] * n[i]) +
                      k.sixteen * (k.sixteen * (Dv[i] * Dv[i]) + A * inner);
        out[i] = growth ? Hii * w : -Hii;
    }
    int m = 3;
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j) {
            const T Hij = k.two_fifty_six_sq * (Dv[i] * Dv[j]) +
                          k.thirty_two * (A * (n[i] * n[j]) * (s - n2[i] - n2[j]));
            out[m++] = growth ? Hij * w : -Hij;
        }
    out[6] = w;
}

/// phi_k with their (a, b) derivatives for n the direction of v = (1, a, b) (face 0),
/// (a, 1, b) (face 1) or (a, b, 1) (face 2).
void face_phis(int face, const D2& a, const D2& b, const EpsCoefficients& k, Objective objective,
               D2 out[7]) {
    const D2 one = cst(1.0);
    const D2 v[3] = {face == 0 ? one : a, face == 0 ? a : face == 1 ? one : b, face == 2 ? one : b};
    phis(v, k, objective, out);
}

int offdiag(int i, int j) { // index of phi_{ij} for i != j
    const int a = std::min(i, j), b = std::max(i, j);
    return a == 0 ? (b == 1 ? 3 : 4) : 5;
}

// ── The problem and its theta quantities ───────────────────────────────────

struct Problem {
    Objective objective;
    double eps, beta, W0;
    double h[3];
    EpsCoefficients k;
    Iv W;          // W0^2
    Iv inv_h[3];   // 1 / h_d
    Iv inv_h2[3];  // 1 / h_d^2
    Iv beta_i;     // beta
    Iv two_beta_W; // 2 beta W0^2
    Iv four_beta;  // 4 beta
    double kappa;  // +2 growth, -2 damping
};

Problem make_problem(Objective objective, double eps, double beta, double W0, const double h[3]) {
    Problem P{};
    P.objective = objective;
    P.eps = eps;
    P.beta = beta;
    P.W0 = W0;
    std::copy(h, h + 3, P.h);
    P.k = eps_coefficients(eps);
    P.W = sq(Iv(W0));
    for (int d = 0; d < 3; ++d) {
        P.inv_h[d] = inv(Iv(h[d]));
        P.inv_h2[d] = inv(sq(Iv(h[d])));
    }
    P.beta_i = Iv(beta);
    P.two_beta_W = Iv(2.0) * P.beta_i * P.W;
    P.four_beta = Iv(4.0) * P.beta_i;
    P.kappa = objective == Objective::Growth ? 2.0 : -2.0;
    return P;
}

/// theta-dependent quantities over a theta box (or a point): c_d, q_d, their first (c1,
/// q1) and second (c2, q2) derivatives in theta_d, mu, W mu + 2 and psi.
struct TData {
    Iv c[3], q[3], c1[3], q1[3], c2[3], q2[3];
    Iv mu, Wm2, psi[7];
};

TData theta_data(const Problem& P, const double lo[3], const double hi[3]) {
    TData t;
    for (int d = 0; d < 3; ++d) {
        const Iv s = isin(lo[d], hi[d]), c = icos(lo[d], hi[d]);
        t.c[d] = (Iv(2.0) - Iv(2.0) * c) * P.inv_h2[d];
        t.c[d].lo = std::max(t.c[d].lo, 0.0); // c_d = 2 (1 - cos) / h^2 >= 0 exactly
        t.q[d] = s * P.inv_h[d];
        t.c1[d] = Iv(2.0) * s * P.inv_h2[d];
        t.c2[d] = Iv(2.0) * c * P.inv_h2[d];
        t.q1[d] = c * P.inv_h[d];
        t.q2[d] = -(s * P.inv_h[d]);
    }
    t.mu = t.c[0] + t.c[1] + t.c[2];
    t.Wm2 = P.W * t.mu + Iv(2.0);
    for (int d = 0; d < 3; ++d)
        t.psi[d] = P.W * t.c[d];
    t.psi[3] = Iv(2.0) * P.W * (t.q[0] * t.q[1]);
    t.psi[4] = Iv(2.0) * P.W * (t.q[0] * t.q[2]);
    t.psi[5] = Iv(2.0) * P.W * (t.q[1] * t.q[2]);
    t.psi[6] = Iv(P.kappa) - P.beta_i * sq(t.Wm2);
    return t;
}

/// Interval value of sum_k phi_k psi_k with the theta quantities of `t`.
Iv natural_value(const Iv phi[7], const TData& t) {
    Iv v(0.0);
    for (int k = 0; k < 7; ++k)
        v = v + phi[k] * t.psi[k];
    return v;
}

/// dF/dtheta_d for phi in the given intervals and theta in t (a box or a point).
Iv theta_gradient(const Problem& P, const Iv phi[7], const TData& t, int d) {
    Iv g = P.W * phi[d] * t.c1[d];
    for (int e = 0; e < 3; ++e)
        if (e != d)
            g = g + Iv(2.0) * P.W * phi[offdiag(d, e)] * (t.q1[d] * t.q[e]);
    return g - P.two_beta_W * phi[6] * t.Wm2 * t.c1[d];
}

// ── Bounds ──────────────────────────────────────────────────────────────────

/// Upper bound of sum_k phi_k psi_k over the theta box for phi in the given intervals,
/// expanding beta (W mu + 2)^2 so that each axis's terms alpha_d c_d - b c_d^2 are
/// maximised exactly over c_d's range (a concave parabola) and the pair terms
/// 2 W phi_de q_d q_e - 2 b c_d c_e by interval extension.
double decomposed(const Problem& P, const Iv phi[7], const TData& t) {
    const Iv b = phi[6] * P.beta_i * sq(P.W);  // >= 0
    const Iv lin = phi[6] * P.four_beta * P.W; // subtracted from alpha
    double total = (phi[6] * (Iv(P.kappa) - P.four_beta)).hi;
    for (int d = 0; d < 3; ++d) {
        // c_d >= 0: alpha c_d <= alpha.hi c_d and -b c_d^2 <= -b.lo c_d^2
        const Iv alpha = phi[d] * P.W - lin;
        const double al = alpha.hi, bb = b.lo, lo = t.c[d].lo, hi = t.c[d].hi;
        auto at = [&](double c) {
            const Iv C(c);
            return (Iv(al) * C - Iv(bb) * (C * C)).hi;
        };
        double m = std::max(at(lo), at(hi));
        if (bb > 0.0) {
            // the vertex al / (2 bb), where the parabola reaches al^2 / (4 bb)
            const Iv vertex = Iv(al) * inv(Iv(2.0) * Iv(bb));
            if (vertex.hi >= lo && vertex.lo <= hi)
                m = std::max(m, (sq(Iv(al)) * inv(Iv(4.0) * Iv(bb))).hi);
        }
        total = add_up(total, m);
    }
    for (int d = 0; d < 3; ++d)
        for (int e = d + 1; e < 3; ++e) {
            const Iv pair = Iv(2.0) * P.W * phi[offdiag(d, e)] * (t.q[d] * t.q[e]) -
                            Iv(2.0) * b * (t.c[d] * t.c[e]);
            total = add_up(total, pair.hi);
        }
    return total;
}

/// Second-order Taylor bound of F(n_c, theta) over the theta box [lo, hi] (t: its
/// quantities): value and gradient enclosed at the centre, Hessian over the box.
double taylor_bound(const Problem& P, const Iv phi[7], const TData& t, const double lo[3],
                    const double hi[3]) {
    double c[3], r[3];
    for (int d = 0; d < 3; ++d) {
        c[d] = 0.5 * (lo[d] + hi[d]);
        r[d] = std::max(up(hi[d] - c[d]), up(c[d] - lo[d]));
    }
    const TData tc = theta_data(P, c, c);
    double taylor = natural_value(phi, tc).hi;
    for (int d = 0; d < 3; ++d)
        taylor = add_up(taylor, mul_up(mag(theta_gradient(P, phi, tc, d)), r[d]));
    for (int d = 0; d < 3; ++d)
        for (int e = 0; e < 3; ++e) {
            if (d == e) {
                Iv Hdd = P.W * phi[d] * t.c2[d];
                for (int f = 0; f < 3; ++f)
                    if (f != d)
                        Hdd = Hdd + Iv(2.0) * P.W * phi[offdiag(d, f)] * (t.q2[d] * t.q[f]);
                Hdd = Hdd - P.two_beta_W * phi[6] * P.W * sq(t.c1[d]) -
                      P.two_beta_W * phi[6] * t.Wm2 * t.c2[d];
                // 1/2 H_dd delta_d^2 <= 1/2 max(H_dd, 0) r_d^2
                taylor =
                    add_up(taylor, mul_up(mul_up(mul_up(0.5, std::max(Hdd.hi, 0.0)), r[d]), r[d]));
            } else {
                const Iv Hde = Iv(2.0) * P.W * phi[offdiag(d, e)] * (t.q1[d] * t.q1[e]) -
                               P.two_beta_W * phi[6] * P.W * (t.c1[d] * t.c1[e]);
                taylor = add_up(taylor, mul_up(mul_up(mul_up(0.5, mag(Hde)), r[d]), r[e]));
            }
        }
    return taylor;
}

/// The expansion of phi over one n-box: values and gradients at the centre, the
/// second-order remainder e_k, and phi over the whole box.
struct NData {
    Iv phi[7], ga[7], gb[7];
    double e[7];
    Iv phi_box[7];
    double ra = 0.0, rb = 0.0;
};

/// 1. phi over the whole n-box, per-axis decomposition in theta.
double joint_bound(const Problem& P, const NData& nd, const TData& t) {
    return decomposed(P, nd.phi_box, t);
}

/// |g_a| r_a + |g_b| r_b + sum_k e_k |psi_k|: an upper bound of F(n, theta) - F(n_c, theta)
/// over the n-box and the theta box of t.
double n_remainder(const NData& nd, const TData& t) {
    Iv ga(0.0), gb(0.0);
    double rest = 0.0;
    for (int k = 0; k < 7; ++k) {
        ga = ga + nd.ga[k] * t.psi[k];
        gb = gb + nd.gb[k] * t.psi[k];
        rest = add_up(rest, mul_up(nd.e[k], iabs(t.psi[k]).hi));
    }
    rest = add_up(rest, mul_up(mag(ga), nd.ra));
    return add_up(rest, mul_up(mag(gb), nd.rb));
}

/// 2. F(n_c, theta) over the theta box (interval extension, and the per-axis
/// decomposition unless the box is a point) plus the remainder in n.
double second_order_n_bound(const Problem& P, const NData& nd, const TData& t, bool point,
                            double rest) {
    double smooth = natural_value(nd.phi, t).hi;
    if (!point)
        smooth = std::min(smooth, decomposed(P, nd.phi, t));
    return add_up(smooth, rest);
}

/// 3. Second order in theta as well, plus the remainder in n.
double second_order_theta_bound(const Problem& P, const NData& nd, const TData& t,
                                const double lo[3], const double hi[3], double rest) {
    return add_up(taylor_bound(P, nd.phi, t, lo, hi), rest);
}

bool is_point(const double lo[3], const double hi[3]) {
    return lo[0] == hi[0] && lo[1] == hi[1] && lo[2] == hi[2];
}

/// Upper bound of F over (n-box x theta box [lo, hi]), t the theta box's quantities: the
/// smallest of the three bounds, from the cheapest to the dearest, stopping once one is
/// <= cutoff (at a point the theta Taylor bound is the interval value again).
double box_bound(const Problem& P, const NData& nd, const TData& t, const double lo[3],
                 const double hi[3], double cutoff) {
    const double joint = joint_bound(P, nd, t);
    if (joint <= cutoff)
        return joint;
    const double rest = n_remainder(nd, t);
    const bool point = is_point(lo, hi);
    const double bound = std::min(joint, second_order_n_bound(P, nd, t, point, rest));
    if (point || bound <= cutoff)
        return bound;
    return std::min(bound, second_order_theta_bound(P, nd, t, lo, hi, rest));
}

/// True if the theta gradient of F over (n-box x theta-box) excludes zero in some
/// component: no stationary point, so not the maximum of the periodic F.
bool theta_monotone(const Problem& P, const NData& nd, const TData& t) {
    for (int d = 0; d < 3; ++d) {
        const Iv g = theta_gradient(P, nd.phi_box, t, d);
        if (g.lo > 0.0 || g.hi < 0.0)
            return true;
    }
    return false;
}

struct TBox {
    double lo[3], hi[3], ub;
};
struct NBox {
    int face;
    double a0, a1, b0, b1, ub;
    std::vector<TBox> theta;
};

NData expand(const Problem& P, int face, double a0, double a1, double b0, double b1) {
    NData nd;
    const double ac = 0.5 * (a0 + a1), bc = 0.5 * (b0 + b1);
    nd.ra = std::max(up(a1 - ac), up(ac - a0));
    nd.rb = std::max(up(b1 - bc), up(bc - b0));
    D2 out[7], box[7];
    face_phis(face, interval::var_a(Iv(ac)), interval::var_b(Iv(bc)), P.k, P.objective, out);
    face_phis(face, interval::var_a(Iv(a0, a1)), interval::var_b(Iv(b0, b1)), P.k, P.objective,
              box);
    for (int k = 0; k < 7; ++k) {
        nd.phi[k] = out[k].v;
        nd.ga[k] = out[k].da;
        nd.gb[k] = out[k].db;
        nd.phi_box[k] = box[k].v;
        // Lagrange remainder: 1/2 (|aa| ra^2 + 2 |ab| ra rb + |bb| rb^2)
        const double aa = mul_up(mul_up(mag(box[k].aa), nd.ra), nd.ra);
        const double ab = mul_up(mul_up(mul_up(2.0, mag(box[k].ab)), nd.ra), nd.rb);
        const double bb = mul_up(mul_up(mag(box[k].bb), nd.rb), nd.rb);
        nd.e[k] = mul_up(0.5, add_up(add_up(aa, ab), bb));
    }
    return nd;
}

/// Enclosure of F at the mode (the direction of n, theta), every operation rounded outward.
Iv enclose_mode(const Problem& P, const double n[3], const double th[3]) {
    const Iv v[3] = {Iv(n[0]), Iv(n[1]), Iv(n[2])};
    Iv phi[7];
    phis(v, P.k, P.objective, phi);
    return natural_value(phi, theta_data(P, th, th));
}

// ── Point evaluation (search guidance only; never part of a bound) ─────────

double point_value(const Problem& P, const double n[3], const double th[3]) {
    double H[3][3];
    gradient_energy_hessian(n, P.eps, H);
    const double s =
        n[0] * n[0] * n[0] * n[0] + n[1] * n[1] * n[1] * n[1] + n[2] * n[2] * n[2] * n[2];
    const double A = 1.0 - 3.0 * P.eps + 4.0 * P.eps * s;
    const double W = P.W0 * P.W0;
    double c[3], q[3], mu = 0.0, sigma = 0.0;
    for (int d = 0; d < 3; ++d) {
        c[d] = 2.0 * (1.0 - std::cos(th[d])) / (P.h[d] * P.h[d]);
        q[d] = std::sin(th[d]) / P.h[d];
        mu += c[d];
    }
    for (int d = 0; d < 3; ++d) {
        sigma += H[d][d] * c[d];
        for (int e = 0; e < 3; ++e)
            if (e != d)
                sigma += H[d][e] * q[d] * q[e];
    }
    const double L = (W * mu + 2.0) * (W * mu + 2.0);
    if (P.objective == Objective::Growth)
        return (W * sigma + 2.0 - P.beta * L) / (A * A);
    return -(W * sigma + 2.0 + P.beta * L);
}

void face_direction(int face, double a, double b, double n[3]) {
    double v[3] = {1.0, a, b};
    if (face == 1) {
        v[0] = a;
        v[1] = 1.0;
    } else if (face == 2) {
        v[0] = a;
        v[1] = b;
        v[2] = 1.0;
    }
    const double r = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    for (int i = 0; i < 3; ++i)
        n[i] = v[i] / r;
}

/// The best mode of a coarse grid, refined by compass search: a mode whose value lets the
/// branch and bound prune from the start.
LiquidMode seed(const Problem& P, bool cubic) {
    constexpr int G = 4;
    double best = -std::numeric_limits<double>::infinity(), x[5] = {0, 0, 0, 0, 0};
    int face = 0;
    for (int f = 0; f < (cubic ? 1 : 3); ++f)
        for (int ia = 0; ia < G; ++ia)
            for (int ib = 0; ib < G; ++ib)
                for (int i0 = 0; i0 < G; ++i0)
                    for (int i1 = 0; i1 < 2 * G; ++i1)
                        for (int i2 = 0; i2 < 2 * G; ++i2) {
                            const double y[5] = {(ia + 0.5) / G, (ib + 0.5) / G,
                                                 kPi * (i0 + 0.5) / G, -kPi + kPi * (i1 + 0.5) / G,
                                                 -kPi + kPi * (i2 + 0.5) / G};
                            double n[3];
                            face_direction(f, y[0], y[1], n);
                            const double v = point_value(P, n, y + 2);
                            if (v > best) {
                                best = v;
                                face = f;
                                std::copy(y, y + 5, x);
                            }
                        }
    double step[5] = {0.125, 0.125, 0.25, 0.25, 0.25};
    for (bool small = false; !small;) {
        bool moved = false;
        for (int k = 0; k < 5; ++k)
            for (const double sgn : {-1.0, 1.0}) {
                double y[5];
                std::copy(x, x + 5, y);
                y[k] += sgn * step[k];
                if (k < 2)
                    y[k] = std::clamp(y[k], 0.0, 1.0);
                double n[3];
                face_direction(face, y[0], y[1], n);
                const double v = point_value(P, n, y + 2);
                if (v > best) {
                    best = v;
                    std::copy(y, y + 5, x);
                    moved = true;
                }
            }
        if (!moved) {
            small = true;
            for (double& s : step) {
                s *= 0.5;
                small = small && s < 1e-12;
            }
        }
    }
    LiquidMode mode{};
    face_direction(face, x[0], x[1], mode.n);
    std::copy(x + 2, x + 5, mode.theta);
    return mode;
}

// kLiquidModeEvaluationBudget (2e7) is a resource guard, not a tolerance: the searches for
// the shipped physics take 1.2e4 to 1.6e5 box evaluations, those of the tests up to 1.5e6
// (a few microseconds each), and a damping search close to the smallest beta that regularises
// eps needs more the closer it is (at eps 0.12, h 0.4: 3 s at beta 0.052949, min
// Lam + beta L in [0, 0.0015]; beyond the budget at beta 0.0529346); past the budget the
// search reports its range instead of running for minutes.
/// theta boxes an inner search may split before it hands an undecided n-box back for
/// splitting (only the order of work; every pruning decision is a proven bound).
constexpr long kInnerBudget = 256;

struct Search {
    const Problem& P;
    /// Decision mode (damping): prove F <= 0 everywhere, or stop at the first mode whose
    /// enclosure of F lies above 0.
    bool decide_sign = false;
    /// The largest lower end of the enclosures of F at the recorded modes, and its mode.
    double best = -std::numeric_limits<double>::infinity();
    LiquidMode mode{};
    bool witness = false;
    long evaluations = 0;
    long budget = kLiquidModeEvaluationBudget;
    /// Set once the budget is spent or a theta box cannot be split. The inner searches of
    /// the direction box in hand still end their usual way (pruned, or every undecided
    /// box handed back, after at most kInnerBudget splits each), then run() stops and
    /// reports the range.
    const char* stopped = nullptr;

    double cutoff() const {
        if (decide_sign)
            return 0.0;
        if (best == -std::numeric_limits<double>::infinity())
            return best; // nothing recorded yet: nothing can be pruned
        return best + kLiquidModeTolerance * std::max(1.0, std::fabs(best));
    }
    bool witnessed() const { return witness; }

    /// Records the mode (n, th) with the lower end of its enclosure of F; in decision
    /// mode a lower end above 0 is a witness and ends the search. The enclosure is
    /// computed only where the double evaluation exceeds the best lower end: skipping a
    /// mode never makes a recorded value or a pruning decision unsound, it can only keep
    /// best lower.
    void record(const double n[3], const double th[3]) {
        if (witness || !(point_value(P, n, th) > best))
            return;
        const double v = enclose_mode(P, n, th).lo;
        if (v > best || (decide_sign && v > 0.0)) {
            best = v;
            std::copy(n, n + 3, mode.n);
            std::copy(th, th + 3, mode.theta);
            witness = decide_sign && v > 0.0;
        }
    }

    void count() {
        if (++evaluations > budget && stopped == nullptr)
            stopped = "spent its budget of box evaluations";
    }

    void consider(const NBox& B, const double th[3]) {
        double n[3];
        face_direction(B.face, 0.5 * (B.a0 + B.a1), 0.5 * (B.b0 + B.b1), n);
        record(n, th);
    }

    /// Bounds the n-box over the theta boxes it inherited; returns an upper bound and
    /// the theta boxes that may still exceed the cutoff (empty: the n-box is pruned).
    double inner(const NBox& B, const NData& nd, std::vector<TBox>& survivors) {
        auto cmp = [](const TBox& x, const TBox& y) { return x.ub < y.ub; };
        std::priority_queue<TBox, std::vector<TBox>, decltype(cmp)> pq(cmp);
        for (TBox T : B.theta) {
            const TData t = theta_data(P, T.lo, T.hi);
            count();
            if (theta_monotone(P, nd, t))
                continue;
            T.ub = box_bound(P, nd, t, T.lo, T.hi, cutoff());
            if (T.ub > cutoff())
                pq.push(T);
        }
        survivors.clear();
        auto drain = [&](const TBox* extra) {
            if (extra != nullptr)
                survivors.push_back(*extra);
            while (!pq.empty()) {
                survivors.push_back(pq.top());
                pq.pop();
            }
        };
        for (long spent = 0; !pq.empty() && !witnessed();) {
            const TBox T = pq.top();
            if (T.ub <= cutoff())
                return cutoff(); // pruned
            if (++spent > kInnerBudget) {
                drain(nullptr);
                return T.ub;
            }
            pq.pop();
            double c[3];
            for (int d = 0; d < 3; ++d)
                c[d] = 0.5 * (T.lo[d] + T.hi[d]);
            consider(B, c);
            // Both bounds at the single point c already exceed the cutoff: refining theta
            // cannot prune this n-box; the outer search splits it.
            count();
            const double at_point = box_bound(P, nd, theta_data(P, c, c), c, c, cutoff());
            if (at_point > cutoff()) {
                drain(&T);
                return std::max(T.ub, at_point);
            }
            int k = 0;
            for (int d = 1; d < 3; ++d)
                if (T.hi[d] - T.lo[d] > T.hi[k] - T.lo[k])
                    k = d;
            TBox L = T, R = T;
            L.hi[k] = R.lo[k] = 0.5 * (T.lo[k] + T.hi[k]);
            if (L.hi[k] == T.lo[k] || L.hi[k] == T.hi[k]) {
                stopped = "met a wave-number box it cannot split in double precision";
                drain(&T);
                return std::max(T.ub, at_point);
            }
            for (TBox* X : {&L, &R}) {
                count();
                const TData t = theta_data(P, X->lo, X->hi);
                if (theta_monotone(P, nd, t))
                    continue;
                X->ub = box_bound(P, nd, t, X->lo, X->hi, cutoff());
                if (X->ub > cutoff())
                    pq.push(*X);
            }
        }
        return cutoff();
    }

    double run(bool cubic) {
        auto cmp = [](const NBox& x, const NBox& y) { return x.ub < y.ub; };
        std::priority_queue<NBox, std::vector<NBox>, decltype(cmp)> pq(cmp);
        constexpr int G = 4;
        using liquid_modes_detail::kThetaHi;
        using liquid_modes_detail::kThetaLo;
        const TBox whole{{kThetaLo[0], kThetaLo[1], kThetaLo[2]},
                         {kThetaHi[0], kThetaHi[1], kThetaHi[2]},
                         std::numeric_limits<double>::infinity()};
        for (int f = 0; f < (cubic ? 1 : 3); ++f)
            for (int ia = 0; ia < G; ++ia)
                for (int ib = 0; ib < G; ++ib) {
                    if (cubic && ib > ia)
                        continue; // n_x >= n_y >= n_z: the triangle 0 <= b <= a <= 1
                    pq.push(NBox{f,
                                 double(ia) / G,
                                 double(ia + 1) / G,
                                 double(ib) / G,
                                 double(ib + 1) / G,
                                 std::numeric_limits<double>::infinity(),
                                 {whole}});
                }
        while (stopped == nullptr && !witnessed() && !pq.empty() && pq.top().ub > cutoff()) {
            const bool split_a = pq.top().a1 - pq.top().a0 >= pq.top().b1 - pq.top().b0;
            const double lo = split_a ? pq.top().a0 : pq.top().b0;
            const double hi = split_a ? pq.top().a1 : pq.top().b1;
            const double mid = 0.5 * (lo + hi);
            if (mid == lo || mid == hi) { // the box stays in the queue
                stopped = "met a direction box it cannot split in double precision";
                break;
            }
            NBox B = pq.top();
            pq.pop();
            NBox L = B, R = B;
            if (split_a)
                L.a1 = R.a0 = mid;
            else
                L.b1 = R.b0 = mid;
            for (NBox* X : {&L, &R}) {
                const NData nd = expand(P, X->face, X->a0, X->a1, X->b0, X->b1);
                std::vector<TBox> survivors;
                X->ub = inner(*X, nd, survivors);
                X->theta = std::move(survivors);
                if (!X->theta.empty() && X->ub > cutoff())
                    pq.push(std::move(*X));
            }
        }
        if (witnessed())
            return best;
        if (stopped != nullptr && !pq.empty() && pq.top().ub > cutoff())
            throw_undecided(pq.top().ub);
        if (decide_sign)
            return 0.0; // every box is proven <= 0
        return up(cutoff());
    }

    /// F <= upper everywhere (the boxes left are below their bounds, upper the largest;
    /// the others are below the cutoff), and F >= best at mode.
    [[noreturn]] void throw_undecided(double upper) const {
        const bool growth = P.objective == Objective::Growth;
        const double lo = growth ? best : -upper, hi = growth ? upper : -best;
        throw LiquidModeSearchExhausted(
            std::string("liquid-mode search ") + stopped + " (" + std::to_string(evaluations) +
                " of " + std::to_string(budget) + ") without a certificate: the " +
                (growth ? "maximum of (Lam - beta L) / A(n)^2" : "minimum of Lam + beta L") +
                " lies in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]",
            lo, hi, mode);
    }
};

void check_inputs(double eps, double beta, double W0, const double h[3]) {
    if (!(eps >= 0.0 && eps < 1.0 / 3.0) || !(beta >= 0.0) || !(W0 > 0.0) || !std::isfinite(W0) ||
        !std::isfinite(beta))
        throw std::invalid_argument(
            "liquid-mode bound: needs 0 <= epsilon < 1/3, beta >= 0, W0 > 0");
    for (int d = 0; d < 3; ++d)
        if (!(h[d] > 0.0) || !std::isfinite(h[d]) || !std::isfinite(1.0 / (h[d] * h[d])))
            throw std::invalid_argument(
                "liquid-mode bound: spacings must be positive, with 1/h^2 finite");
    if (!std::isfinite(W0 * W0) || !std::isfinite(W0 * W0 * W0 * W0 * beta))
        throw std::invalid_argument("liquid-mode bound: W0^2 and beta W0^4 must be finite");
}

/// enclose_mode needs |n|^2 enclosed away from 0 and infinity (inv_sqrt's precondition; a
/// direction whose squared length underflows or overflows has none), and finite theta.
/// The search's own modes are unit vectors.
void check_mode(const double n[3], const double theta[3]) {
    const Iv r2 = sq(Iv(n[0])) + sq(Iv(n[1])) + sq(Iv(n[2]));
    if (!(r2.lo > 0.0) || !std::isfinite(r2.hi) || !std::isfinite(theta[0]) ||
        !std::isfinite(theta[1]) || !std::isfinite(theta[2]))
        throw std::invalid_argument(
            "liquid-mode mode: needs a direction whose squared length is enclosed between 0 "
            "and infinity, and finite wave numbers");
}

bool cubic_spacing(const double h[3]) {
    return h[0] == h[1] && h[1] == h[2];
}

/// The branch and bound from the recorded mode `start`. Growth: an upper bound of max F,
/// and the lower end of the enclosure at the best mode. Damping (decision mode): 0 when
/// F <= 0 is proven, else a witness whose lower end is > 0.
LiquidModeBound search(const Problem& P, const LiquidMode& start, long budget) {
    Search S{P, P.objective == Objective::Damping};
    S.budget = budget;
    S.record(start.n, start.theta);
    const double bound = S.run(cubic_spacing(P.h));
    return {bound, S.best, S.mode};
}

/// Damping in terms of Lam + beta L = -F: the bound 0 (every mode >= 0) or the upper end
/// < 0 of a witness's enclosure, and the smallest upper end recorded.
LiquidModeBound damping_result(LiquidModeBound r) {
    r.bound = r.bound == 0.0 ? 0.0 : -r.bound; // +0 when every box is proven <= 0
    r.attained = -r.attained;
    return r;
}

} // namespace

LiquidModeBound liquid_gradient_growth(double epsilon, double beta, double W0, const double h[3],
                                       long budget) {
    check_inputs(epsilon, beta, W0, h);
    const Problem P = make_problem(Objective::Growth, epsilon, beta, W0, h);
    return search(P, seed(P, cubic_spacing(h)), budget);
}

LiquidModeBound liquid_gradient_damping(double epsilon, double beta, double W0, const double h[3],
                                        long budget) {
    check_inputs(epsilon, beta, W0, h);
    const Problem P = make_problem(Objective::Damping, epsilon, beta, W0, h);
    return damping_result(search(P, seed(P, cubic_spacing(h)), budget));
}

namespace liquid_modes_detail {

void phi(int face, interval::Iv a, interval::Iv b, double epsilon, Objective objective,
         interval::D2 out[7]) {
    face_phis(face, interval::var_a(a), interval::var_b(b), eps_coefficients(epsilon), objective,
              out);
}

BoxBounds box_bounds(Objective objective, double epsilon, double beta, double W0, const double h[3],
                     int face, double a0, double a1, double b0, double b1, const double theta_lo[3],
                     const double theta_hi[3]) {
    check_inputs(epsilon, beta, W0, h);
    const Problem P = make_problem(objective, epsilon, beta, W0, h);
    const NData nd = expand(P, face, a0, a1, b0, b1);
    const TData t = theta_data(P, theta_lo, theta_hi);
    const double rest = n_remainder(nd, t);
    return {joint_bound(P, nd, t),
            second_order_n_bound(P, nd, t, is_point(theta_lo, theta_hi), rest),
            second_order_theta_bound(P, nd, t, theta_lo, theta_hi, rest), theta_monotone(P, nd, t)};
}

ThetaBounds theta_bounds(Objective objective, double epsilon, double beta, double W0,
                         const double h[3], const interval::Iv phi[7], const double theta_lo[3],
                         const double theta_hi[3]) {
    check_inputs(epsilon, beta, W0, h);
    const Problem P = make_problem(objective, epsilon, beta, W0, h);
    const TData t = theta_data(P, theta_lo, theta_hi);
    return {decomposed(P, phi, t), natural_value(phi, t).hi,
            taylor_bound(P, phi, t, theta_lo, theta_hi)};
}

interval::Iv mode_enclosure(Objective objective, double epsilon, double beta, double W0,
                            const double h[3], const double n[3], const double theta[3]) {
    check_inputs(epsilon, beta, W0, h);
    check_mode(n, theta);
    return enclose_mode(make_problem(objective, epsilon, beta, W0, h), n, theta);
}

LiquidModeBound search_from(Objective objective, double epsilon, double beta, double W0,
                            const double h[3], const LiquidMode& start, long budget) {
    check_inputs(epsilon, beta, W0, h);
    check_mode(start.n, start.theta);
    const LiquidModeBound r = search(make_problem(objective, epsilon, beta, W0, h), start, budget);
    return objective == Objective::Damping ? damping_result(r) : r;
}

} // namespace liquid_modes_detail

} // namespace ac
