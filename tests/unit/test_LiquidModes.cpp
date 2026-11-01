// The certified liquid-mode search (core/LiquidModes.hpp, docs/THEORY.md 10.1) and the
// interval arithmetic it is built from (core/Interval.hpp), against binary128 references
// (common/LiquidModeReference.hpp). Every check below is an exact comparison with a
// binary128 value, or a bound derived in its comment.
#include "common/LiquidModeReference.hpp"
#include "core/Interval.hpp"
#include "core/LiquidModes.hpp"
#include "core/LiquidModesDetail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace ac;
using ac::interval::D2;
using ac::interval::Iv;
using ac::liquid_modes_detail::Objective;
using ac::test::Q;
using ac::test::qstr;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kMax = std::numeric_limits<double>::max();
const Q kPiQ = acosq(-1); // pi in binary128 (M_PIq needs the Q literal suffix, a GNU extension)

/// x in [r.lo, r.hi], compared in binary128 (an infinite endpoint bounds nothing).
bool contains(Iv r, Q x) {
    return Q(r.lo) <= x && x <= Q(r.hi);
}
std::string str(Iv r) {
    return "[" + qstr(r.lo) + ", " + qstr(r.hi) + "]";
}

/// gamma_k = k u / (1 - k u), u = 2^-53: the relative bound of k roundings to nearest
/// (Higham, Accuracy and Stability of Numerical Algorithms, 3.1), in binary128.
Q gamma(int k) {
    const Q u = ldexpq(1, -53);
    return k * u / (1 - k * u);
}

/// A double in +-[2^lo, 2^hi) with a random significand, or 0 one time in sixteen.
double random_double(std::mt19937_64& rng, int lo, int hi) {
    std::uniform_int_distribution<int> pick(0, 15), exponent(lo, hi - 1), sign(0, 1);
    std::uniform_real_distribution<double> significand(1.0, 2.0);
    if (pick(rng) == 0)
        return 0.0;
    const double x = std::ldexp(significand(rng), exponent(rng));
    return sign(rng) == 0 ? x : -x;
}

Iv random_interval(std::mt19937_64& rng, int lo, int hi) {
    const double x = random_double(rng, lo, hi);
    std::uniform_int_distribution<int> pick(0, 3);
    const double y = pick(rng) == 0 ? x : random_double(rng, lo, hi); // a point one time in four
    return {std::min(x, y), std::max(x, y)};
}

/// Points of [r.lo, r.hi]: its ends and two random doubles inside it.
std::vector<double> points_of(Iv r, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> t(0.0, 1.0);
    std::vector<double> p = {r.lo, r.hi};
    for (int i = 0; i < 2; ++i)
        p.push_back(std::clamp(r.lo + t(rng) * (r.hi - r.lo), r.lo, r.hi));
    return p;
}

} // namespace

// ── The interval arithmetic ─────────────────────────────────────────────────

// Each operation encloses its exact result for every argument in its intervals:
// sums of doubles between 2^-29 and 2^29 in magnitude and products of any two doubles are
// exact in binary128 (at most 111 and 106 significant bits); quotients and square roots are
// rounded there 2^-113 relative, while an outward-rounded end lies at least half a unit
// in the last place of a double (2^-54 relative) beyond the exact value. mul_up and
// add_up bound one product / sum from above; down and up step past it.
TEST(Interval, ArithmeticEnclosesTheExactResults) {
    std::mt19937_64 rng(11);
    long checked = 0;
    for (int i = 0; i < 20000; ++i) {
        const Iv a = random_interval(rng, -29, 29), b = random_interval(rng, -29, 29);
        const Iv sum = a + b, diff = a - b, prod = a * b, neg = -a, square = sq(a);
        for (double x : points_of(a, rng)) {
            ASSERT_TRUE(contains(neg, -Q(x)));
            ASSERT_TRUE(contains(square, Q(x) * Q(x))) << x << " in " << str(a);
            ASSERT_EQ(interval::mag(a), std::max(std::fabs(a.lo), std::fabs(a.hi)));
            ASSERT_TRUE(contains(interval::iabs(a), fabsq(Q(x))));
            for (double y : points_of(b, rng)) {
                ASSERT_TRUE(contains(sum, Q(x) + Q(y))) << x << " + " << y << ": " << str(sum);
                ASSERT_TRUE(contains(diff, Q(x) - Q(y))) << x << " - " << y << ": " << str(diff);
                ASSERT_TRUE(contains(prod, Q(x) * Q(y))) << x << " * " << y << ": " << str(prod);
                ASSERT_TRUE(Q(interval::add_up(x, y)) >= Q(x) + Q(y)) << x << " + " << y;
                ASSERT_TRUE(Q(interval::mul_up(x, y)) >= Q(x) * Q(y)) << x << " * " << y;
                ++checked;
            }
        }
        if (a.lo > 0.0) {
            const Iv reciprocal = inv(a), root = inv_sqrt(a);
            for (double x : points_of(a, rng)) {
                ASSERT_TRUE(contains(reciprocal, 1 / Q(x))) << "1/" << x << ": " << str(reciprocal);
                ASSERT_TRUE(contains(root, 1 / sqrtq(Q(x)))) << "1/sqrt " << x << ": " << str(root);
            }
        }
        // down and up: strictly below / above, by at least one unit in the last place
        const double x = random_double(rng, -1074, 1024);
        ASSERT_LE(interval::down(x), std::nextafter(x, -kInf));
        ASSERT_GE(interval::up(x), std::nextafter(x, kInf));
    }
    EXPECT_GT(checked, 300000);
}

// down and up step at least one unit at every power of two and its neighbours, where the
// units below and above differ by a factor 2 (a step of half the size would round back to
// x above it): every binade from the smallest subnormal to the largest double.
TEST(Interval, StepsPassEveryBinadeEdge) {
    for (int k = -1074; k <= 1023; ++k)
        for (double sign : {1.0, -1.0}) {
            const double p = sign * std::ldexp(1.0, k);
            for (double x : {std::nextafter(p, -kInf), p, std::nextafter(p, kInf)}) {
                ASSERT_LE(interval::down(x), std::nextafter(x, -kInf)) << x;
                ASSERT_GE(interval::up(x), std::nextafter(x, kInf)) << x;
            }
        }
}

// inv and inv_sqrt round twice (sqrt, then the division) or once; at 200000 random points
// between 2^-60 and 2^60 each point interval encloses the binary128 value (an upward
// step after a downward-rounded square root would miss it about once in 8000 points).
TEST(Interval, InverseAndInverseRootEncloseEveryPoint) {
    std::mt19937_64 rng(19);
    std::uniform_real_distribution<double> significand(1.0, 2.0);
    std::uniform_int_distribution<int> exponent(-60, 59);
    for (int i = 0; i < 200000; ++i) {
        const double x = std::ldexp(significand(rng), exponent(rng));
        ASSERT_TRUE(contains(inv(Iv(x)), 1 / Q(x))) << x;
        ASSERT_TRUE(contains(inv_sqrt(Iv(x)), 1 / sqrtq(Q(x)))) << x;
    }
}

// Results beyond the largest double: an overflowed end becomes +-infinity on its own
// side and the largest double on the other (the exact value is finite and beyond it), a
// 0 times an overflowed factor is 0, and no end is ever NaN.
TEST(Interval, EnclosuresSurviveOverflow) {
    const Iv big = Iv(1e200) * Iv(1e200); // exact 1e400 (binary128 exponent range 16383)
    EXPECT_EQ(big.lo, kMax);
    EXPECT_EQ(big.hi, kInf);
    EXPECT_TRUE(contains(big, Q(1e200) * Q(1e200)));
    const Iv negative = Iv(-1e200) * Iv(1e200);
    EXPECT_EQ(negative.lo, -kInf);
    EXPECT_EQ(negative.hi, -kMax);
    EXPECT_TRUE(contains(Iv(kMax) + Iv(kMax), 2 * Q(kMax)));
    EXPECT_TRUE(contains(Iv(-kMax) - Iv(kMax), -2 * Q(kMax)));
    // 0 times (1e400): exactly 0, and the interval says so
    const Iv zero = Iv(0.0) * big;
    EXPECT_TRUE(contains(zero, 0));
    EXPECT_FALSE(std::isnan(zero.lo) || std::isnan(zero.hi));
    EXPECT_LE(zero.hi, std::numeric_limits<double>::denorm_min());
    EXPECT_GE(zero.lo, -std::numeric_limits<double>::denorm_min());
    // [-1, 1] times [1e400, 1e400]: products of both signs of infinity
    const Iv both = Iv(-1.0, 1.0) * big;
    EXPECT_TRUE(contains(both, Q(1e200) * Q(1e200)) && contains(both, -Q(1e200) * Q(1e200)));
    EXPECT_TRUE(contains(Iv(0.0, 1.0) * big, 0));
    // [0, 1] times [-inf, 2]: 0 * -inf is the exact 0, so the product's upper end is 2's
    const Iv half_open = Iv(0.0, 1.0) * Iv(-kInf, 2.0);
    EXPECT_EQ(half_open.lo, -kInf);
    EXPECT_EQ(half_open.hi, interval::up(2.0));
    EXPECT_TRUE(contains(Iv(-1.0, 0.0) * big, 0));
    EXPECT_TRUE(contains(big * Iv(-1.0, 0.0), -Q(1e200) * Q(1e200)));
    EXPECT_TRUE(contains(inv(big), 1 / (Q(1e200) * Q(1e200))));
    EXPECT_TRUE(contains(inv_sqrt(big), 1 / sqrtq(Q(1e200) * Q(1e200))));
    EXPECT_TRUE(contains(sq(big), Q(1e200) * Q(1e200) * Q(1e200) * Q(1e200)));
    EXPECT_TRUE(Q(interval::mul_up(0.0, kInf)) >= 0);
    EXPECT_EQ(interval::mul_up(kMax, 2.0), kInf);
    EXPECT_EQ(interval::down(std::numeric_limits<double>::quiet_NaN()), -kInf);
    EXPECT_EQ(interval::up(std::numeric_limits<double>::quiet_NaN()), kInf);
    EXPECT_EQ(interval::down(kInf), kMax);
    EXPECT_EQ(interval::up(-kInf), -kMax);
    EXPECT_EQ(interval::down(-kInf), -kInf);
    EXPECT_EQ(interval::up(kInf), kInf);
}

// sin and cos over sub-intervals of [-3.2, 3.2] (the wave-number boxes) enclose the
// binary128 values at their ends, at random interior points and at the extrema
// (+-pi/2 for sin, 0 and +-pi for cos) they hold; intervals with an end next to an
// extremum's abscissa, on either side, included.
TEST(Interval, SineAndCosineEncloseTheirRanges) {
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 52);
    std::vector<Iv> boxes;
    for (int i = 0; i < 20000; ++i) {
        const double w = unit(rng) < 0.125 ? 0.0 : 6.4 * std::ldexp(unit(rng), -scale(rng));
        const double lo = -3.2 + (6.4 - w) * unit(rng);
        boxes.push_back({lo, std::min(lo + w, 3.2)});
    }
    const double kPi = interval::kPi;
    for (double x : {0.5 * kPi, -0.5 * kPi, 0.0, kPi, -kPi}) {
        const double below = std::nextafter(x, -kInf), above = std::nextafter(x, kInf);
        for (double w : {0.0, 1e-12, 0.1}) {
            boxes.push_back({x - w, x});
            boxes.push_back({x, x + w});
            boxes.push_back({below - w, below});
            boxes.push_back({above, above + w});
            boxes.push_back({below, above});
        }
    }
    for (const Iv& box : boxes) {
        const double lo = std::max(box.lo, -3.2), hi = std::min(box.hi, 3.2);
        const Iv s = interval::isin(lo, hi), c = interval::icos(lo, hi);
        std::vector<Q> xs = {lo, hi};
        for (int k = 0; k < 4; ++k)
            xs.push_back(Q(std::clamp(lo + (hi - lo) * unit(rng), lo, hi)));
        for (Q x : {kPiQ / 2, -kPiQ / 2, Q(0), kPiQ, -kPiQ})
            if (Q(lo) <= x && x <= Q(hi))
                xs.push_back(x);
        for (Q x : xs) {
            ASSERT_TRUE(contains(s, sinq(x)))
                << "sin " << qstr(x) << " in [" << lo << ", " << hi << "]: " << str(s);
            ASSERT_TRUE(contains(c, cosq(x)))
                << "cos " << qstr(x) << " in [" << lo << ", " << hi << "]: " << str(c);
        }
    }
}

// ── The coefficients and the bounds ─────────────────────────────────────────

// The Hessian of the gradient energy: the closed form of LiquidModes.hpp is the
// derivative of the kernel's flux, and gradient_energy_hessian evaluates it.
//  * binary128, at unit vectors: the closed form against forward differentiation of the
//    flux. Both take fewer than 200 operations on values below 2^10 (|n| = 1, eps < 1/3),
//    so each errs by less than 200 * 2^-113 * 2^10 < 2^-95: they agree to 2^-94.
//  * double: gradient_energy_hessian against the binary128 closed form at the same double
//    n (both read it as a unit vector). Counting its roundings per term of the expanded
//    formula (a product carries the counts of both factors plus one, a sum the larger
//    count plus one: s 5, a 8, D_i 7, H_ii 22, H_ij 21), each term is the exact one times
//    at most 22 factors 1 + delta, |delta| <= 2^-53, and the terms, all signs made
//    positive, sum to at most 446 (|n_i| <= 1, s <= 1, a <= 1 + 7 eps <= 10/3,
//    |D_i| <= 2), so it errs by at most gamma_22 * 446 (Higham, Lemma 3.3 and 3.1);
//    binary128's own error is 2^-60 of that.
// Plus the closed forms (1 + eps)^2 along an axis, (1 + eps)(1 - 15 eps) across it,
// A^2 + 64 eps A / 9 at <111>, the identity at eps = 0, and exact symmetry.
TEST(LiquidModes, HessianIsTheDerivativeOfTheKernelFlux) {
    std::mt19937_64 rng(13);
    std::normal_distribution<double> normal;
    const Q quad_bound = ldexpq(1, -94);
    const Q double_bound = gamma(22) * 446 * (1 + ldexpq(1, -50));
    for (double eps : {0.0, 0.05, 1.0 / 15.0, 0.07, 0.12, 0.3, 0.33}) {
        for (int i = 0; i < 2000; ++i) {
            double v[3] = {normal(rng), normal(rng), normal(rng)};
            if (i < 4) { // axes and <111> too
                v[0] = 1.0;
                v[1] = i == 1 || i == 3 ? 1.0 : 0.0;
                v[2] = i >= 2 ? 1.0 : 0.0;
            }
            const Q vq[3] = {v[0], v[1], v[2]};
            const Q len = sqrtq(vq[0] * vq[0] + vq[1] * vq[1] + vq[2] * vq[2]);
            const Q nq[3] = {vq[0] / len, vq[1] / len, vq[2] / len};
            Q Hc[3][3], Hf[3][3];
            test::closed_form_hessian(nq, eps, Hc);
            test::flux_hessian(nq, eps, Hf);
            const double nd[3] = {static_cast<double>(nq[0]), static_cast<double>(nq[1]),
                                  static_cast<double>(nq[2])};
            const Q ndq[3] = {nd[0], nd[1], nd[2]};
            Q Hq[3][3];
            test::closed_form_hessian(ndq, eps, Hq);
            double H[3][3];
            gradient_energy_hessian(nd, eps, H);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) {
                    ASSERT_TRUE(fabsq(Hc[r][c] - Hf[r][c]) <= quad_bound)
                        << "eps " << eps << " H" << r << c << ": closed " << qstr(Hc[r][c])
                        << " flux " << qstr(Hf[r][c]);
                    ASSERT_TRUE(fabsq(Q(H[r][c]) - Hq[r][c]) <= double_bound)
                        << "eps " << eps << " H" << r << c << " = " << H[r][c] << ", binary128 "
                        << qstr(Hq[r][c]);
                    ASSERT_EQ(H[r][c], H[c][r]);
                }
        }
        // closed forms (to the double bound above)
        double H[3][3];
        const double x[3] = {1, 0, 0};
        gradient_energy_hessian(x, eps, H);
        const Q e = eps;
        EXPECT_TRUE(fabsq(Q(H[0][0]) - (1 + e) * (1 + e)) <= double_bound) << eps;
        EXPECT_TRUE(fabsq(Q(H[1][1]) - (1 + e) * (1 - 15 * e)) <= double_bound) << eps;
        EXPECT_EQ(H[0][1], 0.0);
        const Q r3 = 1 / sqrtq(Q(3)), a = 1 - 5 * e / 3;
        Q n111[3] = {r3, r3, r3}, H111[3][3];
        test::closed_form_hessian(n111, e, H111);
        for (int d = 0; d < 3; ++d)
            EXPECT_TRUE(fabsq(H111[d][d] - (a * a + 64 * e * a / 9)) <= quad_bound) << eps;
    }
    double I[3][3];
    const double odd[3] = {0.48, 0.6, std::sqrt(1.0 - 0.48 * 0.48 - 0.6 * 0.6)};
    gradient_energy_hessian(odd, 0.0, I);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_EQ(I[r][c], r == c ? 1.0 : 0.0);
}

// phi_k(n(a, b)) and its derivatives, enclosed over boxes of (a, b) on each face (the
// D2 arithmetic): the binary128 values at the box's corners, edge midpoints and centre
// lie in the value enclosure, and difference quotients lie in the derivative enclosures,
// since by the mean value theorem each equals a derivative at some point of the box:
//   (f(a1, b) - f(a0, b)) / (a1 - a0) = f_a(xi, b),
//   2 f[a0, am, a1](b) = f_aa(xi, b)  (second divided difference),
//   (f(a1, b1) - f(a1, b0) - f(a0, b1) + f(a0, b0)) / ((a1 - a0)(b1 - b0)) = f_ab(xi, eta).
// Box sides are at least 2^-10 and |phi_k| below 2^11, so with binary128 values within
// 2^-104 relative (2^-93 absolute) the first differences err by less than 2 2^-93 / 2^-10
// = 2^-82, the pure second ones (4 (f2 - 2 f1 + f0) / side^2) by less than 16 2^-93 / 2^-20
// = 2^-69 and the mixed ones by less than 4 2^-93 / 2^-20 = 2^-71, far inside the double
// rounding (2^-52 of the terms) of each enclosure.
TEST(LiquidModes, PhiEnclosesValuesAndDerivatives) {
    std::mt19937_64 rng(14);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 10);
    long checked = 0;
    for (Objective o : {Objective::Growth, Objective::Damping})
        for (double eps : {0.0, 0.05, 0.07, 0.12, 0.3})
            for (int face = 0; face < 3; ++face)
                for (int i = 0; i < 60; ++i) {
                    const double wa = std::ldexp(1.0, -scale(rng)),
                                 wb = std::ldexp(1.0, -scale(rng));
                    const double a0 = (1.0 - wa) * unit(rng), b0 = (1.0 - wb) * unit(rng);
                    const double a1 = a0 + wa, b1 = b0 + wb;
                    const double am = 0.5 * (a0 + a1), bm = 0.5 * (b0 + b1);
                    D2 out[7];
                    liquid_modes_detail::phi(face, Iv(a0, a1), Iv(b0, b1), eps, o, out);
                    const double as[3] = {a0, am, a1}, bs[3] = {b0, bm, b1};
                    Q f[3][3][7];
                    for (int ia = 0; ia < 3; ++ia)
                        for (int ib = 0; ib < 3; ++ib) {
                            double v[3];
                            test::face_vector(face, as[ia], bs[ib], v);
                            test::phis(o, eps, v, f[ia][ib]);
                        }
                    const Q da = Q(a1) - Q(a0), db = Q(b1) - Q(b0);
                    const Q dal = Q(am) - Q(a0), dar = Q(a1) - Q(am);
                    const Q dbl = Q(bm) - Q(b0), dbr = Q(b1) - Q(bm);
                    for (int k = 0; k < 7; ++k) {
                        const std::string where = "face " + std::to_string(face) + " eps " +
                                                  std::to_string(eps) + " k " + std::to_string(k) +
                                                  " a [" + qstr(a0) + ", " + qstr(a1) + "] b [" +
                                                  qstr(b0) + ", " + qstr(b1) + "]";
                        for (int ia = 0; ia < 3; ++ia)
                            for (int ib = 0; ib < 3; ++ib)
                                ASSERT_TRUE(contains(out[k].v, f[ia][ib][k])) << where;
                        for (int j = 0; j < 3; ++j) {
                            ASSERT_TRUE(contains(out[k].da, (f[2][j][k] - f[0][j][k]) / da))
                                << where << " d/da " << str(out[k].da);
                            ASSERT_TRUE(contains(out[k].db, (f[j][2][k] - f[j][0][k]) / db))
                                << where << " d/db " << str(out[k].db);
                            const Q faa = 2 *
                                          ((f[2][j][k] - f[1][j][k]) / dar -
                                           (f[1][j][k] - f[0][j][k]) / dal) /
                                          da;
                            const Q fbb = 2 *
                                          ((f[j][2][k] - f[j][1][k]) / dbr -
                                           (f[j][1][k] - f[j][0][k]) / dbl) /
                                          db;
                            ASSERT_TRUE(contains(out[k].aa, faa))
                                << where << " aa " << str(out[k].aa);
                            ASSERT_TRUE(contains(out[k].bb, fbb))
                                << where << " bb " << str(out[k].bb);
                        }
                        const Q fab =
                            (f[2][2][k] - f[2][0][k] - f[0][2][k] + f[0][0][k]) / (da * db);
                        ASSERT_TRUE(contains(out[k].ab, fab)) << where << " ab " << str(out[k].ab);
                        ++checked;
                    }
                }
    EXPECT_EQ(checked, 2 * 5 * 3 * 60 * 7);
}

namespace {

/// Physics the bound tests range over: the shipped default and dendrite physics, physics
/// that beta does not regularise, no Willmore term, the isotropic energy, unequal
/// spacings and W0 != 1.
const test::Physics kPhysics[] = {
    {0.07, 0.0149, 1.0, {0.4, 0.4, 0.4}}, {0.12, 0.25, 1.0, {0.4, 0.4, 0.4}},
    {0.3, 0.1, 1.0, {0.4, 0.4, 0.4}},     {0.05, 0.0, 1.0, {0.4, 0.4, 0.4}},
    {0.12, 0.01, 1.0, {0.5, 0.4, 0.3}},   {0.0, 0.0, 1.0, {0.3, 0.4, 0.5}},
    {0.07, 0.0149, 1.7, {0.4, 0.5, 0.6}},
};

/// A random box of the search's domain: (a, b) on a face, theta in [0, 3.2] x
/// [-3.2, 3.2]^2; each side a point one time in eight, else 2^-k of the domain,
/// k in [0, 20].
struct Box {
    int face;
    double a0, a1, b0, b1, lo[3], hi[3];
};
Box random_box(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 20), point(0, 7), face(0, 2);
    auto side = [&](double full) {
        return point(rng) == 0 ? 0.0 : full * std::ldexp(1.0, -scale(rng));
    };
    Box B{};
    B.face = face(rng);
    const double wa = side(1.0), wb = side(1.0);
    B.a0 = (1.0 - wa) * unit(rng);
    B.a1 = B.a0 + wa;
    B.b0 = (1.0 - wb) * unit(rng);
    B.b1 = B.b0 + wb;
    for (int d = 0; d < 3; ++d) {
        const double from = d == 0 ? 0.0 : -3.2, w = side(3.2 - from);
        B.lo[d] = from + (3.2 - from - w) * unit(rng);
        B.hi[d] = std::min(B.lo[d] + w, 3.2);
    }
    return B;
}

/// The box's 32 corners, its centre and eight random points.
std::vector<std::pair<std::array<double, 2>, std::array<double, 3>>>
box_points(const Box& B, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<std::pair<std::array<double, 2>, std::array<double, 3>>> p;
    for (int corner = 0; corner < 32; ++corner)
        p.push_back({{corner & 1 ? B.a1 : B.a0, corner & 2 ? B.b1 : B.b0},
                     {corner & 4 ? B.hi[0] : B.lo[0], corner & 8 ? B.hi[1] : B.lo[1],
                      corner & 16 ? B.hi[2] : B.lo[2]}});
    auto inside = [&](double lo, double hi, double t) {
        return std::clamp(lo + (hi - lo) * t, lo, hi);
    };
    p.push_back({{inside(B.a0, B.a1, 0.5), inside(B.b0, B.b1, 0.5)},
                 {inside(B.lo[0], B.hi[0], 0.5), inside(B.lo[1], B.hi[1], 0.5),
                  inside(B.lo[2], B.hi[2], 0.5)}});
    for (int i = 0; i < 8; ++i)
        p.push_back({{inside(B.a0, B.a1, unit(rng)), inside(B.b0, B.b1, unit(rng))},
                     {inside(B.lo[0], B.hi[0], unit(rng)), inside(B.lo[1], B.hi[1], unit(rng)),
                      inside(B.lo[2], B.hi[2], unit(rng))}});
    return p;
}

std::string describe(const test::Physics& p, Objective o, const Box& B) {
    return std::string(o == Objective::Growth ? "growth" : "damping") + " eps " +
           std::to_string(p.eps) + " beta " + std::to_string(p.beta) + " W0 " +
           std::to_string(p.W0) + " face " + std::to_string(B.face) + " a [" + qstr(B.a0) + ", " +
           qstr(B.a1) + "] b [" + qstr(B.b0) + ", " + qstr(B.b1) + "] theta [" + qstr(B.lo[0]) +
           ", " + qstr(B.hi[0]) + "] x [" + qstr(B.lo[1]) + ", " + qstr(B.hi[1]) + "] x [" +
           qstr(B.lo[2]) + ", " + qstr(B.hi[2]) + "]";
}

} // namespace

// Each of the three bounds the search takes the smallest of is an upper bound of F by
// itself: never below the binary128 value of F at any of 41 points of the box. Boxes of
// every size from the whole domain down to points, in n and in theta separately. A sound
// bound ends with an upward step on terms of order 1 (2^-53 relative), so it is far above
// the binary128 error of the reference (2^-100 relative).
TEST(LiquidModes, EveryBoxBoundIsAnUpperBound) {
    std::mt19937_64 rng(15);
    long checked = 0;
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping})
            for (int i = 0; i < 160; ++i) {
                const Box B = random_box(rng);
                const auto bounds = liquid_modes_detail::box_bounds(
                    o, p.eps, p.beta, p.W0, p.h, B.face, B.a0, B.a1, B.b0, B.b1, B.lo, B.hi);
                Q highest = -Q(kMax);
                for (const auto& [ab, theta] : box_points(B, rng)) {
                    double v[3];
                    test::face_vector(B.face, ab[0], ab[1], v);
                    highest = std::max(highest, test::objective(o, p, v, theta.data()).v);
                }
                ASSERT_TRUE(Q(bounds.joint) >= highest)
                    << describe(p, o, B) << ": joint " << bounds.joint << " < " << qstr(highest);
                ASSERT_TRUE(Q(bounds.second_order_n) >= highest)
                    << describe(p, o, B) << ": second order in n " << bounds.second_order_n << " < "
                    << qstr(highest);
                ASSERT_TRUE(Q(bounds.second_order_theta) >= highest)
                    << describe(p, o, B) << ": second order in theta " << bounds.second_order_theta
                    << " < " << qstr(highest);
                ++checked;
            }
    EXPECT_EQ(checked, 7 * 2 * 160);
}

// The search drops a box when its interval theta gradient excludes zero in a component.
//  * Every theta in {0, pi}^3 is a stationary point for every n (c_d' = 2 sin theta_d /
//    h^2 and every q_e vanish there): a box holding one is never dropped, whatever its
//    n-box and size.
//  * A dropped box has a component whose derivative keeps one strict sign on it: at its
//    41 points the binary128 derivatives (forward differentiation of the definitions) of
//    that component never take both signs. A sample counts only where its magnitude
//    exceeds 2^-80, above the reference's error (fewer than 2^9 operations on terms below
//    2^17, 2^-113 each).
TEST(LiquidModes, DroppedBoxesHoldNoStationaryPoint) {
    std::mt19937_64 rng(16);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 30), corner(0, 7);
    long kept = 0, dropped = 0;
    const Q certain = ldexpq(1, -80);
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping}) {
            for (int i = 0; i < 300; ++i) {
                Box B = random_box(rng);
                const int c = corner(rng);
                for (int d = 0; d < 3; ++d) { // around a point of {0, pi}^3
                    const Q at = (c >> d) & 1 ? kPiQ : Q(0);
                    const double w = 3.2 * std::ldexp(1.0, -scale(rng));
                    B.lo[d] =
                        std::max(d == 0 ? 0.0 : -3.2, static_cast<double>(at) - w * unit(rng));
                    B.hi[d] = std::min(3.2, std::nextafter(static_cast<double>(at), kInf) +
                                                w * unit(rng));
                    ASSERT_TRUE(Q(B.lo[d]) <= at && at <= Q(B.hi[d]));
                }
                const auto bounds = liquid_modes_detail::box_bounds(
                    o, p.eps, p.beta, p.W0, p.h, B.face, B.a0, B.a1, B.b0, B.b1, B.lo, B.hi);
                ASSERT_FALSE(bounds.no_stationary_point) << describe(p, o, B);
                ++kept;
            }
            for (int i = 0; i < 300; ++i) {
                const Box B = random_box(rng);
                const auto bounds = liquid_modes_detail::box_bounds(
                    o, p.eps, p.beta, p.W0, p.h, B.face, B.a0, B.a1, B.b0, B.b1, B.lo, B.hi);
                if (!bounds.no_stationary_point)
                    continue;
                ++dropped;
                bool one_signed = false;
                const auto points = box_points(B, rng);
                for (int d = 0; d < 3 && !one_signed; ++d) {
                    bool positive = false, negative = false;
                    for (const auto& [ab, theta] : points) {
                        double v[3];
                        test::face_vector(B.face, ab[0], ab[1], v);
                        const Q g = test::objective(o, p, v, theta.data(), d).d;
                        positive = positive || g > certain;
                        negative = negative || g < -certain;
                    }
                    one_signed = !(positive && negative);
                }
                ASSERT_TRUE(one_signed) << describe(p, o, B);
            }
        }
    EXPECT_EQ(kept, 7 * 2 * 300);
    EXPECT_GT(dropped, 0);
}

// The three bounds also hold where they are tightest: on boxes around a stationary point
// theta in {0, pi} x {0, +-pi}^2 (sides 3.2 2^-k, k up to 30, any n-box), where the theta
// gradient vanishes and no first-order term hides a missing second-order one.
TEST(LiquidModes, BoundsHoldAtStationaryCheckerboards) {
    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 30), at(0, 2);
    long checked = 0;
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping})
            for (int i = 0; i < 100; ++i) {
                Box B = random_box(rng);
                for (int d = 0; d < 3; ++d) {
                    const int which = d == 0 ? at(rng) % 2 : at(rng); // 0, pi, -pi
                    const double c = which == 0 ? 0.0 : which == 1 ? interval::kPi : -interval::kPi;
                    const double w = 3.2 * std::ldexp(1.0, -scale(rng));
                    // [lo, hi] holds 0, the real pi (above kPi) or -pi (below -kPi)
                    B.lo[d] = std::max(d == 0 ? 0.0 : -3.2,
                                       (which == 2 ? std::nextafter(c, -kInf) : c) - w * unit(rng));
                    B.hi[d] =
                        std::min(3.2, (which == 1 ? std::nextafter(c, kInf) : c) + w * unit(rng));
                }
                const auto bounds = liquid_modes_detail::box_bounds(
                    o, p.eps, p.beta, p.W0, p.h, B.face, B.a0, B.a1, B.b0, B.b1, B.lo, B.hi);
                Q highest = -Q(kMax);
                for (const auto& [ab, theta] : box_points(B, rng)) {
                    double v[3];
                    test::face_vector(B.face, ab[0], ab[1], v);
                    highest = std::max(highest, test::objective(o, p, v, theta.data()).v);
                }
                ASSERT_TRUE(Q(bounds.joint) >= highest)
                    << describe(p, o, B) << ": joint " << bounds.joint << " < " << qstr(highest);
                ASSERT_TRUE(Q(bounds.second_order_n) >= highest)
                    << describe(p, o, B) << ": second order in n " << bounds.second_order_n << " < "
                    << qstr(highest);
                ASSERT_TRUE(Q(bounds.second_order_theta) >= highest)
                    << describe(p, o, B) << ": second order in theta " << bounds.second_order_theta
                    << " < " << qstr(highest);
                ++checked;
            }
    EXPECT_EQ(checked, 7 * 2 * 100);
}

// The theta bounds hold for every phi in the given intervals, not only for the phi an
// n-box realises (decomposed() maximises each axis's parabola with the extreme alpha and
// b of the intervals): at the 128 corners of the phi box and nine points of the theta box
// (its corners and centre) the binary128 value of sum_k phi_k psi_k(theta) is below the
// per-axis decomposition, the interval value and the Taylor bound. Interval centres in
// [-5, 5] (phi_6 in [0.1, 5], its sign in the search), widths 2^-k, k in [0, 30], or 0.
TEST(LiquidModes, ThetaBoundsHoldForEveryCoefficientInTheirRanges) {
    std::mt19937_64 rng(20);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> scale(0, 30), point(0, 7);
    long checked = 0;
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping})
            for (int i = 0; i < 120; ++i) {
                Iv phi[7];
                for (int k = 0; k < 7; ++k) {
                    const double c = k < 6 ? -5.0 + 10.0 * unit(rng) : 0.1 + 4.9 * unit(rng);
                    const double w = point(rng) == 0 ? 0.0 : std::ldexp(1.0, -scale(rng));
                    phi[k] = {c - w, c + w};
                }
                const Box B = random_box(rng);
                const auto bounds =
                    liquid_modes_detail::theta_bounds(o, p.eps, p.beta, p.W0, p.h, phi, B.lo, B.hi);
                Q highest = -Q(kMax);
                for (int t = 0; t < 9; ++t) {
                    double th[3];
                    for (int d = 0; d < 3; ++d)
                        th[d] = t < 8 ? ((t >> d) & 1 ? B.hi[d] : B.lo[d])
                                      : std::clamp(0.5 * (B.lo[d] + B.hi[d]), B.lo[d], B.hi[d]);
                    Q psi[7];
                    test::psi(o, p, th, psi);
                    for (int corner = 0; corner < 128; ++corner) {
                        Q f = 0;
                        for (int k = 0; k < 7; ++k)
                            f += Q((corner >> k) & 1 ? phi[k].hi : phi[k].lo) * psi[k];
                        highest = std::max(highest, f);
                    }
                }
                ASSERT_TRUE(Q(bounds.decomposed) >= highest)
                    << describe(p, o, B) << ": decomposed " << bounds.decomposed << " < "
                    << qstr(highest);
                ASSERT_TRUE(Q(bounds.interval_value) >= highest)
                    << describe(p, o, B) << ": interval value " << bounds.interval_value << " < "
                    << qstr(highest);
                ASSERT_TRUE(Q(bounds.taylor) >= highest)
                    << describe(p, o, B) << ": Taylor " << bounds.taylor << " < " << qstr(highest);
                ++checked;
            }
    EXPECT_EQ(checked, 7 * 2 * 120);
}

// The domain the search covers suffices: theta_x from 0 to beyond pi, theta_y and theta_z
// over a whole period (F depends on theta only through sin theta_d and cos theta_d, so it
// is 2 pi periodic in each). The symmetries the reduction to that domain and to the octant
// (or the triangle n_x >= n_y >= n_z) relies on hold in binary128 at 2000 random modes per
// physics: F is even in theta, unchanged by a sign change of n_d together with theta_d,
// and, for equal spacings, by a permutation of the axes of n and theta together. Both
// sides take fewer than 2^9 operations on terms below 2^17 (2^-113 each), so they agree to
// 2^-86.
TEST(LiquidModes, SearchDomainAndSymmetries) {
    using liquid_modes_detail::kThetaHi;
    using liquid_modes_detail::kThetaLo;
    EXPECT_TRUE(Q(kThetaLo[0]) <= 0 && Q(kThetaHi[0]) >= kPiQ);
    for (int d = 1; d < 3; ++d)
        EXPECT_TRUE(Q(kThetaLo[d]) <= -kPiQ && Q(kThetaHi[d]) >= kPiQ) << d;
    std::mt19937_64 rng(21);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> angle(-3.2, 3.2);
    const Q agree = ldexpq(1, -86);
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping})
            for (int i = 0; i < 2000; ++i) {
                const double n[3] = {normal(rng), normal(rng), normal(rng)};
                const double th[3] = {angle(rng), angle(rng), angle(rng)};
                const Q f = test::objective(o, p, n, th).v;
                const double neg[3] = {-th[0], -th[1], -th[2]};
                ASSERT_TRUE(fabsq(test::objective(o, p, n, neg).v - f) <= agree);
                const int d = i % 3;
                double flipped_n[3] = {n[0], n[1], n[2]}, flipped_th[3] = {th[0], th[1], th[2]};
                flipped_n[d] = -n[d];
                flipped_th[d] = -th[d];
                ASSERT_TRUE(fabsq(test::objective(o, p, flipped_n, flipped_th).v - f) <= agree)
                    << "sign change of axis " << d;
                if (p.h[0] == p.h[1] && p.h[1] == p.h[2]) {
                    const double pn[3] = {n[1], n[2], n[0]}, pth[3] = {th[1], th[2], th[0]};
                    ASSERT_TRUE(fabsq(test::objective(o, p, pn, pth).v - f) <= agree)
                        << "axis permutation";
                }
            }
}

// The enclosure the search records at a mode holds the binary128 value of F there, for
// directions in every octant (negative components too) and wave numbers anywhere in
// [-3.2, 3.2]^3.
TEST(LiquidModes, ModeEnclosureHoldsTheValue) {
    std::mt19937_64 rng(17);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> angle(-3.2, 3.2);
    for (const test::Physics& p : kPhysics)
        for (Objective o : {Objective::Growth, Objective::Damping})
            for (int i = 0; i < 2000; ++i) {
                const double n[3] = {normal(rng), normal(rng), normal(rng)};
                const double th[3] = {angle(rng), angle(rng), angle(rng)};
                const Iv r =
                    liquid_modes_detail::mode_enclosure(o, p.eps, p.beta, p.W0, p.h, n, th);
                const Q f = test::objective(o, p, n, th).v;
                ASSERT_TRUE(contains(r, f)) << qstr(f) << " not in " << str(r);
            }
}

// ── The search ──────────────────────────────────────────────────────────────

// Parameter types of the value-parameterised tests: outside the anonymous namespace, since
// gtest's test factories (header templates) hold them.
namespace liquid_mode_cases {

/// The growth objective's maxima found by an independent search (numpy: a 2-degree
/// direction grid times a 40^3 wave-number grid, then Nelder-Mead in all five variables;
/// exp/stab2/landscape.py), converged to 1e-9, h = 0.4.
struct Reference {
    double eps, beta, maximum;
};
constexpr Reference kReferences[] = {{0.07, 0.0149, 41.189758899},
                                     {0.05, 0.0, 106.471074380},
                                     {0.12, 0.25, 3.883798363},
                                     {0.12, 0.01, 73.296905932}};
/// gtest prints a parameter with PrintTo; ctest names the case after that text.
void PrintTo(const Reference& c, std::ostream* os) {
    *os << "eps" << c.eps << "_beta" << c.beta;
}

} // namespace liquid_mode_cases

using liquid_mode_cases::kReferences;
using liquid_mode_cases::Reference;

namespace {

/// The cutoff the search stops at: attained + kLiquidModeTolerance max(1, |attained|),
/// rounded up.
double stopping_bound(double attained) {
    return interval::up(attained + kLiquidModeTolerance * std::max(1.0, std::fabs(attained)));
}

} // namespace

// liquid_gradient_growth: the exact maximum lies in [attained, bound]. attained is below
// the binary128 value at the reported mode; bound is the search's stopping value of
// attained, and above the binary128 value of 20000 random modes; attained agrees with the
// independent search to its convergence. One test per physics (each has its own timeout).
class GrowthBound : public ::testing::TestWithParam<Reference> {};

TEST_P(GrowthBound, IsCertified) {
    const Reference c = GetParam();
    std::mt19937_64 rng(18);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> angle(-3.2, 3.2);
    const test::Physics p{c.eps, c.beta, 1.0, {0.4, 0.4, 0.4}};
    const LiquidModeBound g = liquid_gradient_growth(c.eps, c.beta, 1.0, p.h);
    const Q at_mode = test::objective(Objective::Growth, p, g.mode.n, g.mode.theta).v;
    EXPECT_TRUE(Q(g.attained) <= at_mode) << g.attained << " > " << qstr(at_mode);
    EXPECT_EQ(g.bound, stopping_bound(g.attained));
    EXPECT_NEAR(g.attained / c.maximum, 1.0, 1e-9);
    Q sampled = -Q(kMax);
    for (int i = 0; i < 20000; ++i) {
        const double n[3] = {normal(rng), normal(rng), normal(rng)};
        const double th[3] = {angle(rng), angle(rng), angle(rng)};
        sampled = std::max(sampled, test::objective(Objective::Growth, p, n, th).v);
    }
    EXPECT_TRUE(sampled <= Q(g.bound)) << qstr(sampled) << " > " << g.bound;
}

INSTANTIATE_TEST_SUITE_P(LiquidModes, GrowthBound, ::testing::ValuesIn(kReferences),
                         [](const ::testing::TestParamInfo<Reference>& case_info) {
                             return "case" + std::to_string(case_info.index);
                         });

// liquid_gradient_damping: where beta does not regularise eps the witness's mode has
// Lam + beta L <= attained = bound < 0 in binary128; elsewhere bound = 0 and the reported
// mode has Lam + beta L <= attained (also at eps 0.12, beta 0.07, whose minimum 0.92 lies
// between 0 and 1, and at beta 0.0529199..., just below the smallest beta that
// regularises eps 0.12 at h 0.4, whose minimum is -0.0006).
TEST(LiquidModes, DampingWitnessIsCertified) {
    for (const auto& [eps, beta, negative] :
         {std::tuple{0.3, 0.1, true}, std::tuple{0.12, 0.01, true}, std::tuple{0.07, 0.0149, false},
          std::tuple{0.12, 0.25, false}, std::tuple{0.15, 0.401, false},
          std::tuple{0.05, 0.0, false}, std::tuple{0.12, 0.07, false},
          std::tuple{0.12, 0.052919921874999998, true}}) {
        SCOPED_TRACE("eps " + std::to_string(eps) + " beta " + std::to_string(beta));
        const test::Physics p{eps, beta, 1.0, {0.4, 0.4, 0.4}};
        const LiquidModeBound d = liquid_gradient_damping(eps, beta, 1.0, p.h);
        const Q at_mode = test::damping(p, d.mode.n, d.mode.theta);
        EXPECT_TRUE(at_mode <= Q(d.attained)) << qstr(at_mode) << " > " << d.attained;
        if (negative) {
            EXPECT_LT(d.bound, 0.0);
            EXPECT_EQ(d.bound, d.attained);
        } else {
            EXPECT_EQ(d.bound, 0.0);
            EXPECT_FALSE(std::signbit(d.bound));
        }
    }
}

namespace liquid_mode_cases {

struct SearchCase {
    double eps, beta, h[3];
};
constexpr SearchCase kSearchCases[] = {{0.07, 0.0149, {0.4, 0.4, 0.4}},
                                       {0.12, 0.25, {0.4, 0.4, 0.4}},
                                       {0.05, 0.0, {0.4, 0.4, 0.4}},
                                       {0.07, 0.0149, {0.4, 0.5, 0.6}},
                                       {0.3, 0.01, {0.4, 0.4, 0.4}}};
void PrintTo(const SearchCase& c, std::ostream* os) {
    *os << "eps" << c.eps << "_beta" << c.beta << "_h" << c.h[0] << "_" << c.h[1] << "_" << c.h[2];
}

} // namespace liquid_mode_cases

using liquid_mode_cases::kSearchCases;
using liquid_mode_cases::SearchCase;

namespace {

const LiquidMode kFar{{1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}; // the uniform shift along x

} // namespace

// The branch and bound finds and certifies the maximum without the seed search's head
// start: from a mode far from it (the uniform shift along x) and from one just below it
// (the seeded mode moved 0.05 rad in theta_x, at least 1e-6 relative below), its
// [attained, bound] still holds the exact maximum, so it meets the seeded run's. The last
// case's maximum lies inside the octant and the theta box (n = (0.71, 0.50, 0.50),
// theta = (2.03, -1.83, -1.83)), so every part of the domain the search starts from
// matters. One test per physics (each has its own timeout).
class SearchWithoutSeed : public ::testing::TestWithParam<SearchCase> {};

TEST_P(SearchWithoutSeed, CertifiesTheMaximum) {
    const SearchCase c = GetParam();
    const test::Physics p{c.eps, c.beta, 1.0, {c.h[0], c.h[1], c.h[2]}};
    const LiquidModeBound seeded = liquid_gradient_growth(c.eps, c.beta, 1.0, p.h);
    LiquidMode near = seeded.mode;
    near.theta[0] += 0.05;
    ASSERT_TRUE(test::objective(Objective::Growth, p, near.n, near.theta).v <
                Q(seeded.attained) * (1 - Q(1e-6)));
    for (const LiquidMode& start : {kFar, near}) {
        const LiquidModeBound r =
            liquid_modes_detail::search_from(Objective::Growth, c.eps, c.beta, 1.0, p.h, start);
        EXPECT_LE(r.attained, seeded.bound);
        EXPECT_LE(seeded.attained, r.bound);
        EXPECT_EQ(r.bound, stopping_bound(r.attained));
        const Q at_mode = test::objective(Objective::Growth, p, r.mode.n, r.mode.theta).v;
        EXPECT_TRUE(Q(r.attained) <= at_mode) << r.attained << " > " << qstr(at_mode);
    }
}

INSTANTIATE_TEST_SUITE_P(LiquidModes, SearchWithoutSeed, ::testing::ValuesIn(kSearchCases),
                         [](const ::testing::TestParamInfo<SearchCase>& case_info) {
                             return "case" + std::to_string(case_info.index);
                         });

// Without the seed the damping search still finds a certified witness, or proves there
// is none, also where the minimum is -0.0006 or 0.92.
TEST(LiquidModes, DampingSearchWithoutSeed) {
    for (const auto& [eps, beta, negative] :
         {std::tuple{0.3, 0.1, true}, std::tuple{0.12, 0.01, true}, std::tuple{0.07, 0.0149, false},
          std::tuple{0.12, 0.25, false}, std::tuple{0.12, 0.07, false},
          std::tuple{0.12, 0.052919921874999998, true}}) {
        SCOPED_TRACE("eps " + std::to_string(eps) + " beta " + std::to_string(beta));
        const test::Physics p{eps, beta, 1.0, {0.4, 0.4, 0.4}};
        ASSERT_TRUE(test::damping(p, kFar.n, kFar.theta) > 0); // the start is no witness
        const LiquidModeBound r =
            liquid_modes_detail::search_from(Objective::Damping, eps, beta, 1.0, p.h, kFar);
        const Q at_mode = test::damping(p, r.mode.n, r.mode.theta);
        EXPECT_TRUE(at_mode <= Q(r.attained)) << qstr(at_mode) << " > " << r.attained;
        if (negative)
            EXPECT_LT(r.bound, 0.0);
        else
            EXPECT_EQ(r.bound, 0.0);
    }
}

namespace liquid_mode_cases {

struct PermutationCase {
    double eps, beta, h[3]; // h sorted ascending
};
constexpr PermutationCase kPermutationCases[] = {
    {0.05, 0.0, {0.3, 0.4, 0.5}}, {0.07, 0.0149, {0.3, 0.4, 0.5}}, {0.05, 0.0, {0.4, 0.5, 0.5}}};
void PrintTo(const PermutationCase& c, std::ostream* os) {
    *os << "eps" << c.eps << "_beta" << c.beta << "_h" << c.h[0] << "_" << c.h[1] << "_" << c.h[2];
}

} // namespace liquid_mode_cases

using liquid_mode_cases::kPermutationCases;
using liquid_mode_cases::PermutationCase;

// Unless all three spacings are equal the search covers all three faces of the cube: the
// orderings of the same spacings (six, or three when two are equal) describe the same
// liquid with its axes permuted, so their [attained, bound] all hold the same maximum and
// meet pairwise. One test per case (each has its own timeout).
class SpacingPermutations : public ::testing::TestWithParam<PermutationCase> {};

TEST_P(SpacingPermutations, Meet) {
    const PermutationCase c = GetParam();
    double h[3] = {c.h[0], c.h[1], c.h[2]};
    std::vector<LiquidModeBound> r;
    do
        r.push_back(liquid_gradient_growth(c.eps, c.beta, 1.0, h));
    while (std::next_permutation(h, h + 3));
    ASSERT_EQ(r.size(), c.h[1] == c.h[2] ? 3u : 6u);
    for (const LiquidModeBound& x : r)
        for (const LiquidModeBound& y : r)
            EXPECT_LE(x.attained, y.bound);
}

INSTANTIATE_TEST_SUITE_P(LiquidModes, SpacingPermutations, ::testing::ValuesIn(kPermutationCases),
                         [](const ::testing::TestParamInfo<PermutationCase>& case_info) {
                             return "case" + std::to_string(case_info.index);
                         });

// A search that spends its budget reports the range its extremum lies in. Damping close to
// the smallest beta that regularises eps 0.12 at h 0.4 (beta 0.0529345703125: the minimum
// of Lam + beta L lies between -0.0006 and 0.0015 by the bisection in
// exp/round24/count/threshold.cpp), with 1e5 box evaluations: no binary128 Lam + beta L of
// 20000 random modes is below the lower end, and the reported mode's is at most the upper
// end. Growth at the default physics with budgets from 1000 to 1e5 evaluations (factors
// of 1.25, so the budget runs out at many stages of the search): each range holds the
// seeded search's [attained, bound], and the reported mode's binary128 value is at least
// the lower end.
TEST(LiquidModes, SpentBudgetReportsTheRange) {
    const double h[3] = {0.4, 0.4, 0.4};
    std::mt19937_64 rng(22);
    std::normal_distribution<double> normal;
    std::uniform_real_distribution<double> angle(-3.2, 3.2);
    const test::Physics close{0.12, 0.0529345703125, 1.0, {0.4, 0.4, 0.4}};
    try {
        (void)liquid_gradient_damping(close.eps, close.beta, 1.0, h, 100000);
        ADD_FAILURE() << "decided within 1e5 evaluations";
    } catch (const LiquidModeSearchExhausted& e) {
        EXPECT_LT(e.lower, 0.0);
        EXPECT_GE(e.upper, 0.0);
        const Q at_mode = test::damping(close, e.mode.n, e.mode.theta);
        EXPECT_TRUE(at_mode <= Q(e.upper)) << qstr(at_mode) << " > " << e.upper;
        Q lowest = Q(kMax);
        for (int i = 0; i < 20000; ++i) {
            const double n[3] = {normal(rng), normal(rng), normal(rng)};
            const double th[3] = {angle(rng), angle(rng), angle(rng)};
            lowest = std::min(lowest, test::damping(close, n, th));
        }
        EXPECT_TRUE(lowest >= Q(e.lower)) << qstr(lowest) << " < " << e.lower;
        EXPECT_NE(std::string(e.what()).find("spent its budget"), std::string::npos) << e.what();
    }
    const test::Physics shipped{0.07, 0.0149, 1.0, {0.4, 0.4, 0.4}};
    const LiquidModeBound seeded = liquid_gradient_growth(shipped.eps, shipped.beta, 1.0, h);
    int exhausted = 0;
    for (long budget = 1000; budget <= 100000; budget = budget * 5 / 4) {
        SCOPED_TRACE("budget " + std::to_string(budget));
        try {
            const LiquidModeBound g =
                liquid_gradient_growth(shipped.eps, shipped.beta, 1.0, h, budget);
            EXPECT_LE(g.attained, seeded.bound);
            EXPECT_LE(seeded.attained, g.bound);
        } catch (const LiquidModeSearchExhausted& e) {
            ++exhausted;
            EXPECT_LE(e.lower, seeded.bound);
            EXPECT_LE(seeded.attained, e.upper);
            const Q at_mode = test::objective(Objective::Growth, shipped, e.mode.n, e.mode.theta).v;
            EXPECT_TRUE(at_mode >= Q(e.lower)) << qstr(at_mode) << " < " << e.lower;
        }
    }
    EXPECT_GT(exhausted, 0);
}

// search_from and mode_enclosure need a mode they can evaluate: a direction whose squared
// length is enclosed away from 0 and infinity (2e-162 squared is the smallest subnormal,
// whose enclosure reaches 0; 1e-170 squared underflows to 0) and finite wave numbers.
TEST(LiquidModes, UnusableModesAreRejected) {
    const double h[3] = {0.4, 0.4, 0.4};
    const LiquidMode zero{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const LiquidMode tiny{{2e-162, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const LiquidMode underflow{{1e-170, 1e-170, 1e-170}, {0.0, 0.0, 0.0}};
    const LiquidMode huge{{1e200, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const LiquidMode nan_theta{{1.0, 0.0, 0.0}, {std::nan(""), 0.0, 0.0}};
    for (const LiquidMode& m : {zero, tiny, underflow, huge, nan_theta}) {
        EXPECT_THROW(
            (void)liquid_modes_detail::search_from(Objective::Growth, 0.07, 0.0149, 1.0, h, m),
            std::invalid_argument);
        EXPECT_THROW((void)liquid_modes_detail::mode_enclosure(Objective::Growth, 0.07, 0.0149, 1.0,
                                                               h, m.n, m.theta),
                     std::invalid_argument);
    }
}
