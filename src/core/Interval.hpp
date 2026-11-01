#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

// Interval arithmetic rounded outward, and second-order forward differentiation in two
// variables over intervals: the arithmetic of the certified liquid-mode bounds
// (LiquidModes.cpp). Every operation returns an interval that contains the exact real
// result of the operation for every argument in its input intervals
// (tests/unit/test_LiquidModes.cpp checks this against binary128 references).

namespace ac::interval {

/// One step below / above a round-to-nearest result: the exact value lies within half a
/// unit in the last place of it, and |x| 2^-52 + the smallest subnormal is at least one
/// unit, so the step passes the exact value. An infinite result on the wrong side is an
/// overflow of finite operands (no lower endpoint is +inf, no upper one -inf), whose exact
/// value lies beyond the largest double: down(+inf) is that double, up(-inf) its negative.
/// NaN, which no operation below produces from non-NaN endpoints, becomes -inf / +inf.
inline double down(double x) {
    if (std::isfinite(x)) [[likely]]
        return x - (std::fabs(x) * 0x1p-52 + std::numeric_limits<double>::denorm_min());
    if (std::isnan(x))
        return -std::numeric_limits<double>::infinity();
    return x > 0.0 ? std::numeric_limits<double>::max() : x;
}
inline double up(double x) {
    if (std::isfinite(x)) [[likely]]
        return x + (std::fabs(x) * 0x1p-52 + std::numeric_limits<double>::denorm_min());
    if (std::isnan(x))
        return std::numeric_limits<double>::infinity();
    return x < 0.0 ? -std::numeric_limits<double>::max() : x;
}

/// x y with 0 times an infinite endpoint taken as 0: an infinite endpoint stands for an
/// unbounded or overflowed finite value, whose product with an exact 0 is 0. Endpoints are
/// never NaN (down and up map NaN to infinities), so a NaN product is such a 0 * inf.
inline double times(double x, double y) {
    const double p = x * y;
    return std::isnan(p) ? 0.0 : p;
}
/// Upper bounds of a + b and a b, each rounded once (a chain of operations needs one
/// upward step per operation: two roundings can together exceed the one step).
inline double add_up(double a, double b) {
    return up(a + b);
}
inline double mul_up(double a, double b) {
    return up(times(a, b));
}

/// The closed interval [lo, hi].
struct Iv {
    double lo = 0.0, hi = 0.0;
    Iv() = default;
    explicit Iv(double v) : lo(v), hi(v) {}
    Iv(double l, double h) : lo(l), hi(h) {}
};
inline Iv operator+(Iv a, Iv b) {
    return {down(a.lo + b.lo), up(a.hi + b.hi)};
}
inline Iv operator-(Iv a, Iv b) {
    return {down(a.lo - b.hi), up(a.hi - b.lo)};
}
inline Iv operator-(Iv a) {
    return {-a.hi, -a.lo};
}
inline Iv operator*(Iv a, Iv b) {
    const double p0 = a.lo * b.lo, p1 = a.lo * b.hi, p2 = a.hi * b.lo, p3 = a.hi * b.hi;
    // A NaN product (0 times an infinite endpoint) makes the sum NaN, so where the sum is
    // not NaN no product is and min / max compare them all; where it is (also when the
    // partial sums meet infinities of both signs) the products are formed again with
    // 0 * inf taken as 0.
    if (!std::isnan(p0 + p1 + p2 + p3)) [[likely]]
        return {down(std::min(std::min(p0, p1), std::min(p2, p3))),
                up(std::max(std::max(p0, p1), std::max(p2, p3)))};
    const double q0 = times(a.lo, b.lo), q1 = times(a.lo, b.hi), q2 = times(a.hi, b.lo),
                 q3 = times(a.hi, b.hi);
    return {down(std::min(std::min(q0, q1), std::min(q2, q3))),
            up(std::max(std::max(q0, q1), std::max(q2, q3)))};
}
inline Iv operator*(double k, Iv a) {
    return Iv(k) * a;
}
/// a^2 (tighter than a * a when a holds 0: the square is never negative).
inline Iv sq(Iv a) {
    if (a.lo >= 0.0)
        return {down(a.lo * a.lo), up(a.hi * a.hi)};
    if (a.hi <= 0.0)
        return {down(a.hi * a.hi), up(a.lo * a.lo)};
    return {0.0, up(std::max(a.lo * a.lo, a.hi * a.hi))};
}
/// 1/a for 0 < a.lo.
inline Iv inv(Iv a) {
    return {down(1.0 / a.hi), up(1.0 / a.lo)};
}
/// 1/sqrt(a) for 0 < a.lo (sqrt is correctly rounded).
inline Iv inv_sqrt(Iv a) {
    return {down(1.0 / up(std::sqrt(a.hi))), up(1.0 / down(std::sqrt(a.lo)))};
}
/// max |x| over a.
inline double mag(Iv a) {
    return std::max(std::fabs(a.lo), std::fabs(a.hi));
}
/// {|x| : x in a}.
inline Iv iabs(Iv a) {
    if (a.lo >= 0.0)
        return a;
    if (a.hi <= 0.0)
        return -a;
    return {0.0, std::max(-a.lo, a.hi)};
}

/// pi rounded to nearest: the largest double below pi.
inline constexpr double kPi = 3.14159265358979323846;

/// [min(a, b), max(a, b)] widened on each side by |x| 2^-51 + the smallest subnormal, at
/// least two units in the last place of a normal x and one unit (the smallest subnormal)
/// of a zero or subnormal one, clamped to [-1, 1]: encloses the exact values of library
/// sines and cosines a, b. glibc's double sin and cos err by at most one unit: 1 in the
/// x86_64 table of maximum known errors (libm-test-ulps, glibc 2.39), about 0.55 by the
/// analysis in sysdeps/ieee754/dbl-64/s_sin.c (unchanged from 2.28 to 2.43). A zero or
/// subnormal result arises over [-3.2, 3.2] only as the sine of a zero or subnormal x,
/// returned as x itself, within half a unit of the exact value.
inline Iv trig_widen(double a, double b) {
    const double lo = std::min(a, b), hi = std::max(a, b);
    return {
        std::max(-1.0, lo - (std::fabs(lo) * 0x1p-51 + std::numeric_limits<double>::denorm_min())),
        std::min(1.0, hi + (std::fabs(hi) * 0x1p-51 + std::numeric_limits<double>::denorm_min()))};
}
/// sin over [lo, hi], -3.2 <= lo <= hi <= 3.2: monotone between the extrema at -pi/2 and
/// pi/2, which the interval takes over when it holds their abscissae. pi/2 lies in
/// [lo, hi] exactly when lo <= kPi/2 (the largest double below pi/2) and hi > kPi/2;
/// testing hi >= kPi/2 takes over the extremum also when hi = kPi/2, a wider enclosure.
inline Iv isin(double lo, double hi) {
    Iv r = trig_widen(std::sin(lo), std::sin(hi));
    if (lo <= 0.5 * kPi && hi >= 0.5 * kPi)
        r.hi = 1.0;
    if (lo <= -0.5 * kPi && hi >= -0.5 * kPi)
        r.lo = -1.0;
    return r;
}
/// cos over [lo, hi], -3.2 <= lo <= hi <= 3.2: extrema at 0 and +-pi, as for isin.
inline Iv icos(double lo, double hi) {
    Iv r = trig_widen(std::cos(lo), std::cos(hi));
    if (lo <= 0.0 && hi >= 0.0)
        r.hi = 1.0;
    if ((lo <= -kPi && hi >= -kPi) || (lo <= kPi && hi >= kPi))
        r.lo = -1.0;
    return r;
}

/// A function of (a, b) with its first and second derivatives, each enclosed over the
/// intervals of a and b the computation started from.
struct D2 {
    Iv v, da, db, aa, ab, bb;
};
/// The constant c.
inline D2 cst(double c) {
    return {Iv(c), Iv(0.0), Iv(0.0), Iv(0.0), Iv(0.0), Iv(0.0)};
}
/// The constant enclosed by c.
inline D2 cst(Iv c) {
    return {c, Iv(0.0), Iv(0.0), Iv(0.0), Iv(0.0), Iv(0.0)};
}
/// The variable a (or b) over the interval x.
inline D2 var_a(Iv x) {
    return {x, Iv(1.0), Iv(0.0), Iv(0.0), Iv(0.0), Iv(0.0)};
}
inline D2 var_b(Iv x) {
    return {x, Iv(0.0), Iv(1.0), Iv(0.0), Iv(0.0), Iv(0.0)};
}
inline D2 operator+(const D2& x, const D2& y) {
    return {x.v + y.v, x.da + y.da, x.db + y.db, x.aa + y.aa, x.ab + y.ab, x.bb + y.bb};
}
inline D2 operator-(const D2& x, const D2& y) {
    return {x.v - y.v, x.da - y.da, x.db - y.db, x.aa - y.aa, x.ab - y.ab, x.bb - y.bb};
}
inline D2 operator*(const D2& x, const D2& y) {
    const Iv two(2.0);
    return {x.v * y.v,
            x.da * y.v + x.v * y.da,
            x.db * y.v + x.v * y.db,
            x.aa * y.v + two * (x.da * y.da) + x.v * y.aa,
            x.ab * y.v + x.da * y.db + x.db * y.da + x.v * y.ab,
            x.bb * y.v + two * (x.db * y.db) + x.v * y.bb};
}
inline D2 operator*(double k, const D2& x) {
    return cst(k) * x;
}
inline D2 operator*(Iv k, const D2& x) {
    return cst(k) * x;
}
inline D2 operator+(Iv k, const D2& x) {
    return cst(k) + x;
}
inline D2 operator-(const D2& x) {
    return {-x.v, -x.da, -x.db, -x.aa, -x.ab, -x.bb};
}
/// f(x) given enclosures of f, f' and f'' over x.v.
inline D2 chain(const D2& x, Iv f, Iv f1, Iv f2) {
    return {f,
            f1 * x.da,
            f1 * x.db,
            f2 * (x.da * x.da) + f1 * x.aa,
            f2 * (x.da * x.db) + f1 * x.ab,
            f2 * (x.db * x.db) + f1 * x.bb};
}
/// 1/x for 0 < x.v.lo: f' = -1/x^2, f'' = 2/x^3.
inline D2 inv(const D2& x) {
    const Iv r = inv(x.v);
    return chain(x, r, -(r * r), Iv(2.0) * (r * r * r));
}
/// 1/sqrt(x) for 0 < x.v.lo: f' = -x^(-3/2)/2, f'' = 3 x^(-5/2)/4.
inline D2 inv_sqrt(const D2& x) {
    const Iv r = inv_sqrt(x.v), r3 = r * r * r;
    return chain(x, r, Iv(-0.5) * r3, Iv(0.75) * (r3 * r * r));
}

} // namespace ac::interval
