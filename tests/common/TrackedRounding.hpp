#pragma once

// Bounds on the rounding error of a double-precision GPU computation that the host repeats
// in long double (running error analysis; Higham, Accuracy and Stability of Numerical
// Algorithms, 2nd ed., sections 2.2 and 3.3).

#include <cmath>
#include <limits>

namespace ac::test {

/// A value the GPU computes in double precision, followed on the host: v is the result of
/// the same operations, in the same order, in long double, and e bounds |GPU value - v|.
///
/// An IEEE operation returns its exact result times 1 + delta, with |delta| <= U_d = 2^-53
/// in double and U_l = epsilon / 2 in long double (the values followed here are exact
/// zeros or far above the underflow threshold). For z = x op y, with the GPU's operands
/// within e_x and e_y of the host's x and y,
///   GPU z - host z = P (1 + delta_d) + (delta_d - delta_l) (x op y),
///   P = (GPU x op GPU y) - (x op y),
/// so with |x op y| = |v_z| / |1 + delta_l| <= |v_z| / (1 - U_l)
///   e_z = |P| (1 + U_d) + (U_d + U_l) |v_z| / (1 - U_l),
///   |P| <= e_x + e_y                           for + and -,
///   |P| <= |x| e_y + |y| e_x + e_x e_y         for *,
///   |P| <= (e_x + |x / y| e_y) / (|y| - e_y)   for /, when |y| > e_y.
/// A multiply-add the compiler contracts into one FMA rounds once: before that rounding
/// the GPU's sum is within |GPU x GPU y - host p| + e_c <= e_p + e_c of the host's
/// (p = x * y), so the bound of the two separate operations holds, contracted or not.
/// Negation is exact.
///
/// e itself is evaluated in long double, rounded to nearest. Every operation of that
/// evaluation is monotone in the operands' e and |v| (decreasing in the divisor) and
/// within a factor 1 +- U_l of its exact result, and at most 8 of them lie on any path
/// from an operand to the result (division: |q| / (1 - U_l), times e_y, plus e_x, the
/// divisor |y| - e_y, the quotient, times 1 + U_d, which rounds when long double is
/// double, and the final sum), so the evaluated e is at least (1 - U_l)^8 times the exact
/// formula. rounded_up multiplies it by 1 + 16 U_l (exact in long double), with one more
/// rounding: (1 - U_l)^9 (1 + 16 U_l) > 1, and by induction every e is an upper bound.
struct Tracked {
    long double v = 0;
    long double e = 0;
};

inline constexpr long double kUnitRoundoffDouble = std::numeric_limits<double>::epsilon() / 2.0L;
inline constexpr long double kUnitRoundoffLongDouble =
    std::numeric_limits<long double>::epsilon() / 2;

/// A nonnegative bound evaluated in long double with at most 8 roundings on any path,
/// made an upper bound on its exact value (see Tracked).
inline long double rounded_up(long double bound) {
    return bound * (1 + 16 * kUnitRoundoffLongDouble);
}

/// The result v of an operation whose operands contribute at most `propagated`.
inline Tracked rounded(long double v, long double propagated) {
    constexpr long double ud = kUnitRoundoffDouble, ul = kUnitRoundoffLongDouble;
    return {v, rounded_up(propagated * (1 + ud) + (ud + ul) * std::fabs(v) / (1 - ul))};
}

inline Tracked operator-(const Tracked& a) {
    return {-a.v, a.e};
}

inline Tracked operator+(const Tracked& a, const Tracked& b) {
    return rounded(a.v + b.v, a.e + b.e);
}

inline Tracked operator-(const Tracked& a, const Tracked& b) {
    return rounded(a.v - b.v, a.e + b.e);
}

inline Tracked operator*(const Tracked& a, const Tracked& b) {
    return rounded(a.v * b.v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e);
}

/// A divisor whose sign the bound does not certify (|y| <= e_y) gives a NaN bound, which
/// fails every comparison.
inline Tracked operator/(const Tracked& a, const Tracked& b) {
    const long double q = a.v / b.v;
    if (!(std::fabs(b.v) > b.e))
        return {q, std::numeric_limits<long double>::quiet_NaN()};
    const long double propagated =
        (a.e + std::fabs(q) / (1 - kUnitRoundoffLongDouble) * b.e) / (std::fabs(b.v) - b.e);
    return rounded(q, propagated);
}

} // namespace ac::test
