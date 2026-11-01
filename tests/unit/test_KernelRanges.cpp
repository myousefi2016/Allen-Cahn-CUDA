#include "core/KernelRanges.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <quadmath.h>
#include <random>
#include <vector>

using namespace ac;

namespace {

using Q = __float128;

/// compute_An (Kernels.cuh) on the host with each occurrence of 1 - 3 eps and of
/// 1 + c1 q evaluated plain or fused, the patterns a compiler may produce; `pattern`
/// selects one of the 8. The products are stored so that the host compiler cannot fuse
/// the plain ones itself.
double compute_an(double px, double py, double pz, double eps, int pattern) {
    auto plain = [](double a, double b, double c) {
        volatile double p = a * b;
        return p + c;
    };
    volatile double x2 = px * px, y2 = py * py, z2 = pz * pz;
    volatile double sq = x2 + y2;
    sq = sq + z2;
    if (!(sq > 1e-30))
        return 1.0 - 3.0 * eps / 5.0;
    volatile double x4 = px * px;
    x4 = x4 * px;
    x4 = x4 * px;
    volatile double y4 = py * py;
    y4 = y4 * py;
    y4 = y4 * py;
    volatile double z4 = pz * pz;
    z4 = z4 * pz;
    z4 = z4 * pz;
    volatile double qrt = x4 + y4;
    qrt = qrt + z4;
    volatile double sq2 = sq * sq;
    const double q = qrt / sq2;
    const double c0_div = (pattern & 1) ? std::fma(-3.0, eps, 1.0) : plain(-3.0, eps, 1.0);
    const double c0_mul = (pattern & 2) ? std::fma(-3.0, eps, 1.0) : plain(-3.0, eps, 1.0);
    const double c1 = 4.0 * eps / c0_div;
    const double t = (pattern & 4) ? std::fma(c1, q, 1.0) : plain(c1, q, 1.0);
    return c0_mul * t;
}

/// Gradients of every kind compute_An meets: the axes, face and body diagonals, random
/// directions at magnitudes from just above the zero-gradient threshold to 1e70 (fourth
/// powers below the double range), and components spanning the subnormal range.
std::vector<std::array<double, 3>> gradients() {
    std::vector<std::array<double, 3>> g;
    for (const double m : {1.0000000000000001e-15, 1e-10, 1e-3, 0.4, 1.0, 7.0, 1e5, 1e40, 1e70}) {
        g.push_back({m, 0.0, 0.0});
        g.push_back({0.0, -m, 0.0});
        g.push_back({m, m, 0.0});
        g.push_back({m, m, m});
        g.push_back({-m, m, -m});
        g.push_back({m, 1e-300, 5e-324});
        g.push_back({m, m * 1e-160, 0.0});
    }
    std::mt19937_64 rng(20261030);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> exponent(-14.0, 70.0);
    for (int i = 0; i < 200000; ++i) {
        const double scale = std::pow(10.0, exponent(rng));
        g.push_back({scale * normal(rng), scale * normal(rng), scale * normal(rng)});
    }
    return g;
}

} // namespace

// The bounds of compute_An's rounded quotient q = qrt / (sq sq): the largest double at most
// (1 + u)^6 / (1 - u)^7 (1 + 1e-250) and the smallest at least (1/3)(1 - u)^6 / (1 + u)^7
// (1 - 1e-250), u = 2^-53, in binary128, whose 113-bit significand holds both bounds to
// far below the spacing of doubles (the slack 1e-250 of underflowed terms is below that
// resolution too, and far below the distance, about 2^-53, from the bounds to the next
// doubles, so it cannot move them).
TEST(KernelRanges, QuotientBoundsAreTheDoublesNextToTheRoundingBounds) {
    const Q u = 1 / powq(2, 53);
    const Q slack = Q(1e-250);
    const Q hi = powq(1 + u, 6) / powq(1 - u, 7) * (1 + slack);
    const Q lo = (1 / Q(3)) * powq(1 - u, 6) / powq(1 + u, 7) * (1 - slack);
    EXPECT_TRUE(Q(kAnisotropyQuotientMax) <= hi);
    EXPECT_TRUE(Q(std::nextafter(kAnisotropyQuotientMax, 2.0)) > hi);
    EXPECT_TRUE(Q(kAnisotropyQuotientMin) >= lo);
    EXPECT_TRUE(Q(std::nextafter(kAnisotropyQuotientMin, 0.0)) < lo);
}

// d_L for the spacing of the shipped configurations, against binary128.
TEST(KernelRanges, WillmoreInteriorDiagonal) {
    const double h[3] = {0.4, 0.4, 0.4};
    EXPECT_EQ(willmore_interior_diagonal(1.0, h), 1794.6249999999995);
    for (const double W0 : {0.5, 1.0, 3.0})
        for (const double hx : {0.25, 0.4, 1.0 / 3.0}) {
            const double hh[3] = {hx, 0.4, 0.5};
            Q s1 = 0, s2 = 0;
            for (const double v : hh) {
                s1 += 1 / (Q(v) * Q(v));
                s2 += 1 / (Q(v) * Q(v) * Q(v) * Q(v));
            }
            s1 *= 2;
            const Q W2 = Q(W0) * Q(W0);
            const Q exact = W2 * W2 * (s1 * s1 + 2 * s2) + 4 * W2 * s1 + 4;
            const double got = willmore_interior_diagonal(W0, hh);
            EXPECT_TRUE(fabsq((Q(got) - exact) / exact) < Q(1e-14)) << W0 << " " << hx;
        }
}

// Every A compute_An gives, in each of its 8 evaluation patterns, lies in [a_lo, a_hi] of
// relaxation_time_range, and so does every tau = (tau0 A) A; the bounds are tight: the
// axes and body diagonals reach them to within a few units in the last place.
TEST(KernelRanges, RelaxationTimeRangeContainsEveryOrientation) {
    const auto g = gradients();
    for (const double eps :
         {0.0, 0.02, 0.05, 1.0 / 15.0, 0.12, 0.25, 0.3, std::nextafter(1.0 / 3.0, 0.0)}) {
        SCOPED_TRACE(eps);
        const RelaxationTimeRange r = relaxation_time_range(2.0, eps);
        double seen_lo = std::numeric_limits<double>::infinity(), seen_hi = 0.0;
        for (const auto& p : g) {
            for (int pattern = 0; pattern < 8; ++pattern) {
                const double a = compute_an(p[0], p[1], p[2], eps, pattern);
                ASSERT_TRUE(a >= r.a_lo && a <= r.a_hi)
                    << "A = " << a << " outside [" << r.a_lo << ", " << r.a_hi << "] at (" << p[0]
                    << ", " << p[1] << ", " << p[2] << "), pattern " << pattern;
                const double tau = 2.0 * a * a;
                ASSERT_TRUE(tau >= r.lo && tau <= r.hi);
                seen_lo = std::min(seen_lo, a);
                seen_hi = std::max(seen_hi, a);
            }
        }
        // The exact A lies in [1 - 5 eps/3, 1 + eps]; the bounds add 4 eps times the slack of
        // q's bounds (6 2^-52 above 1, 8 2^-54 below 1/3) and three roundings of A, less
        // than 2^-48 relative. The gradients reach the exact ends (q = 1 on the axes, 1/3
        // on the body diagonals) to within their own roundings.
        EXPECT_LE(r.a_hi, (1.0 + eps) * (1.0 + 0x1p-48));
        EXPECT_GE(r.a_lo, (1.0 - 5.0 * eps / 3.0) * (1.0 - 0x1p-48));
        EXPECT_GE(seen_hi, (1.0 + eps) * (1.0 - 0x1p-48));
        EXPECT_LE(seen_lo, (1.0 - 5.0 * eps / 3.0) * (1.0 + 0x1p-48));
        if (eps == 0.0) {
            EXPECT_EQ(r.a_lo, 1.0);
            EXPECT_EQ(r.a_hi, 1.0);
        }
    }
}
