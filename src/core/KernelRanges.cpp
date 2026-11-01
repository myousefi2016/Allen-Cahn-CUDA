#include "core/KernelRanges.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace ac {

double willmore_interior_diagonal(double W0, const double h[3]) {
    double sum_inv_h2 = 0.0, sum_inv_h4 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        const double inv_h2 = 1.0 / (h[axis] * h[axis]);
        sum_inv_h2 += inv_h2;
        sum_inv_h4 += inv_h2 * inv_h2;
    }
    const double W2 = W0 * W0;
    const double s1 = 2.0 * sum_inv_h2;
    return W2 * W2 * (s1 * s1 + 2.0 * sum_inv_h4) + 4.0 * W2 * s1 + 4.0;
}

namespace {

/// a b + c with the product rounded on its own (the store keeps the host compiler from
/// contracting the two operations into one fused multiply-add).
double plain_multiply_add(double a, double b, double c) {
    volatile double product = a * b;
    return product + c;
}

} // namespace

RelaxationTimeRange relaxation_time_range(double tau0, double epsilon) {
    // compute_An: (1.0 - 3.0 * epsilon) * (1.0 + (4.0 * epsilon / (1.0 - 3.0 * epsilon)) * q).
    // 3.0 * epsilon and 4.0 * epsilon are exact for the epsilons validate() accepts (a
    // product with 4 is exact; 3 eps rounds once, plain or inside a fused 1 - 3 eps).
    const double c0_plain = plain_multiply_add(-3.0, epsilon, 1.0);
    const double c0_fused = std::fma(-3.0, epsilon, 1.0);
    RelaxationTimeRange r{};
    r.a_lo = r.a_hi = 1.0 - 3.0 * epsilon / 5.0; // zero gradient
    for (const double c0_quotient : {c0_plain, c0_fused}) {
        const double c1 = 4.0 * epsilon / c0_quotient;
        for (const double c0_product : {c0_plain, c0_fused}) {
            for (const double q : {kAnisotropyQuotientMin, kAnisotropyQuotientMax}) {
                for (const double t : {plain_multiply_add(c1, q, 1.0), std::fma(c1, q, 1.0)}) {
                    const double a = c0_product * t;
                    r.a_lo = std::min(r.a_lo, a);
                    r.a_hi = std::max(r.a_hi, a);
                }
            }
        }
    }
    r.lo = tau0 * r.a_lo * r.a_lo;
    r.hi = tau0 * r.a_hi * r.a_hi;
    return r;
}

} // namespace ac
