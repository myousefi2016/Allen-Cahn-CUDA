#pragma once

// Values the kernels compute from the configuration, as validate() needs them to reject a
// configuration whose run would leave the double or the single precision range.

namespace ac {

/// d_L, the interior diagonal per unit willmore_beta of the Willmore operator
/// L = W0^4 lap^2 - 4 W0^2 lap + 4 for the 7-point lap with spacings h: W0^4 (s1^2 +
/// 2 sum_d h_d^-4) + 4 W0^2 s1 + 4, s1 = 2 sum_d h_d^-2 (docs/THEORY.md 4.4). The Willmore
/// solve takes it from here (willmore_spectrum), so validate() checks the same double.
[[nodiscard]] double willmore_interior_diagonal(double W0, const double h[3]);

/// The largest and the smallest double q = qrt / (sq sq) of compute_An (Kernels.cuh) can
/// take for a gradient whose squares and fourth powers do not overflow. The exact quotient
/// sum p^4 / (sum p^2)^2 lies in [1/3, 1]; the kernel rounds each term of sq at most 3
/// times, each term of qrt at most 5 times (all terms are >= 0), sq sq once more and the
/// quotient once (a fused multiply-add only removes roundings), so its q lies in
/// [q (1-u)^6 / (1+u)^7, q (1+u)^6 / (1-u)^7], u = 2^-53. Terms that underflow change qrt
/// and sq by less than 2^-1072 against qrt >= sq^2 / 3 > 3e-61 (sq > 1e-30 on this branch),
/// relative 1e-250, which does not reach the next double of either bound
/// (tests/unit/test_KernelRanges.cpp checks both against binary128).
inline constexpr double kAnisotropyQuotientMax = 0x1.0000000000006p+0; // 1 + 6 2^-52
inline constexpr double kAnisotropyQuotientMin = 0x1.555555555554dp-2; // (1/3)(1 - 13 u), up

/// The extremes of the relaxation time tau = (tau0 A) A of allen_cahn_tau (Kernels.cuh)
/// over every interface orientation, as the kernel computes them: A = (1 - 3 eps) (1 + (4
/// eps / (1 - 3 eps)) q) at q in [kAnisotropyQuotientMin, kAnisotropyQuotientMax], with
/// each occurrence of 1 - 3 eps and 1 + c1 q evaluated plain or fused (the compiler may
/// contract either), and A = 1 - 3 eps / 5 at a zero gradient. Every operation is a rounded
/// double operation monotone in q and in A, so no orientation gives a tau outside
/// [lo, hi] (for tau0 > 0); lo and hi are taken at the bounds of q, which the kernel may
/// not reach exactly. Requires 0 <= epsilon < 1/3 (validate()).
struct RelaxationTimeRange {
    double a_lo, a_hi; ///< the extremes of A
    double lo, hi;     ///< (tau0 a_lo) a_lo and (tau0 a_hi) a_hi
};
[[nodiscard]] RelaxationTimeRange relaxation_time_range(double tau0, double epsilon);

} // namespace ac
