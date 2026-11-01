#pragma once

#include <limits>

namespace ac::cuda {

/// Of the last Jacobi sweep, over a set of cells: the change
/// max|u_{k+1} - u_k| and the magnitude max(|u_k|, |u_{k+1}|).
struct JacobiChange {
    double change = 0.0;
    double magnitude = 0.0;
};

/// The change and magnitude over the union of two sets of cells: the larger
/// of each, NaN if either is NaN (std::max would drop a NaN operand).
[[nodiscard]] JacobiChange max_of(const JacobiChange& a, const JacobiChange& b);

/// Stopping rule of the IMEX thermal solve (docs/THEORY.md 10.2).
///
/// A Jacobi sweep maps u_k to u_{k+1} = (rhs + off-diagonal part) / (1 + s),
/// s = dt D times the sum of the stencil's off-diagonal weights. For every
/// boundary condition SimulationConfig::validate accepts, the sweep contracts
/// the error by q = s / (1 + s) in the max norm, so after a sweep that changed
/// the iterate by at most d (largest over the cells) the error of the new
/// iterate is at most s d in exact arithmetic. Rounding adds at most
/// gamma_K ((1 + 3s) m + (1 + 2s) error), with m the largest |u| of the two
/// iterates and K the number of roundings one term of a sweep passes through
/// (10.2 counts 35 for the 7-point and 80 for the 27-point stencil):
///
///   error <= bound(d, m) = (s d + gamma_K (1 + 3s) m) / (1 - gamma_K (1 + 2s)).
class JacobiStoppingRule {
public:
    JacobiStoppingRule(double s, int rounding_units, double tolerance);

    /// After `sweeps` sweeps, with d = c.change and m = c.magnitude over every
    /// cell: true when bound(d, m) <= tolerance. Throws std::runtime_error
    /// when d or m is not finite; when the rounding term alone, bound(0, m),
    /// reaches the tolerance (no number of sweeps certifies it); or when d is
    /// not smaller than at the previous call (in exact arithmetic it shrinks
    /// by q per sweep, so rounding has stopped the iteration above the
    /// tolerance).
    bool converged(int sweeps, const JacobiChange& c);

    [[nodiscard]] double bound(double change, double magnitude) const;
    [[nodiscard]] double s() const { return s_; }

private:
    double s_;
    double gamma_;
    double tolerance_;
    double previous_change_ = std::numeric_limits<double>::infinity();
};

} // namespace ac::cuda
