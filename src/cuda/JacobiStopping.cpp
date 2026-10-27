#include "cuda/JacobiStopping.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ac::cuda {

namespace {

std::string sci(double v) {
    std::ostringstream os;
    os << std::scientific << std::setprecision(3) << v;
    return os.str();
}

double nan_propagating_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    return std::max(a, b);
}

} // namespace

JacobiChange max_of(const JacobiChange& a, const JacobiChange& b) {
    return {nan_propagating_max(a.change, b.change), nan_propagating_max(a.magnitude, b.magnitude)};
}

JacobiStoppingRule::JacobiStoppingRule(double s, int rounding_units, double tolerance)
    : s_(s), tolerance_(tolerance) {
    if (!(s >= 0.0) || !std::isfinite(s) || rounding_units < 1 || !(tolerance > 0.0))
        throw std::invalid_argument(
            "JacobiStoppingRule: needs finite s >= 0, rounding_units >= 1, tolerance > 0");
    // gamma_K = K u / (1 - K u), u = 2^-53 the unit roundoff of double.
    const double ku = rounding_units * (std::numeric_limits<double>::epsilon() / 2.0);
    gamma_ = ku / (1.0 - ku);
}

double JacobiStoppingRule::bound(double change, double magnitude) const {
    const double denominator = 1.0 - gamma_ * (1.0 + 2.0 * s_);
    if (!(denominator > 0.0))
        return std::numeric_limits<double>::infinity();
    return (s_ * change + gamma_ * (1.0 + 3.0 * s_) * magnitude) / denominator;
}

bool JacobiStoppingRule::converged(int sweeps, const JacobiChange& c) {
    const double change = c.change, magnitude = c.magnitude;
    if (!std::isfinite(change) || !std::isfinite(magnitude)) {
        throw std::runtime_error("IMEX thermal solve: the Jacobi iterate is not finite after " +
                                 std::to_string(sweeps) + " sweeps (change " + sci(change) +
                                 ", max |u| " + sci(magnitude) + "): the simulation diverged");
    }
    const double error_bound = bound(change, magnitude);
    if (error_bound <= tolerance_)
        return true;
    const double rounding = bound(0.0, magnitude);
    if (rounding >= tolerance_) {
        throw std::runtime_error(
            "IMEX thermal solve: the rounding bound " + sci(rounding) +
            " of the Jacobi sweeps reaches the tolerance " + sci(tolerance_) + " (s = " + sci(s_) +
            ", max |u| = " + sci(magnitude) +
            "): no number of sweeps certifies it; reduce dt (docs/THEORY.md 10.2)");
    }
    if (!(change < previous_change_)) {
        throw std::runtime_error(
            "IMEX thermal solve: the change between Jacobi iterates stopped decreasing after " +
            std::to_string(sweeps) + " sweeps (" + sci(previous_change_) + ", then " + sci(change) +
            "; error bound " + sci(error_bound) + " > tolerance " + sci(tolerance_) + ", s = " +
            sci(s_) + "): rounding limits the solve at this dt; reduce dt (docs/THEORY.md 10.2)");
    }
    previous_change_ = change;
    return false;
}

} // namespace ac::cuda
