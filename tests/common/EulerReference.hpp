#pragma once

// Host reference of one explicit Euler step of cuda::CudaSolver (CudaSolver::step_euler with
// physics.willmore_beta = 0 and uniform boundary conditions):
//   phi' = phi + dt (div F(grad phi) - dF/dphi(phi, u)) / tau        at interior cells,
//   u'   = u + (phi' - phi) / 2 + dt D lap(u)                        at interior cells,
// then phi's and u's boundary cells from their BCs. The operators are the kernels' discrete
// definitions (Kernels.cuh, ThermalKernels.cu, BoundaryKernels.cu), written here
// independently and evaluated in long double:
//  - div F = sum_d [F_d(upper face) - F_d(lower face)] / h_d, the gradient at a face being the
//    compact difference of its two cells (normal) and the mean of their central differences
//    (tangential), with F in closed form
//      F_d(p) = W0^2 A B_d p_d,  n = p / |p|,  Q = sum_i n_i^4,
//      A = 1 - 3 eps + 4 eps Q,  B_d = 1 - 3 eps + 4 eps (4 n_d^2 - 3 Q),
//    which is the kernel's w^2 p_d + 16 eps W0 w dFunc (w = W0 A, dFunc = p_d (n_d^2 - Q)) in
//    exact arithmetic; for |p|^2 <= 1e-30 (the double nearest 1e-30) A is the zero-gradient
//    value 1 - 3 eps / 5 and dFunc is 0, as in compute_An and dFunc;
//  - tau = tau0 A(c)^2, c the cell's central differences (allen_cahn_tau);
//  - dF/dphi = -phi (1 - phi^2) + lambda u (1 - phi^2)^2;
//  - lap: the 7-point Laplacian, or the 27-point one (14 faces, 3 edges, 1 corners, -128 centre,
//    over 30 h^2);
//  - a zero-flux phi boundary cell holds the interior cell nearest to it, each index clamped to
//    [1, N - 2] (apply_bc_face_kernel copies the adjacent cell; with the faces applied Z, Y, X
//    an edge or corner cell gets the copy of a copy); a Dirichlet cell holds its value.
//
// Each result comes with a bound B per cell on |GPU value - reference value| when the GPU's
// step and this reference start from the same double fields: the sum of forward rounding-error
// bounds (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., ch. 3) of the
// kernels' double arithmetic and of this file's long double arithmetic, each against the exact
// real-arithmetic step from those fields. The rounding counts are those of the kernels'
// expressions, stated at each term. nvcc contracts a * b + c into an FMA (its default
// --fmad=true) but never reassociates; an FMA rounds once, so contraction only shortens a
// chain, and the bounds hold whichever contractions it makes.

#include "core/SimulationConfig.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ac::test {

using LD = long double;

/// The unit roundoff of an arithmetic and a factor on the rounding counts.
struct Rounding {
    LD unit;
    int chain;
    /// gamma_n = n u / (1 - n u), n = count * chain: a product of n factors (1 + delta)^(+-1),
    /// |delta| <= u, is 1 + theta with |theta| <= gamma_n (Higham, Lemma 3.1).
    /// Not constexpr: nvcc's --expt-relaxed-constexpr would also compile a constexpr function
    /// for the device, where long double is double (warning 20208-D).
    [[nodiscard]] LD gamma(int count) const {
        const LD nu = static_cast<LD>(count) * static_cast<LD>(chain) * unit;
        return nu / (1.0L - nu);
    }
};

/// The kernels' double arithmetic; the counts below are its chains.
inline constexpr Rounding kKernelDouble{std::numeric_limits<double>::epsilon() / 2, 1};
/// This file's long double arithmetic: its chains are at most twice the kernels' (the
/// closed-form flux about 20 roundings to the kernel's 47, tau about 65 to its 77).
inline constexpr Rounding kReferenceLongDouble{std::numeric_limits<LD>::epsilon() / 2, 2};
inline constexpr Rounding kRoundings[2] = {kKernelDouble, kReferenceLongDouble};
/// The bounds are sums, products and quotients of nonnegative terms (and the difference of the
/// two tau branches, whose rounding is far below the padding gamma_77 tau added to it),
/// evaluated in long double in fewer than 256 operations; the comparison |GPU - reference|
/// takes one more. This factor makes every computed bound at least its exact value.
inline const LD kBoundSlack = 1.0L + Rounding{std::numeric_limits<LD>::epsilon() / 2, 1}.gamma(256);
/// The zero-gradient threshold of compute_An and dFunc: the double 1e-30.
inline constexpr LD kZeroGradient = 1e-30;

/// Constants of F_d = W0^2 A B_d p_d for 0 <= eps <= 1/15.
///  K: |F_d(p)| <= W0^2 K |p|, and the magnitudes of the kernel's terms w^2 |p_d|,
///     16 eps W0 w |p_d|^3 (p_a^2 + p_b^2) / |p|^4 and 16 eps W0 w |p_d| (p_a^4 + p_b^4) / |p|^4
///     sum to at most W0^2 K |p|, in every branch of compute_An and dFunc. Here A <= 1 + eps
///     (Q <= 1), the zero-gradient A = 1 - 3 eps / 5 <= 1, |dFunc| = |p_d| |n_d^2 - Q| <= |p_d|,
///     and n_d^2 (1 - n_d^2) + sum_{e != d} n_e^4 <= (1 - n_d^2)(n_d^2 + 1 - n_d^2) <= 1, so
///     K = (1 + eps)(1 + 16 eps).
///  L: |dF_d/dp_e| <= W0^2 L for p != 0. Write F_d = |p| G_d(n); then
///     dF_d/dp_e = G_d n_e + [(I - n n^T) grad G_d]_e, which is at most |G_d| + |grad G_d| for
///     the polynomial extension G_d = (a + b sum n_i^4)(a + b (4 n_d^2 - 3 sum n_i^4)) n_d
///     (a = 1 - 3 eps, b = 4 eps) on the unit sphere. Use |A| <= 1 + eps, |B_d| <= 1 + 9 eps
///     (B_d lies in [1 - 15 eps, 1 + 17 eps / 9]), |grad A| = b |4 n^3| <= 16 eps,
///     |grad B_d| = b |8 n_d e_d - 12 n^3| <= 80 eps and |grad n_d| = 1. Hence
///     L = 2 (1 + eps)(1 + 9 eps) + 16 eps (1 + 9 eps) + 80 eps (1 + eps).
struct FluxConstants {
    LD K, L;
    explicit FluxConstants(LD eps)
        : K((1 + eps) * (1 + 16 * eps)),
          L(2 * (1 + eps) * (1 + 9 * eps) + 16 * eps * (1 + 9 * eps) + 80 * eps * (1 + eps)) {}
};

/// A field after a reference step, and per cell the bound on |GPU value - value|.
struct ReferenceStep {
    std::vector<LD> value;
    std::vector<LD> bound;
};

class EulerReference {
public:
    explicit EulerReference(const SimulationConfig& cfg)
        : nx_(cfg.grid.Nx), ny_(cfg.grid.Ny), nz_(cfg.grid.Nz),
          stride_{static_cast<std::size_t>(cfg.grid.Ny) * static_cast<std::size_t>(cfg.grid.Nz),
                  static_cast<std::size_t>(cfg.grid.Nz), 1},
          h_{cfg.grid.dx, cfg.grid.dy, cfg.grid.dz},
          w2_(static_cast<LD>(cfg.physics.W0) * static_cast<LD>(cfg.physics.W0)),
          eps_(cfg.physics.epsilon), tau0_(cfg.physics.tau0()), lambda_(cfg.physics.lambda()),
          d_(cfg.physics.D), stencil27_(cfg.stencil == StencilType::Isotropic27Point),
          phi_bc_(cfg.boundary.phi_bc), u_bc_(cfg.boundary.u_bc), flux_(cfg.physics.epsilon) {
        if (cfg.boundary.per_face || cfg.physics.willmore_beta != 0.0)
            throw std::invalid_argument("EulerReference: uniform BCs and willmore_beta = 0 only");
        const bool phi_ok = phi_bc_.type == BCType::Dirichlet ||
                            (phi_bc_.type == BCType::Neumann && phi_bc_.flux == 0.0);
        if (!phi_ok || u_bc_.type != BCType::Dirichlet)
            throw std::invalid_argument(
                "EulerReference: phi Dirichlet or zero-flux Neumann, u Dirichlet only");
        // FluxConstants, and gamma_2 for the kernel's 1 - 3 eps (relative error at most
        // u (1 + 3 eps / (1 - 3 eps))), hold for 0 <= eps <= 1/15, the unregularised range.
        if (!(cfg.physics.epsilon >= 0.0 && cfg.physics.epsilon <= 1.0 / 15.0))
            throw std::invalid_argument("EulerReference: 0 <= epsilon <= 1/15 only");
    }

    [[nodiscard]] std::size_t size() const { return stride_[0] * static_cast<std::size_t>(nx_); }
    [[nodiscard]] std::size_t index(int x, int y, int z) const {
        return static_cast<std::size_t>(x) * stride_[0] + static_cast<std::size_t>(y) * stride_[1] +
               static_cast<std::size_t>(z);
    }
    [[nodiscard]] bool is_boundary(int x, int y, int z) const {
        return x == 0 || y == 0 || z == 0 || x == nx_ - 1 || y == ny_ - 1 || z == nz_ - 1;
    }

    /// phi after one step of dt from the GPU's fields phi and u (x-major, as FieldData::data()).
    [[nodiscard]] ReferenceStep phi_step(const double* phi, const double* u, double dt) const {
        ReferenceStep out{std::vector<LD>(phi, phi + size()), std::vector<LD>(size(), 0.0L)};
        for (int x = 1; x < nx_ - 1; ++x)
            for (int y = 1; y < ny_ - 1; ++y)
                for (int z = 1; z < nz_ - 1; ++z)
                    phi_cell(phi, u, dt, index(x, y, z), out);
        for (int x = 0; x < nx_; ++x)
            for (int y = 0; y < ny_; ++y)
                for (int z = 0; z < nz_; ++z) {
                    if (!is_boundary(x, y, z))
                        continue;
                    const std::size_t c = index(x, y, z);
                    if (phi_bc_.type == BCType::Dirichlet) {
                        out.value[c] = phi_bc_.value; // written exactly by the kernel
                        out.bound[c] = 0.0L;
                    } else { // an exact copy of the nearest interior cell
                        const std::size_t s =
                            index(std::clamp(x, 1, nx_ - 2), std::clamp(y, 1, ny_ - 2),
                                  std::clamp(z, 1, nz_ - 2));
                        out.value[c] = out.value[s];
                        out.bound[c] = out.bound[s];
                    }
                }
        return out;
    }

    /// u after one step of dt from u, given phi before (phi_old) and after (phi_new) the step,
    /// exactly as the thermal kernel reads them (phi_new already carries phi's BCs).
    [[nodiscard]] ReferenceStep u_step(const double* u, const double* phi_old,
                                       const double* phi_new, double dt) const {
        ReferenceStep out{std::vector<LD>(u, u + size()), std::vector<LD>(size(), 0.0L)};
        const LD dtd = static_cast<LD>(dt) * d_;
        for (int x = 1; x < nx_ - 1; ++x)
            for (int y = 1; y < ny_ - 1; ++y)
                for (int z = 1; z < nz_ - 1; ++z) {
                    const std::size_t c = index(x, y, z);
                    const LD m = u[c];
                    LD lap = 0.0L, lap_magnitude = 0.0L;
                    int lap_roundings = 0;
                    if (!stencil27_) {
                        // Per axis (a + b - 2 m) / (h h): a sum, a difference, h h and the
                        // quotient; then two sums of the axis terms: 6 roundings.
                        for (int d = 0; d < 3; ++d) {
                            const LD a = u[c + stride_[d]], b = u[c - stride_[d]];
                            const LD hh = h_[d] * h_[d];
                            lap += (a + b - 2.0L * m) / hh;
                            lap_magnitude +=
                                (std::fabs(a) + std::fabs(b) + 2.0L * std::fabs(m)) / hh;
                        }
                        lap_roundings = 6;
                    } else {
                        // (14 face + 3 edge + corner - 128 m) / (30 h h): the 12-term edge sum
                        // takes 11 roundings, its weight 1, the four-term total 3, 30 h h 2 and
                        // the quotient 1: 18 roundings.
                        LD sum = 0.0L, sum_magnitude = 0.0L;
                        for (int i = -1; i <= 1; ++i)
                            for (int j = -1; j <= 1; ++j)
                                for (int k = -1; k <= 1; ++k) {
                                    const int n = std::abs(i) + std::abs(j) + std::abs(k);
                                    if (n == 0)
                                        continue;
                                    const LD w = n == 1 ? 14.0L : n == 2 ? 3.0L : 1.0L;
                                    const LD v = u[index(x + i, y + j, z + k)];
                                    sum += w * v;
                                    sum_magnitude += w * std::fabs(v);
                                }
                        const LD den = 30.0L * h_[0] * h_[0];
                        lap = (sum - 128.0L * m) / den;
                        lap_magnitude = (sum_magnitude + 128.0L * std::fabs(m)) / den;
                        lap_roundings = 18;
                    }
                    const LD latent = (static_cast<LD>(phi_new[c]) - phi_old[c]) / 2.0L;
                    const LD diffusion = dtd * lap;
                    const LD next = m + latent + diffusion;
                    out.value[c] = next;
                    LD bound = 0.0L;
                    for (const Rounding& r : kRoundings) {
                        const LD e_lap = r.gamma(lap_roundings) * lap_magnitude;
                        // dt D and its product with lap: 2 roundings; 0.5 (phi' - phi): 1;
                        // u + latent + diffusion: 2.
                        const LD e_diffusion =
                            dtd * (e_lap * (1 + r.gamma(2)) + r.gamma(2) * std::fabs(lap));
                        const LD e_latent = r.gamma(1) * std::fabs(latent);
                        bound +=
                            (e_latent + e_diffusion) * (1 + r.gamma(2)) +
                            r.gamma(2) * (std::fabs(m) + std::fabs(latent) + std::fabs(diffusion));
                    }
                    out.bound[c] = bound * kBoundSlack;
                }
        for (int x = 0; x < nx_; ++x)
            for (int y = 0; y < ny_; ++y)
                for (int z = 0; z < nz_; ++z)
                    if (is_boundary(x, y, z)) {
                        out.value[index(x, y, z)] = u_bc_.value; // Dirichlet, written exactly
                        out.bound[index(x, y, z)] = 0.0L;
                    }
        return out;
    }

private:
    struct FaceFlux {
        LD F;        ///< F_d at the face, as defined above
        LD error[2]; ///< bounds on |kernel - exact| and |this file - exact|
    };

    /// F_d at the face between cell a and a + e_d.
    [[nodiscard]] FaceFlux face_flux(const double* phi, std::size_t a, int d) const {
        const std::size_t b = a + stride_[d];
        // Normal: (b - a) / h_d. The kernel rounds twice: relative error gamma_2.
        const LD pn = (static_cast<LD>(phi[b]) - phi[a]) / h_[d];
        // Tangential: (a+ - a- + b+ - b-) / (4 h_e). The kernel does three sums, 0.25 / h_e
        // and the product: gamma_5 times the sum of the four magnitudes over 4 h_e.
        LD pt[2], magnitude[2];
        for (int k = 0; k < 2; ++k) {
            const std::size_t s = stride_[(d + 1 + k) % 3];
            const LD h = h_[(d + 1 + k) % 3];
            const LD v0 = phi[a + s], v1 = phi[a - s], v2 = phi[b + s], v3 = phi[b - s];
            pt[k] = (v0 - v1 + v2 - v3) / (4.0L * h);
            magnitude[k] =
                (std::fabs(v0) + std::fabs(v1) + std::fabs(v2) + std::fabs(v3)) / (4.0L * h);
        }
        const LD r2 = pn * pn + pt[0] * pt[0] + pt[1] * pt[1];
        FaceFlux f{};
        if (r2 > kZeroGradient) {
            const LD n2 = pn * pn, t0 = pt[0] * pt[0], t1 = pt[1] * pt[1];
            const LD q = (n2 * n2 + t0 * t0 + t1 * t1) / (r2 * r2);
            const LD alpha = 1.0L - 3.0L * eps_, beta = 4.0L * eps_;
            f.F = w2_ * (alpha + beta * q) * (alpha + beta * (4.0L * n2 / r2 - 3.0L * q)) * pn;
        } else {
            const LD a0 = 1.0L - 3.0L * eps_ / 5.0L;
            f.F = w2_ * a0 * a0 * pn;
        }
        const LD p1 = std::fabs(pn) + std::fabs(pt[0]) + std::fabs(pt[1]);
        const LD norm = std::sqrt(r2);
        for (int k = 0; k < 2; ++k) {
            const Rounding& r = kRoundings[k];
            const LD e_p = r.gamma(2) * std::fabs(pn) + r.gamma(5) * (magnitude[0] + magnitude[1]);
            // Exact flux, kernel and reference all take the smooth branch when the computed
            // |p|^2, |p + dp|^2 (1 + theta_3), exceeds the threshold for every |dp|_1 <= e_p.
            // Then:
            //  - the kernel's rounding from its computed p is gamma_47 W0^2 K |p + dp|. The
            //    count: compute_An 21 (sq 3, qrt 5, sq sq, quotient, 4 eps / (1 - 3 eps) 3, the
            //    product, 1 +, times 1 - 3 eps); w = W0 A one more; w w p_d 46; 16 eps W0 w
            //    dFunc 39 (dFunc 14); their sum 47;
            //  - the input error moves F by at most W0^2 L e_p.
            // Otherwise either branch may be taken, and the error is at most |kernel| + |exact|.
            const bool smooth =
                norm > e_p && (norm - e_p) * (norm - e_p) * (1 - r.gamma(3)) > kZeroGradient;
            f.error[k] = smooth ? r.gamma(47) * w2_ * flux_.K * (p1 + e_p) + w2_ * flux_.L * e_p
                                : 2 * w2_ * flux_.K * (1 + r.gamma(47)) * (p1 + e_p);
        }
        return f;
    }

    void phi_cell(const double* phi, const double* u, double dt, std::size_t c,
                  ReferenceStep& out) const {
        LD div = 0.0L, flux_magnitude = 0.0L, flux_error[2] = {0.0L, 0.0L};
        for (int d = 0; d < 3; ++d) {
            const FaceFlux upper = face_flux(phi, c, d), lower = face_flux(phi, c - stride_[d], d);
            div += (upper.F - lower.F) / h_[d];
            flux_magnitude += (std::fabs(upper.F) + std::fabs(lower.F)) / h_[d];
            for (int k = 0; k < 2; ++k)
                flux_error[k] += (upper.error[k] + lower.error[k]) / h_[d];
        }
        const LD ph = phi[c];
        const LD omega = 1.0L - ph * ph;
        const LD lu = lambda_ * static_cast<LD>(u[c]);
        const LD g = -ph * omega + lu * omega * omega; // dF/dphi
        const LD force = div - g;
        // tau = tau0 A(c)^2 from the central differences c_d = (phi(+e_d) - phi(-e_d)) / (2 h_d).
        LD s2 = 0.0L, s4 = 0.0L;
        for (int d = 0; d < 3; ++d) {
            const LD cd =
                (static_cast<LD>(phi[c + stride_[d]]) - phi[c - stride_[d]]) / (2.0L * h_[d]);
            s2 += cd * cd;
            s4 += cd * cd * cd * cd;
        }
        const LD alpha = 1.0L - 3.0L * eps_, beta = 4.0L * eps_, a0 = 1.0L - 3.0L * eps_ / 5.0L;
        const LD a_s = s2 > 0.0L ? alpha + beta * s4 / (s2 * s2) : a0;
        const LD tau_s = tau0_ * a_s * a_s, tau_0 = tau0_ * a0 * a0;
        const LD tau = s2 > kZeroGradient ? tau_s : tau_0;
        const LD rate = force / tau;
        const LD next = static_cast<LD>(phi[c]) + static_cast<LD>(dt) * rate;
        out.value[c] = next;

        LD bound = 0.0L;
        for (int k = 0; k < 2; ++k) {
            const Rounding& r = kRoundings[k];
            // div F: each flux then passes through the axis difference, the division by h_d
            // and the two sums of the axis terms (4 roundings).
            const LD e_div =
                (1 + r.gamma(4)) * flux_error[k] + r.gamma(4) * (flux_magnitude + flux_error[k]);
            // dF/dphi. omega = 1 - phi^2 is off by at most gamma_2 (phi^2 + |omega|). Then
            // -phi omega is one product and lambda u omega omega three, plus one because tau0
            // and lambda are recomputed here from the config; the sum is one more rounding.
            const LD a_phi = std::fabs(ph), a_om = std::fabs(omega), a_lu = std::fabs(lu);
            const LD e_om = r.gamma(2) * (a_om + ph * ph);
            const LD e_t1 = a_phi * e_om * (1 + r.gamma(1)) + r.gamma(1) * a_phi * a_om;
            const LD e_t2 = a_lu * ((2 * a_om * e_om + e_om * e_om) * (1 + r.gamma(4)) +
                                    r.gamma(4) * a_om * a_om);
            const LD e_g =
                e_t1 + e_t2 + r.gamma(1) * (a_phi * a_om + e_t1 + a_lu * a_om * a_om + e_t2);
            const LD e_force =
                e_div + e_g + r.gamma(1) * (std::fabs(div) + e_div + std::fabs(g) + e_g);
            // tau. The kernel's central differences carry relative errors gamma_2, which move
            // Q = sum c^4 / |c|^4 by gamma_16 and A (a positive sum) by as much. compute_An
            // adds 21 roundings, tau0 (recomputed) 1 and tau0 A A 2: tau to gamma_77. Near
            // |c|^2 = 1e-30 (the computed |c|^2 is off by gamma_7) either branch may be taken,
            // so the interval [tau_lo, tau_hi] spans both.
            const bool either =
                s2 > kZeroGradient / (1 + r.gamma(8)) && s2 <= kZeroGradient / (1 - r.gamma(8));
            const LD lo = either ? std::min(tau_s, tau_0) : tau;
            const LD hi = either ? std::max(tau_s, tau_0) : tau;
            const LD gt = r.gamma(77);
            const LD tau_lo = lo * (1 - gt), tau_hi = hi * (1 + gt);
            const LD width = (hi - lo) + gt * (hi + lo); // tau_hi - tau_lo without cancellation
            // |F^/tau^ - F/tau| <= e_force / tau_lo + |F| (1/tau_lo - 1/tau_hi). Then the
            // quotient's rounding, dt times the rate (1 rounding) and the sum with phi (1).
            const LD e_rate = (e_force / tau_lo + std::fabs(force) * width / (tau_lo * tau_hi)) *
                                  (1 + r.gamma(1)) +
                              r.gamma(1) * std::fabs(rate);
            const LD e_step =
                static_cast<LD>(dt) * (e_rate * (1 + r.gamma(1)) + r.gamma(1) * std::fabs(rate));
            bound += e_step * (1 + r.gamma(1)) + r.gamma(1) * std::fabs(next);
        }
        out.bound[c] = bound * kBoundSlack;
    }

    int nx_, ny_, nz_;
    std::size_t stride_[3];
    LD h_[3];
    LD w2_, eps_, tau0_, lambda_, d_;
    bool stencil27_;
    BoundaryConfig phi_bc_, u_bc_;
    FluxConstants flux_;
};

} // namespace ac::test
