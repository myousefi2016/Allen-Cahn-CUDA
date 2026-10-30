#pragma once

// Binary128 references for the liquid-mode tests (docs/THEORY.md 10.1): the definitions
// evaluated in __float128, whose 113-bit significand puts its rounding errors some 2^60
// below the double rounding that every certified enclosure adds at its last operation.
// The Hessian comes from forward differentiation of the kernel's flux, independently of
// the closed form gradient_energy_hessian and the certified search use.

#include "core/LiquidModesDetail.hpp"

#include <quadmath.h>
#include <string>

namespace ac::test {

using Q = __float128;
using liquid_modes_detail::Objective;

inline std::string qstr(Q x) {
    char buf[64];
    quadmath_snprintf(buf, sizeof buf, "%.36Qg", x);
    return buf;
}

/// v + d t: a binary128 number and its derivative along one direction.
struct QDual {
    Q v, d;
};
inline QDual operator+(QDual a, QDual b) {
    return {a.v + b.v, a.d + b.d};
}
inline QDual operator-(QDual a, QDual b) {
    return {a.v - b.v, a.d - b.d};
}
inline QDual operator*(QDual a, QDual b) {
    return {a.v * b.v, a.d * b.v + a.v * b.d};
}
inline QDual operator/(QDual a, QDual b) {
    return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)};
}
inline QDual qsin(QDual x) {
    return {sinq(x.v), cosq(x.v) * x.d};
}
inline QDual constant(Q c) {
    return {c, 0};
}

/// Component i of Kernels.cuh anisotropic_flux / W0^2 at the gradient p:
///   A(p)^2 p_i + 16 eps A(p) dFunc(p_i, others),
///   dFunc = (p_i^3 (|p|^2 - p_i^2) - p_i (sum_j p_j^4 - p_i^4)) / |p|^4,
///   A(p) = 1 - 3 eps + 4 eps sum_j p_j^4 / |p|^4   (compute_An).
inline QDual flux(const QDual p[3], int i, Q eps) {
    const QDual r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    const QDual s4 =
        p[0] * p[0] * p[0] * p[0] + p[1] * p[1] * p[1] * p[1] + p[2] * p[2] * p[2] * p[2];
    const QDual r4 = r2 * r2;
    const QDual A = constant(1 - 3 * eps) + constant(4 * eps) * s4 / r4;
    const QDual pi2 = p[i] * p[i];
    const QDual dfunc = (p[i] * pi2 * (r2 - pi2) - p[i] * (s4 - pi2 * pi2)) / r4;
    return A * A * p[i] + constant(16 * eps) * A * dfunc;
}

/// H(v) = d flux / dp at p = v by forward differentiation (0-homogeneous: H of the
/// direction of v).
inline void flux_hessian(const Q v[3], Q eps, Q H[3][3]) {
    for (int j = 0; j < 3; ++j) {
        const QDual p[3] = {{v[0], j == 0 ? Q(1) : Q(0)},
                            {v[1], j == 1 ? Q(1) : Q(0)},
                            {v[2], j == 2 ? Q(1) : Q(0)}};
        for (int i = 0; i < 3; ++i)
            H[i][j] = flux(p, i, eps).d;
    }
}

/// The closed form of LiquidModes.hpp at n, n taken as a unit vector (as
/// gradient_energy_hessian takes it), in binary128.
inline void closed_form_hessian(const Q n[3], Q eps, Q H[3][3]) {
    const Q s = n[0] * n[0] * n[0] * n[0] + n[1] * n[1] * n[1] * n[1] + n[2] * n[2] * n[2] * n[2];
    const Q a = 1 - 3 * eps + 4 * eps * s;
    Q D[3];
    for (int i = 0; i < 3; ++i)
        D[i] = n[i] * (n[i] * n[i] - s);
    for (int i = 0; i < 3; ++i) {
        const Q ni2 = n[i] * n[i];
        H[i][i] =
            a * a + 32 * eps * a * D[i] * n[i] +
            16 * eps * (16 * eps * D[i] * D[i] + a * (3 * ni2 - 6 * ni2 * ni2 - s + 4 * s * ni2));
        for (int j = 0; j < 3; ++j)
            if (j != i)
                H[i][j] = 256 * eps * eps * D[i] * D[j] +
                          32 * eps * a * n[i] * n[j] * (s - ni2 - n[j] * n[j]);
    }
}

/// The physics a mode's value depends on.
struct Physics {
    double eps, beta, W0, h[3];
};

/// A(n)^2 for n the direction of v.
inline Q anisotropy_squared(const Q v[3], Q eps) {
    const Q r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    const Q s4 = v[0] * v[0] * v[0] * v[0] + v[1] * v[1] * v[1] * v[1] + v[2] * v[2] * v[2] * v[2];
    const Q A = 1 - 3 * eps + 4 * eps * s4 / (r2 * r2);
    return A * A;
}

/// Lam (sigma from the flux Hessian) and L of the mode with direction v and wave numbers
/// theta, as functions of theta with the derivative along theta_k (k = -1: none).
struct ModeValues {
    QDual lam, L;
};
inline ModeValues mode_values(const Physics& p, const double v[3], const double theta[3], int k) {
    const Q vq[3] = {v[0], v[1], v[2]};
    Q H[3][3];
    flux_hessian(vq, p.eps, H);
    QDual c[3], q[3], mu = constant(0);
    for (int d = 0; d < 3; ++d) {
        const QDual t = {theta[d], d == k ? Q(1) : Q(0)};
        const QDual half = qsin(t * constant(Q(0.5)));
        const Q h = p.h[d];
        c[d] = constant(4) * half * half / constant(h * h); // 2 (1 - cos theta_d) / h_d^2
        q[d] = qsin(t) / constant(h);
        mu = mu + c[d];
    }
    QDual sigma = constant(0);
    for (int d = 0; d < 3; ++d) {
        sigma = sigma + constant(H[d][d]) * c[d];
        for (int e = 0; e < 3; ++e)
            if (e != d)
                sigma = sigma + constant(H[d][e]) * q[d] * q[e];
    }
    const QDual W = constant(Q(p.W0) * Q(p.W0));
    const QDual Wm2 = W * mu + constant(2);
    return {W * sigma + constant(2), Wm2 * Wm2};
}

/// psi_k(theta) of the bilinear form F = sum_k phi_k psi_k (LiquidModes.cpp): W c_d
/// (k = 0..2), 2 W q_x q_y, 2 W q_x q_z, 2 W q_y q_z (k = 3..5) and
/// kappa - beta (W mu + 2)^2 with kappa = 2 (growth) or -2 (damping).
inline void psi(Objective o, const Physics& p, const double theta[3], Q out[7]) {
    const Q W = Q(p.W0) * Q(p.W0);
    Q c[3], q[3], mu = 0;
    for (int d = 0; d < 3; ++d) {
        const Q half = sinq(Q(theta[d]) / 2), h = p.h[d];
        c[d] = 4 * half * half / (h * h);
        q[d] = sinq(Q(theta[d])) / h;
        mu += c[d];
    }
    for (int d = 0; d < 3; ++d)
        out[d] = W * c[d];
    out[3] = 2 * W * q[0] * q[1];
    out[4] = 2 * W * q[0] * q[2];
    out[5] = 2 * W * q[1] * q[2];
    const Q Wm2 = W * mu + 2;
    out[6] = (o == Objective::Growth ? Q(2) : Q(-2)) - Q(p.beta) * Wm2 * Wm2;
}

/// F of the search at the mode, and with k >= 0 its derivative in theta_k.
inline QDual objective(Objective o, const Physics& p, const double v[3], const double theta[3],
                       int k = -1) {
    const ModeValues m = mode_values(p, v, theta, k);
    const QDual beta = constant(p.beta);
    if (o == Objective::Growth) {
        const Q vq[3] = {v[0], v[1], v[2]};
        return (m.lam - beta * m.L) / constant(anisotropy_squared(vq, p.eps));
    }
    return constant(0) - (m.lam + beta * m.L);
}

/// Lam + beta L at the mode.
inline Q damping(const Physics& p, const double v[3], const double theta[3]) {
    const ModeValues m = mode_values(p, v, theta, -1);
    return m.lam.v + Q(p.beta) * m.L.v;
}

/// phi_k (LiquidModes.cpp) at the direction of v.
inline void phis(Objective o, double eps, const double v[3], Q out[7]) {
    const Q vq[3] = {v[0], v[1], v[2]};
    Q H[3][3];
    flux_hessian(vq, eps, H);
    const bool growth = o == Objective::Growth;
    const Q w = growth ? 1 / anisotropy_squared(vq, eps) : Q(1);
    const Q sign = growth ? w : Q(-1);
    out[0] = sign * H[0][0];
    out[1] = sign * H[1][1];
    out[2] = sign * H[2][2];
    out[3] = sign * H[0][1];
    out[4] = sign * H[0][2];
    out[5] = sign * H[1][2];
    out[6] = w;
}

/// The direction (1, a, b) (face 0), (a, 1, b) (face 1) or (a, b, 1) (face 2).
inline void face_vector(int face, double a, double b, double v[3]) {
    v[0] = face == 0 ? 1.0 : a;
    v[1] = face == 0 ? a : face == 1 ? 1.0 : b;
    v[2] = face == 2 ? 1.0 : b;
}

} // namespace ac::test
