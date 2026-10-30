#pragma once

#include "core/SimulationConfig.hpp"
#include "cuda/Kernels.cuh"

#include <cuda_runtime.h>

namespace ac::cuda {

// ── Willmore-regularised phase-field update (linearly implicit Euler) ───────
//
// With h(phi) = phi - phi^3 and omega = W0^2 lap(phi) + h(phi), the Willmore
// energy (beta/2) int omega^2 adds -beta M(phi),
//   M = W0^2 lap(omega) + h'(phi) omega,   h'(phi) = 1 - 3 phi^2,
// to the Allen-Cahn right-hand side. Its stiff part is the bulk (phi = +-1)
// linearisation L = W0^4 lap^2 - 4 W0^2 lap + 4, taken implicitly: the
// increment dphi = phi^{n+1} - phi^n solves
//   (tau/dt + beta L) dphi = div F - dF/dphi - beta M(phi^n)      (all at t^n)
// on the interior cells. lap(phi) in omega reads phi's own boundary cells
// (phi_old, filled by apply_phi_bc and, in a multi-GPU run, on the halo and
// periodic X ghost planes by the halo exchange). Every other lap (of omega,
// of the CG direction and of its lap, and in the preconditioner) acts on a
// field whose boundary cells are set by the homogeneous form of phi's BCs
// (Dirichlet 0, zero flux, periodic, Robin with gamma = 0). That lap is
// symmetric, so the matrix is symmetric positive definite (L is a square,
// tau > 0). It is solved by flexible conjugate gradients in double precision
// with a Chebyshev polynomial preconditioner applied in single precision
// (WillmoreSolve.cuh). beta = 0 gives explicit Euler. All laps are 7-point.

/// omega = W0^2 lap(phi) + phi - phi^3 on interior cells (boundary cells untouched).
void launch_willmore_omega(const double* phi, double* omega, const KernelParams& p,
                           cudaStream_t stream);

/// On interior cells: tau = tau0 A(n)^2 and r = div F - dF/dphi - beta M (the
/// right-hand side, from omega with its homogeneous-BC boundary cells set).
/// x = z = dir = 0 on every cell and r = tau = 0 on boundary cells.
void launch_willmore_rhs_init(const double* phi, const double* u, const double* omega, double* tau,
                              double* x, double* r, double* z, double* dir, const KernelParams& p,
                              cudaStream_t stream);

/// dst = lap(src) on interior cells.
void launch_laplacian_interior(const double* src, double* dst, const KernelParams& p,
                               cudaStream_t stream);

/// Ap = (tau/dt) dir + beta (W0^4 lap(t) - 4 W0^2 t + 4 dir), t = lap(dir), on interior cells.
void launch_willmore_apply(const double* dir, const double* t, const double* tau, double* Ap,
                           const KernelParams& p, cudaStream_t stream);

/// x += alpha dir, r -= alpha Ap on interior cells.
void launch_cg_update_xr(double* x, double* r, const double* dir, const double* Ap, double alpha,
                         const KernelParams& p, cudaStream_t stream);

/// dir = z + beta dir on interior cells.
void launch_cg_direction(const double* z, double* dir, double beta, const KernelParams& p,
                         cudaStream_t stream);

/// phi_new = phi_old + x on interior cells.
void launch_willmore_finish(const double* phi_old, const double* x, double* phi_new,
                            const KernelParams& p, cudaStream_t stream);

/// For every X plane in [x_begin, x_end): out[x - x_begin] = sum over the
/// plane's interior cells (1 <= y <= Ny-2, 1 <= z <= Nz-2) of a*b. Each plane
/// is reduced by one block in a fixed order, so a plane's sum depends only on
/// its values: summing the planes in global order gives the same bits however
/// the domain is split across GPUs.
void launch_plane_dot(const double* a, const double* b, double* out, int x_begin, int x_end,
                      const KernelParams& p, cudaStream_t stream);

// ── Single-precision Chebyshev preconditioner ──────────────────────────────
//
// The preconditioner's fields live in the same padded Nx*Ny*Nz layout as the
// double-precision ones, so multi-GPU halo planes are exchanged the same way.
// Every kernel that produces such a field also writes the face ghost cells
// that depend on it (the 7-point stencil reads no edge or corner ghost): the
// homogeneous rule of the face, or, on a periodic axis, the ghost at the
// opposite face, so no separate boundary-condition launch is needed.

/// Homogeneous ghost rule of each face (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi):
/// kGhostZero (Dirichlet), kGhostScaled (ghost = coef * adjacent interior
/// value: Neumann coef = 1, Robin coef = robin_ghost_coefficient) or
/// kGhostPeriodic (ghost = interior value next to the opposite face).
enum GhostKind : int { kGhostZero = 0, kGhostScaled = 1, kGhostPeriodic = 2 };
struct GhostRules32 {
    int kind[6];
    float coef[6];
};
/// Ghost rules of the homogeneous form of `faces` on the grid of `p`.
GhostRules32 make_ghost_rules32(const PerFaceBoundary& faces, const KernelParams& p);

/// Constants of the single-precision operator
///   A x = tdt x + beta (W0^4 lap(t) - 4 W0^2 t + 4 x),  t = lap(x),
/// and of its diagonal scaling D = tdt + beta d_L, tdt = tau/dt.
struct Operator32 {
    float inv_hx2, inv_hy2, inv_hz2; ///< 1/h^2 per axis
    float beta, W4, four_W2;         ///< beta, W0^4, 4 W0^2
    float beta_dL;                   ///< beta d_L
};
/// Constants for the grid and physics of `p` and the diagonal constant d_L.
Operator32 make_operator32(const KernelParams& p, double d_L);

/// On interior cells: tdt[c] = tau[c] / dt in single precision (tau * (1/dt)
/// rounded once). Returns through range_bits the exact min and max of tau
/// over the interior cells as the bit patterns of positive doubles (their
/// order as unsigned integers is their numeric order; a NaN tau yields a NaN
/// maximum). range_bits must hold two values, initialised by this call.
void launch_precond_tau_scan(const double* tau, float* tdt, double inv_dt,
                             unsigned long long* range_bits, const KernelParams& p,
                             cudaStream_t stream);

/// First Chebyshev step from zero: f = scale * r (rounded to single) and
/// x1 = s0 D^-1 f on interior cells, x1's face ghosts from `ghosts`. With
/// z_final != nullptr it instead writes z_final = x1 * inv_scale (double)
/// on interior cells (degree-1 plan).
void launch_precond_first(const double* r, double scale, float* f, float* x1, const float* tdt,
                          Operator32 op, float s0, GhostRules32 ghosts, double* z_final,
                          double inv_scale, const KernelParams& p, cudaStream_t stream);

/// t = lap(x) on interior cells (single precision), t's face ghosts from `ghosts`.
void launch_precond_laplacian(const float* x, float* t, Operator32 op, GhostRules32 ghosts,
                              const KernelParams& p, cudaStream_t stream);

/// Chebyshev step: x_next = x + a (x - x_prev) + b D^-1 (f - A x) on interior
/// cells, with t = lap(x) (face ghosts set). x_prev_next holds x_prev on
/// entry (x_prev = 0 when prev_is_zero) and x_next on exit, with its face
/// ghosts from `ghosts`. With z_final != nullptr it instead writes
/// z_final = x_next * inv_scale (double) on interior cells.
void launch_precond_step(const float* t, const float* x, float* x_prev_next, bool prev_is_zero,
                         const float* f, const float* tdt, Operator32 op, float a, float b,
                         GhostRules32 ghosts, double* z_final, double inv_scale,
                         const KernelParams& p, cudaStream_t stream);

} // namespace ac::cuda
