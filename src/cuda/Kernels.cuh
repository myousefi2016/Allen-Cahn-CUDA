#pragma once

#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"

#include <cuda_runtime.h>

namespace ac::cuda {

// ── Kernel parameter struct (passed by value, fits in registers) ──────────

struct KernelParams {
    int Nx, Ny, Nz;
    double dx, dy, dz, dt;
    double epsilon, W0, tau0, lambda, delta, D;
    int stencil_type; // 0 = 7pt, 1 = 27pt

    static KernelParams from_config(const SimulationConfig& cfg) {
        KernelParams p{};
        p.Nx = cfg.grid.Nx;
        p.Ny = cfg.grid.Ny;
        p.Nz = cfg.grid.Nz;
        p.dx = cfg.grid.dx;
        p.dy = cfg.grid.dy;
        p.dz = cfg.grid.dz;
        p.dt = cfg.time.dt;
        p.epsilon = cfg.physics.epsilon;
        p.W0 = cfg.physics.W0;
        p.tau0 = cfg.physics.tau0();
        p.lambda = cfg.physics.lambda();
        p.delta = cfg.physics.delta;
        p.D = cfg.physics.D;
        p.stencil_type = (cfg.stencil == StencilType::Isotropic27Point) ? 1 : 0;
        return p;
    }
};

// ── Index arithmetic ───────────────────────────────────────────────────────

__device__ __forceinline__ int idx3d(int x, int y, int z, int Ny, int Nz) {
    return static_cast<int>(static_cast<long long>(x) * Ny * Nz + y * Nz + z);
}

/// max(a, b) that propagates NaN. fmax returns the non-NaN operand, which lets
/// a diverged field reduce to a finite maximum and hide the blow-up from the
/// saturation guard and adaptive time stepping.
__device__ __forceinline__ double nan_max(double a, double b) {
    return (isnan(a) || isnan(b)) ? a + b : fmax(a, b);
}

__device__ __forceinline__ void linear_to_3d(int idx, int Ny, int Nz, int& x, int& y, int& z) {
    z = idx % Nz;
    y = (idx / Nz) % Ny;
    x = idx / (Ny * Nz);
}

// ── Stencil functions ──────────────────────────────────────────────────────

/// 2nd-order central difference gradient in X.
__device__ __forceinline__ double gradient_x(const double* __restrict__ phi, int x, int y, int z,
                                             int Ny, int Nz, double dx) {
    return (phi[idx3d(x + 1, y, z, Ny, Nz)] - phi[idx3d(x - 1, y, z, Ny, Nz)]) / (2.0 * dx);
}

/// 2nd-order central difference gradient in Y.
__device__ __forceinline__ double gradient_y(const double* __restrict__ phi, int x, int y, int z,
                                             int Ny, int Nz, double dy) {
    return (phi[idx3d(x, y + 1, z, Ny, Nz)] - phi[idx3d(x, y - 1, z, Ny, Nz)]) / (2.0 * dy);
}

/// 2nd-order central difference gradient in Z.
__device__ __forceinline__ double gradient_z(const double* __restrict__ phi, int x, int y, int z,
                                             int Ny, int Nz, double dz) {
    return (phi[idx3d(x, y, z + 1, Ny, Nz)] - phi[idx3d(x, y, z - 1, Ny, Nz)]) / (2.0 * dz);
}

/// 4th-order central difference gradient in X.
__device__ __forceinline__ double gradient_x_4th(const double* __restrict__ phi, int x, int y,
                                                 int z, int Ny, int Nz, double dx) {
    return (-phi[idx3d(x + 2, y, z, Ny, Nz)] + 8.0 * phi[idx3d(x + 1, y, z, Ny, Nz)] -
            8.0 * phi[idx3d(x - 1, y, z, Ny, Nz)] + phi[idx3d(x - 2, y, z, Ny, Nz)]) /
           (12.0 * dx);
}

/// 4th-order central difference gradient in Y.
__device__ __forceinline__ double gradient_y_4th(const double* __restrict__ phi, int x, int y,
                                                 int z, int Ny, int Nz, double dy) {
    return (-phi[idx3d(x, y + 2, z, Ny, Nz)] + 8.0 * phi[idx3d(x, y + 1, z, Ny, Nz)] -
            8.0 * phi[idx3d(x, y - 1, z, Ny, Nz)] + phi[idx3d(x, y - 2, z, Ny, Nz)]) /
           (12.0 * dy);
}

/// 4th-order central difference gradient in Z.
__device__ __forceinline__ double gradient_z_4th(const double* __restrict__ phi, int x, int y,
                                                 int z, int Ny, int Nz, double dz) {
    return (-phi[idx3d(x, y, z + 2, Ny, Nz)] + 8.0 * phi[idx3d(x, y, z + 1, Ny, Nz)] -
            8.0 * phi[idx3d(x, y, z - 1, Ny, Nz)] + phi[idx3d(x, y, z - 2, Ny, Nz)]) /
           (12.0 * dz);
}

/// Standard 7-point Laplacian (2nd order).
__device__ __forceinline__ double laplacian_7pt(const double* __restrict__ phi, int x, int y, int z,
                                                int Ny, int Nz, double dx, double dy, double dz) {
    int c = idx3d(x, y, z, Ny, Nz);
    double phixx =
        (phi[idx3d(x + 1, y, z, Ny, Nz)] + phi[idx3d(x - 1, y, z, Ny, Nz)] - 2.0 * phi[c]) /
        (dx * dx);
    double phiyy =
        (phi[idx3d(x, y + 1, z, Ny, Nz)] + phi[idx3d(x, y - 1, z, Ny, Nz)] - 2.0 * phi[c]) /
        (dy * dy);
    double phizz =
        (phi[idx3d(x, y, z + 1, Ny, Nz)] + phi[idx3d(x, y, z - 1, Ny, Nz)] - 2.0 * phi[c]) /
        (dz * dz);
    return phixx + phiyy + phizz;
}

/// Isotropic 27-point Laplacian (Patra-Karttunen, 2nd order with improved isotropy).
/// Assumes dx == dy == dz.
__device__ __forceinline__ double laplacian_27pt(const double* __restrict__ phi, int x, int y,
                                                 int z, int Ny, int Nz, double h) {
    double center = phi[idx3d(x, y, z, Ny, Nz)];

    // 6 face neighbors (weight 4)
    double face = phi[idx3d(x + 1, y, z, Ny, Nz)] + phi[idx3d(x - 1, y, z, Ny, Nz)] +
                  phi[idx3d(x, y + 1, z, Ny, Nz)] + phi[idx3d(x, y - 1, z, Ny, Nz)] +
                  phi[idx3d(x, y, z + 1, Ny, Nz)] + phi[idx3d(x, y, z - 1, Ny, Nz)];

    // 12 edge neighbors (weight 2)
    double edge = phi[idx3d(x + 1, y + 1, z, Ny, Nz)] + phi[idx3d(x + 1, y - 1, z, Ny, Nz)] +
                  phi[idx3d(x - 1, y + 1, z, Ny, Nz)] + phi[idx3d(x - 1, y - 1, z, Ny, Nz)] +
                  phi[idx3d(x + 1, y, z + 1, Ny, Nz)] + phi[idx3d(x + 1, y, z - 1, Ny, Nz)] +
                  phi[idx3d(x - 1, y, z + 1, Ny, Nz)] + phi[idx3d(x - 1, y, z - 1, Ny, Nz)] +
                  phi[idx3d(x, y + 1, z + 1, Ny, Nz)] + phi[idx3d(x, y + 1, z - 1, Ny, Nz)] +
                  phi[idx3d(x, y - 1, z + 1, Ny, Nz)] + phi[idx3d(x, y - 1, z - 1, Ny, Nz)];

    // 8 corner neighbors (weight 1)
    double corner =
        phi[idx3d(x + 1, y + 1, z + 1, Ny, Nz)] + phi[idx3d(x + 1, y + 1, z - 1, Ny, Nz)] +
        phi[idx3d(x + 1, y - 1, z + 1, Ny, Nz)] + phi[idx3d(x + 1, y - 1, z - 1, Ny, Nz)] +
        phi[idx3d(x - 1, y + 1, z + 1, Ny, Nz)] + phi[idx3d(x - 1, y + 1, z - 1, Ny, Nz)] +
        phi[idx3d(x - 1, y - 1, z + 1, Ny, Nz)] + phi[idx3d(x - 1, y - 1, z - 1, Ny, Nz)];

    // Patra-Karttunen weights (2nd-order accurate, improved isotropy):
    // face=14, edge=3, corner=1, center=-(6*14+12*3+8*1)=-128, divisor 30*h^2
    return (14.0 * face + 3.0 * edge + 1.0 * corner - 128.0 * center) / (30.0 * h * h);
}

/// Dispatch Laplacian based on stencil type.
__device__ __forceinline__ double laplacian(const double* __restrict__ phi, int x, int y, int z,
                                            const KernelParams& p) {
    if (p.stencil_type == 1) {
        return laplacian_27pt(phi, x, y, z, p.Ny, p.Nz, p.dx);
    }
    return laplacian_7pt(phi, x, y, z, p.Ny, p.Nz, p.dx, p.dy, p.dz);
}

// ── Anisotropy functions ───────────────────────────────────────────────────

/// Anisotropy function An(grad phi).
__device__ __forceinline__ double compute_An(double phix, double phiy, double phiz,
                                             double epsilon) {
    double sq = phix * phix + phiy * phiy + phiz * phiz;
    if (sq > 1e-30) {
        double qrt =
            phix * phix * phix * phix + phiy * phiy * phiy * phiy + phiz * phiz * phiz * phiz;
        return (1.0 - 3.0 * epsilon) *
               (1.0 + (4.0 * epsilon / (1.0 - 3.0 * epsilon)) * (qrt / (sq * sq)));
    }
    return 1.0 - 3.0 * epsilon / 5.0;
}

/// Derivative helper for anisotropic force.
__device__ __forceinline__ double dFunc(double l, double m, double n) {
    double sq = l * l + m * m + n * n;
    if (sq > 1e-30) {
        return (l * l * l * (m * m + n * n) - l * (m * m * m * m + n * n * n * n)) / (sq * sq);
    }
    return 0.0;
}

/// Free energy derivative dF/dphi.
__device__ __forceinline__ double dF_dphi(double phi, double u, double lambda) {
    double phi2 = phi * phi;
    double omp2 = 1.0 - phi2;
    return -phi * omp2 + lambda * u * omp2 * omp2;
}

/// Component d of the anisotropic flux F = df/dp of the gradient energy
/// f(p) = 1/2 W0^2 A(p)^2 |p|^2, with p = (pd, pa, pb) listed component d
/// first (A and dFunc are symmetric in the other two):
///   F_d = w^2 p_d + 16 eps W0 w dFunc(p_d, p_a, p_b),  w = W0 A(p).
/// (dA/dp_d = 16 eps dFunc_d / |p|^2, which cancels the |p|^2 of the chain rule.)
__device__ __forceinline__ double anisotropic_flux(double pd, double pa, double pb, double W0,
                                                   double epsilon) {
    const double w = W0 * compute_An(pd, pa, pb, epsilon);
    return w * w * pd + 16.0 * epsilon * W0 * w * dFunc(pd, pa, pb);
}

/// Cells the Allen-Cahn, thermal and Jacobi stencils read on each side of the
/// cell they update. MultiGPUSolver sizes its halos from this.
inline constexpr int kStencilReach = 1;

/// div F(grad phi) at interior cell (x, y, z) in conservative face-flux form:
///   sum_d [F_d(face d+1/2) - F_d(face d-1/2)] / h_d.
/// At a face the normal derivative is the compact difference of the two cells
/// it separates and each tangential derivative is the average of the central
/// differences in those two cells, so the stencil reaches +/-1 cell (3x3x3)
/// and at eps = 0 reduces exactly to W0^2 times the 7-point Laplacian.
__device__ __forceinline__ double anisotropic_divergence(const double* __restrict__ phi, int x,
                                                         int y, int z, const KernelParams& p) {
    const int Ny = p.Ny, Nz = p.Nz;
    auto at = [&](int i, int j, int k) { return phi[idx3d(i, j, k, Ny, Nz)]; };
    const double inv4dx = 0.25 / p.dx, inv4dy = 0.25 / p.dy, inv4dz = 0.25 / p.dz;

    // Face between (i, y, z) and (i + 1, y, z).
    auto flux_x = [&](int i) {
        const double gx = (at(i + 1, y, z) - at(i, y, z)) / p.dx;
        const double gy =
            (at(i, y + 1, z) - at(i, y - 1, z) + at(i + 1, y + 1, z) - at(i + 1, y - 1, z)) *
            inv4dy;
        const double gz =
            (at(i, y, z + 1) - at(i, y, z - 1) + at(i + 1, y, z + 1) - at(i + 1, y, z - 1)) *
            inv4dz;
        return anisotropic_flux(gx, gy, gz, p.W0, p.epsilon);
    };
    // Face between (x, j, z) and (x, j + 1, z).
    auto flux_y = [&](int j) {
        const double gy = (at(x, j + 1, z) - at(x, j, z)) / p.dy;
        const double gz =
            (at(x, j, z + 1) - at(x, j, z - 1) + at(x, j + 1, z + 1) - at(x, j + 1, z - 1)) *
            inv4dz;
        const double gx =
            (at(x + 1, j, z) - at(x - 1, j, z) + at(x + 1, j + 1, z) - at(x - 1, j + 1, z)) *
            inv4dx;
        return anisotropic_flux(gy, gz, gx, p.W0, p.epsilon);
    };
    // Face between (x, y, k) and (x, y, k + 1).
    auto flux_z = [&](int k) {
        const double gz = (at(x, y, k + 1) - at(x, y, k)) / p.dz;
        const double gx =
            (at(x + 1, y, k) - at(x - 1, y, k) + at(x + 1, y, k + 1) - at(x - 1, y, k + 1)) *
            inv4dx;
        const double gy =
            (at(x, y + 1, k) - at(x, y - 1, k) + at(x, y + 1, k + 1) - at(x, y - 1, k + 1)) *
            inv4dy;
        return anisotropic_flux(gz, gx, gy, p.W0, p.epsilon);
    };

    return (flux_x(x) - flux_x(x - 1)) / p.dx + (flux_y(y) - flux_y(y - 1)) / p.dy +
           (flux_z(z) - flux_z(z - 1)) / p.dz;
}

/// dphi/dt of the Allen-Cahn equation at interior cell (x, y, z):
///   tau0 A(n)^2 dphi/dt = div F(grad phi) - dF/dphi(phi, u),
/// with n from the central-difference gradient at the cell.
__device__ __forceinline__ double allen_cahn_rate(const double* __restrict__ phi,
                                                  const double* __restrict__ u, int x, int y, int z,
                                                  const KernelParams& p) {
    const double an = compute_An(gradient_x(phi, x, y, z, p.Ny, p.Nz, p.dx),
                                 gradient_y(phi, x, y, z, p.Ny, p.Nz, p.dy),
                                 gradient_z(phi, x, y, z, p.Ny, p.Nz, p.dz), p.epsilon);
    const int c = idx3d(x, y, z, p.Ny, p.Nz);
    return (anisotropic_divergence(phi, x, y, z, p) - dF_dphi(phi[c], u[c], p.lambda)) /
           (p.tau0 * an * an);
}

// ── Kernel declarations ────────────────────────────────────────────────────

/// Explicit Euler Allen-Cahn update: phi_new = phi_old + dt * allen_cahn_rate
/// at interior cells (boundary cells are left to the BC kernels).
void launch_allen_cahn_fused(const double* phi_old, double* phi_new, const double* u_old,
                             const KernelParams& params, cudaStream_t stream = nullptr);

/// Allen-Cahn rate for RK stages: rhs = allen_cahn_rate at interior cells, 0 on
/// boundary cells.
__global__ void allen_cahn_rhs_kernel(const double* __restrict__ phi, double* __restrict__ rhs,
                                      const double* __restrict__ u, KernelParams p);

/// Thermal diffusion equation kernel.
void launch_thermal_equation(const double* u_old, double* u_new, const double* phi_new,
                             const double* phi_old, const KernelParams& params,
                             cudaStream_t stream = nullptr);

/// Thermal RHS kernel (diffusion + latent heat coupling) for RK stages.
/// k_phi is the Allen-Cahn RHS for this stage: rhs = D*Lap(u) + 0.5*k_phi.
__global__ void thermal_rhs_kernel(const double* __restrict__ u, double* __restrict__ rhs,
                                   const double* __restrict__ k_phi, KernelParams p);

/// Boundary condition kernels (uniform BC on all faces).
void launch_boundary_conditions(double* field, const KernelParams& params, BCType bc_type,
                                double bc_value, double bc_flux, double bc_alpha, double bc_beta,
                                double bc_gamma, cudaStream_t stream = nullptr);

/// Boundary condition kernels (per-face BC specification).
void launch_boundary_conditions_per_face(double* field, const KernelParams& params,
                                         const PerFaceBoundary& face_bcs,
                                         cudaStream_t stream = nullptr);

/// Compute maximum absolute value of a single field via parallel reduction.
void launch_max_abs_reduction(const double* field, double* result, std::size_t N,
                              cudaStream_t stream = nullptr);

/// Compute max absolute difference |a - b| via parallel reduction.
void launch_max_abs_diff(const double* a, const double* b, double* d_result, std::size_t N,
                         cudaStream_t stream = nullptr);

/// Overloads with pre-allocated scratch buffer (avoids per-call allocation).
void launch_max_abs_reduction(const double* field, double* result, std::size_t N, double* scratch,
                              int scratch_size, cudaStream_t stream);
void launch_max_abs_diff(const double* a, const double* b, double* d_result, std::size_t N,
                         double* scratch, int scratch_size, cudaStream_t stream);

} // namespace ac::cuda
