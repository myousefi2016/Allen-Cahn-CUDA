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
    int stencil_type;  // 0 = 7pt, 1 = 27pt

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

__device__ __forceinline__
int idx3d(int x, int y, int z, int Ny, int Nz) {
    return x * Ny * Nz + y * Nz + z;
}

__device__ __forceinline__
void linear_to_3d(int idx, int Ny, int Nz, int& x, int& y, int& z) {
    z = idx % Nz;
    y = (idx / Nz) % Ny;
    x = idx / (Ny * Nz);
}

// ── Stencil functions ──────────────────────────────────────────────────────

/// 2nd-order central difference gradient in X.
__device__ __forceinline__
double gradient_x(const double* __restrict__ phi, int x, int y, int z,
                  int Ny, int Nz, double dx) {
    return (phi[idx3d(x+1,y,z,Ny,Nz)] - phi[idx3d(x-1,y,z,Ny,Nz)]) / (2.0 * dx);
}

/// 2nd-order central difference gradient in Y.
__device__ __forceinline__
double gradient_y(const double* __restrict__ phi, int x, int y, int z,
                  int Ny, int Nz, double dy) {
    return (phi[idx3d(x,y+1,z,Ny,Nz)] - phi[idx3d(x,y-1,z,Ny,Nz)]) / (2.0 * dy);
}

/// 2nd-order central difference gradient in Z.
__device__ __forceinline__
double gradient_z(const double* __restrict__ phi, int x, int y, int z,
                  int Ny, int Nz, double dz) {
    return (phi[idx3d(x,y,z+1,Ny,Nz)] - phi[idx3d(x,y,z-1,Ny,Nz)]) / (2.0 * dz);
}

/// 4th-order central difference gradient in X.
__device__ __forceinline__
double gradient_x_4th(const double* __restrict__ phi, int x, int y, int z,
                      int Ny, int Nz, double dx) {
    return (-phi[idx3d(x+2,y,z,Ny,Nz)] + 8.0*phi[idx3d(x+1,y,z,Ny,Nz)]
            -8.0*phi[idx3d(x-1,y,z,Ny,Nz)] + phi[idx3d(x-2,y,z,Ny,Nz)])
           / (12.0 * dx);
}

/// 4th-order central difference gradient in Y.
__device__ __forceinline__
double gradient_y_4th(const double* __restrict__ phi, int x, int y, int z,
                      int Ny, int Nz, double dy) {
    return (-phi[idx3d(x,y+2,z,Ny,Nz)] + 8.0*phi[idx3d(x,y+1,z,Ny,Nz)]
            -8.0*phi[idx3d(x,y-1,z,Ny,Nz)] + phi[idx3d(x,y-2,z,Ny,Nz)])
           / (12.0 * dy);
}

/// 4th-order central difference gradient in Z.
__device__ __forceinline__
double gradient_z_4th(const double* __restrict__ phi, int x, int y, int z,
                      int Ny, int Nz, double dz) {
    return (-phi[idx3d(x,y,z+2,Ny,Nz)] + 8.0*phi[idx3d(x,y,z+1,Ny,Nz)]
            -8.0*phi[idx3d(x,y,z-1,Ny,Nz)] + phi[idx3d(x,y,z-2,Ny,Nz)])
           / (12.0 * dz);
}

/// Standard 7-point Laplacian (2nd order).
__device__ __forceinline__
double laplacian_7pt(const double* __restrict__ phi, int x, int y, int z,
                     int Ny, int Nz, double dx, double dy, double dz) {
    int c = idx3d(x, y, z, Ny, Nz);
    double phixx = (phi[idx3d(x+1,y,z,Ny,Nz)] + phi[idx3d(x-1,y,z,Ny,Nz)] - 2.0*phi[c]) / (dx*dx);
    double phiyy = (phi[idx3d(x,y+1,z,Ny,Nz)] + phi[idx3d(x,y-1,z,Ny,Nz)] - 2.0*phi[c]) / (dy*dy);
    double phizz = (phi[idx3d(x,y,z+1,Ny,Nz)] + phi[idx3d(x,y,z-1,Ny,Nz)] - 2.0*phi[c]) / (dz*dz);
    return phixx + phiyy + phizz;
}

/// Isotropic 27-point Laplacian (Kumar 2004, 4th order isotropic).
/// Assumes dx == dy == dz.
__device__ __forceinline__
double laplacian_27pt(const double* __restrict__ phi, int x, int y, int z,
                      int Ny, int Nz, double h) {
    double center = phi[idx3d(x,y,z,Ny,Nz)];

    // 6 face neighbors (weight 4)
    double face = phi[idx3d(x+1,y,z,Ny,Nz)] + phi[idx3d(x-1,y,z,Ny,Nz)]
                + phi[idx3d(x,y+1,z,Ny,Nz)] + phi[idx3d(x,y-1,z,Ny,Nz)]
                + phi[idx3d(x,y,z+1,Ny,Nz)] + phi[idx3d(x,y,z-1,Ny,Nz)];

    // 12 edge neighbors (weight 2)
    double edge = phi[idx3d(x+1,y+1,z,Ny,Nz)] + phi[idx3d(x+1,y-1,z,Ny,Nz)]
                + phi[idx3d(x-1,y+1,z,Ny,Nz)] + phi[idx3d(x-1,y-1,z,Ny,Nz)]
                + phi[idx3d(x+1,y,z+1,Ny,Nz)] + phi[idx3d(x+1,y,z-1,Ny,Nz)]
                + phi[idx3d(x-1,y,z+1,Ny,Nz)] + phi[idx3d(x-1,y,z-1,Ny,Nz)]
                + phi[idx3d(x,y+1,z+1,Ny,Nz)] + phi[idx3d(x,y+1,z-1,Ny,Nz)]
                + phi[idx3d(x,y-1,z+1,Ny,Nz)] + phi[idx3d(x,y-1,z-1,Ny,Nz)];

    // 8 corner neighbors (weight 1)
    double corner = phi[idx3d(x+1,y+1,z+1,Ny,Nz)] + phi[idx3d(x+1,y+1,z-1,Ny,Nz)]
                  + phi[idx3d(x+1,y-1,z+1,Ny,Nz)] + phi[idx3d(x+1,y-1,z-1,Ny,Nz)]
                  + phi[idx3d(x-1,y+1,z+1,Ny,Nz)] + phi[idx3d(x-1,y+1,z-1,Ny,Nz)]
                  + phi[idx3d(x-1,y-1,z+1,Ny,Nz)] + phi[idx3d(x-1,y-1,z-1,Ny,Nz)];

    // Weights: face=4, edge=2, corner=1, center=-(6*4+12*2+8*1)=-56
    // Total weight sum = 6*4+12*2+8*1 = 56
    return (4.0*face + 2.0*edge + 1.0*corner - 56.0*center) / (26.0 * h * h);
}

/// Dispatch Laplacian based on stencil type.
__device__ __forceinline__
double laplacian(const double* __restrict__ phi, int x, int y, int z,
                 const KernelParams& p) {
    if (p.stencil_type == 1) {
        return laplacian_27pt(phi, x, y, z, p.Ny, p.Nz, p.dx);
    }
    return laplacian_7pt(phi, x, y, z, p.Ny, p.Nz, p.dx, p.dy, p.dz);
}

// ── Anisotropy functions ───────────────────────────────────────────────────

/// Anisotropy function An(grad phi).
__device__ __forceinline__
double compute_An(double phix, double phiy, double phiz, double epsilon) {
    double sq = phix*phix + phiy*phiy + phiz*phiz;
    if (sq > 1e-30) {
        double qrt = phix*phix*phix*phix + phiy*phiy*phiy*phiy + phiz*phiz*phiz*phiz;
        return (1.0 - 3.0*epsilon) * (1.0 + (4.0*epsilon/(1.0-3.0*epsilon)) * (qrt/(sq*sq)));
    }
    return 1.0 - (5.0/3.0)*epsilon;
}

/// Derivative helper for anisotropic force.
__device__ __forceinline__
double dFunc(double l, double m, double n) {
    double sq = l*l + m*m + n*n;
    if (sq > 1e-30) {
        return (l*l*l*(m*m+n*n) - l*(m*m*m*m+n*n*n*n)) / (sq*sq);
    }
    return 0.0;
}

/// Free energy derivative dF/dphi.
__device__ __forceinline__
double dF_dphi(double phi, double u, double lambda) {
    double phi2 = phi * phi;
    double omp2 = 1.0 - phi2;
    return -phi * omp2 + lambda * u * omp2 * omp2;
}

// ── Kernel declarations ────────────────────────────────────────────────────

/// Fused Allen-Cahn kernel: computes force, divergence, and updates phi.
void launch_allen_cahn_fused(
    const double* phi_old, double* phi_new, const double* u_old,
    const KernelParams& params, cudaStream_t stream = nullptr);

/// Thermal diffusion equation kernel.
void launch_thermal_equation(
    const double* u_old, double* u_new,
    const double* phi_new, const double* phi_old,
    const KernelParams& params, cudaStream_t stream = nullptr);

/// Boundary condition kernels (uniform BC on all faces).
void launch_boundary_conditions(
    double* field, const KernelParams& params,
    BCType bc_type, double bc_value, double bc_flux,
    double bc_alpha, double bc_beta, double bc_gamma,
    cudaStream_t stream = nullptr);

/// Boundary condition kernels (per-face BC specification).
void launch_boundary_conditions_per_face(
    double* field, const KernelParams& params,
    const PerFaceBoundary& face_bcs,
    cudaStream_t stream = nullptr);

/// Compute maximum absolute value via parallel reduction (for adaptive dt).
void launch_max_abs_reduction(
    const double* field_a, const double* field_b,
    double* result, std::size_t N,
    cudaStream_t stream = nullptr);

/// Compute max absolute difference |a - b| via parallel reduction.
void launch_max_abs_diff(
    const double* a, const double* b,
    double* d_result, std::size_t N,
    cudaStream_t stream = nullptr);

} // namespace ac::cuda
