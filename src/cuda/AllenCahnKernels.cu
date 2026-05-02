#include "cuda/CudaUtils.cuh"
#include "cuda/Kernels.cuh"

namespace ac::cuda {

/// Fused kernel: computes anisotropic force AND updates phi in a single pass.
/// Eliminates Fx, Fy, Fz global memory arrays entirely.
/// Force divergence is computed by recomputing the force at neighboring stencil
/// points, which trades extra arithmetic for massive memory bandwidth savings.
__global__ void __launch_bounds__(256, 2)
    allen_cahn_fused_kernel(const double* __restrict__ phi_old, double* __restrict__ phi_new,
                            const double* __restrict__ u_old, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);

    // Only update interior points
    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        return;
    }

    // Compute gradients of phi at this point
    double phix = gradient_x(phi_old, x, y, z, p.Ny, p.Nz, p.dx);
    double phiy = gradient_y(phi_old, x, y, z, p.Ny, p.Nz, p.dy);
    double phiz = gradient_z(phi_old, x, y, z, p.Ny, p.Nz, p.dz);

    // Compute anisotropy ONCE (was computed 8+ times in original code)
    double an = compute_An(phix, phiy, phiz, p.epsilon);
    double wn = p.W0 * an;
    double wn2 = wn * wn;
    double tn = p.tau0 * an * an;

    // Anisotropy force coefficient: F_i = wn²·φ_i + 16·W₀·wn·ε·dFunc_i
    // Derived from δF/δ(∇φ) of gradient energy ½W₀²A²|∇φ|².
    // ∂A/∂φ_i = 16ε·dFunc_i/|∇φ|², which cancels the |∇φ|² from the chain rule.
    double coeff = wn * 16.0 * p.W0 * p.epsilon;
    double Fx_here = wn2 * phix + coeff * dFunc(phix, phiy, phiz);
    double Fy_here = wn2 * phiy + coeff * dFunc(phiy, phiz, phix);
    double Fz_here = wn2 * phiz + coeff * dFunc(phiz, phix, phiy);

    // We need the divergence of F = dFx/dx + dFy/dy + dFz/dz.
    // Compute forces at neighboring points for numerical divergence.
    // This is more compute but avoids writing/reading 3 global arrays.

    // Helper lambda to compute force component at an offset point.
    // At boundary cells (x=0/Nx-1 etc.) the gradient uses the actual
    // BC-enforced field values via clamped indexing, avoiding the old
    // "return 0" that created an asymmetric force bias at x=1 cells.
    auto safe_gradient = [&](const double* f, int gx, int gy, int gz, int comp) -> double {
        int xm = max(gx - 1, 0), xp = min(gx + 1, p.Nx - 1);
        int ym = max(gy - 1, 0), yp = min(gy + 1, p.Ny - 1);
        int zm = max(gz - 1, 0), zp = min(gz + 1, p.Nz - 1);
        if (comp == 0)
            return (f[idx3d(xp, gy, gz, p.Ny, p.Nz)] - f[idx3d(xm, gy, gz, p.Ny, p.Nz)]) /
                   ((xp - xm) * p.dx);
        if (comp == 1)
            return (f[idx3d(gx, yp, gz, p.Ny, p.Nz)] - f[idx3d(gx, ym, gz, p.Ny, p.Nz)]) /
                   ((yp - ym) * p.dy);
        return (f[idx3d(gx, gy, zp, p.Ny, p.Nz)] - f[idx3d(gx, gy, zm, p.Ny, p.Nz)]) /
               ((zp - zm) * p.dz);
    };

    auto compute_force_at = [&](int ox, int oy, int oz, int component) -> double {
        int nx = x + ox, ny = y + oy, nz = z + oz;
        nx = max(0, min(nx, p.Nx - 1));
        ny = max(0, min(ny, p.Ny - 1));
        nz = max(0, min(nz, p.Nz - 1));

        double px = safe_gradient(phi_old, nx, ny, nz, 0);
        double py = safe_gradient(phi_old, nx, ny, nz, 1);
        double pz = safe_gradient(phi_old, nx, ny, nz, 2);
        double a = compute_An(px, py, pz, p.epsilon);
        double w = p.W0 * a;
        double w2 = w * w;
        double cf = w * 16.0 * p.W0 * p.epsilon;

        if (component == 0)
            return w2 * px + cf * dFunc(px, py, pz);
        if (component == 1)
            return w2 * py + cf * dFunc(py, pz, px);
        return w2 * pz + cf * dFunc(pz, px, py);
    };

    // Divergence via central differences of force field
    double dFx_dx = (compute_force_at(1, 0, 0, 0) - compute_force_at(-1, 0, 0, 0)) / (2.0 * p.dx);
    double dFy_dy = (compute_force_at(0, 1, 0, 1) - compute_force_at(0, -1, 0, 1)) / (2.0 * p.dy);
    double dFz_dz = (compute_force_at(0, 0, 1, 2) - compute_force_at(0, 0, -1, 2)) / (2.0 * p.dz);
    double div_F = dFx_dx + dFy_dy + dFz_dz;

    // Allen-Cahn update
    int c = idx3d(x, y, z, p.Ny, p.Nz);
    double phi_c = phi_old[c];
    double u_c = u_old[c];
    phi_new[c] = phi_c + (p.dt / tn) * (div_F - dF_dphi(phi_c, u_c, p.lambda));
}

/// Lightweight Allen-Cahn kernel using separate Fx, Fy, Fz arrays (for RK stages).
/// This kernel reads pre-computed forces and computes the RHS of the Allen-Cahn eq.
__global__ void __launch_bounds__(256)
    allen_cahn_rhs_kernel(const double* __restrict__ phi_old, double* __restrict__ rhs,
                          const double* __restrict__ u_old, const double* __restrict__ Fx,
                          const double* __restrict__ Fy, const double* __restrict__ Fz,
                          KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);

    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        rhs[c] = 0.0;
        return;
    }

    double phix = gradient_x(phi_old, x, y, z, p.Ny, p.Nz, p.dx);
    double phiy = gradient_y(phi_old, x, y, z, p.Ny, p.Nz, p.dy);
    double phiz = gradient_z(phi_old, x, y, z, p.Ny, p.Nz, p.dz);

    double an = compute_An(phix, phiy, phiz, p.epsilon);
    double tn = p.tau0 * an * an;

    // Divergence of F
    double div_Fx = gradient_x(Fx, x, y, z, p.Ny, p.Nz, p.dx);
    double div_Fy = gradient_y(Fy, x, y, z, p.Ny, p.Nz, p.dy);
    double div_Fz = gradient_z(Fz, x, y, z, p.Ny, p.Nz, p.dz);
    double div_F = div_Fx + div_Fy + div_Fz;

    rhs[c] = (1.0 / tn) * (div_F - dF_dphi(phi_old[c], u_old[c], p.lambda));
}

/// Compute anisotropic force field (Fx, Fy, Fz) from phi.
__global__ void __launch_bounds__(256)
    compute_force_kernel(const double* __restrict__ phi, double* __restrict__ Fx,
                         double* __restrict__ Fy, double* __restrict__ Fz, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        Fx[c] = 0.0;
        Fy[c] = 0.0;
        Fz[c] = 0.0;
        return;
    }

    double phix = gradient_x(phi, x, y, z, p.Ny, p.Nz, p.dx);
    double phiy = gradient_y(phi, x, y, z, p.Ny, p.Nz, p.dy);
    double phiz = gradient_z(phi, x, y, z, p.Ny, p.Nz, p.dz);

    double an = compute_An(phix, phiy, phiz, p.epsilon);
    double wn = p.W0 * an;
    double wn2 = wn * wn;
    double coeff = wn * 16.0 * p.W0 * p.epsilon;

    Fx[c] = wn2 * phix + coeff * dFunc(phix, phiy, phiz);
    Fy[c] = wn2 * phiy + coeff * dFunc(phiy, phiz, phix);
    Fz[c] = wn2 * phiz + coeff * dFunc(phiz, phix, phiy);
}

// ── Launch wrappers ────────────────────────────────────────────────────────

void launch_allen_cahn_fused(const double* phi_old, double* phi_new, const double* u_old,
                             const KernelParams& params, cudaStream_t stream) {
    std::size_t total = static_cast<std::size_t>(params.Nx) * params.Ny * params.Nz;
    auto cfg = LaunchConfig::for_1d(total, 256);
    allen_cahn_fused_kernel<<<cfg.grid, cfg.block, 0, stream>>>(phi_old, phi_new, u_old, params);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
