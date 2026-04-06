#include "cuda/CudaUtils.cuh"
#include "cuda/Kernels.cuh"

namespace ac::cuda {

/// Thermal diffusion kernel: u^{n+1} = u^n + 0.5*(phi^{n+1} - phi^n) + dt*D*Laplacian(u^n)
__global__ void __launch_bounds__(256)
    thermal_equation_kernel(const double* __restrict__ u_old, double* __restrict__ u_new,
                            const double* __restrict__ phi_new, const double* __restrict__ phi_old,
                            KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.Nx * p.Ny * p.Nz;
    if (tid >= static_cast<unsigned>(total))
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        return;
    }

    int c = idx3d(x, y, z, p.Ny, p.Nz);

    double lap_u = laplacian(u_old, x, y, z, p);
    double latent_heat = 0.5 * (phi_new[c] - phi_old[c]);

    u_new[c] = u_old[c] + latent_heat + p.dt * p.D * lap_u;
}

/// Thermal RHS kernel (for higher-order time integration).
/// Computes rhs = 0.5*(dphi/dt) + D*Laplacian(u)
/// The latent heat coupling term is handled by the caller.
__global__ void __launch_bounds__(256)
    thermal_rhs_kernel(const double* __restrict__ u, double* __restrict__ rhs, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = p.Nx * p.Ny * p.Nz;
    if (tid >= static_cast<unsigned>(total))
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        rhs[c] = 0.0;
        return;
    }

    rhs[c] = p.D * laplacian(u, x, y, z, p);
}

// ── Launch wrapper ─────────────────────────────────────────────────────────

void launch_thermal_equation(const double* u_old, double* u_new, const double* phi_new,
                             const double* phi_old, const KernelParams& params,
                             cudaStream_t stream) {
    std::size_t total = static_cast<std::size_t>(params.Nx) * params.Ny * params.Nz;
    auto cfg = LaunchConfig::for_1d(total, 256);
    thermal_equation_kernel<<<cfg.grid, cfg.block, 0, stream>>>(u_old, u_new, phi_new, phi_old,
                                                                params);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
