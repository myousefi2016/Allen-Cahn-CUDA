#include "cuda/CudaUtils.cuh"
#include "cuda/Kernels.cuh"

namespace ac::cuda {

/// Explicit Euler step of the Allen-Cahn equation at interior cells. Boundary
/// cells are not written; the BC kernels set them afterwards.
__global__ void __launch_bounds__(256)
    allen_cahn_fused_kernel(const double* __restrict__ phi_old, double* __restrict__ phi_new,
                            const double* __restrict__ u_old, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1)
        return;

    const int c = idx3d(x, y, z, p.Ny, p.Nz);
    phi_new[c] = phi_old[c] + p.dt * allen_cahn_rate(phi_old, u_old, x, y, z, p);
}

/// Allen-Cahn rate for the RK stages: allen_cahn_rate at interior cells and 0
/// on boundary cells (their values come from the BCs, not from the PDE).
__global__ void __launch_bounds__(256)
    allen_cahn_rhs_kernel(const double* __restrict__ phi, double* __restrict__ rhs,
                          const double* __restrict__ u, KernelParams p) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    const int c = idx3d(x, y, z, p.Ny, p.Nz);
    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        rhs[c] = 0.0;
        return;
    }
    rhs[c] = allen_cahn_rate(phi, u, x, y, z, p);
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
