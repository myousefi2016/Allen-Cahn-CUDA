#include "cuda/CudaUtils.cuh"
#include "cuda/WillmoreKernels.cuh"
#include "cuda/WillmoreSolve.cuh"

#include <algorithm>

namespace ac::cuda {

namespace {

/// Decode a 1-D thread id into an interior cell; false for boundary cells and
/// threads past the end.
__device__ __forceinline__ bool interior_cell(const KernelParams& p, int& x, int& y, int& z,
                                              int& c) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >=
        static_cast<unsigned>(p.Nx) * static_cast<unsigned>(p.Ny) * static_cast<unsigned>(p.Nz))
        return false;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);
    c = static_cast<int>(tid);
    return x >= 1 && x < p.Nx - 1 && y >= 1 && y < p.Ny - 1 && z >= 1 && z < p.Nz - 1;
}

LaunchConfig cells(const KernelParams& p) {
    return LaunchConfig::for_1d(static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz, 256);
}

/// 1/h^2 per axis and 1/dt, rounded once on the host: a double-precision
/// division costs as much as 8 other double-precision instructions on GPUs
/// with a reduced double-precision rate, and these stencils are bound by
/// double-precision throughput there (docs/THEORY.md section 4.4).
struct Recip64 {
    double inv_hx2, inv_hy2, inv_hz2, inv_dt;
};

Recip64 recip64(const KernelParams& p) {
    return {1.0 / (p.dx * p.dx), 1.0 / (p.dy * p.dy), 1.0 / (p.dz * p.dz), 1.0 / p.dt};
}

/// 7-point Laplacian at interior cell c (the Willmore operator's stencil).
__device__ __forceinline__ double lap7(const double* __restrict__ v, int c, int sx, int sy,
                                       const Recip64& rc) {
    const double two_vc = 2.0 * v[c];
    return (v[c + sx] + v[c - sx] - two_vc) * rc.inv_hx2 +
           (v[c + sy] + v[c - sy] - two_vc) * rc.inv_hy2 +
           (v[c + 1] + v[c - 1] - two_vc) * rc.inv_hz2;
}

__global__ void __launch_bounds__(256)
    willmore_omega_kernel(const double* __restrict__ phi, double* __restrict__ omega,
                          KernelParams p, Recip64 rc) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    const double f = phi[c];
    omega[c] = p.W0 * p.W0 * lap7(phi, c, p.Ny * p.Nz, p.Nz, rc) + f - f * f * f;
}

__global__ void __launch_bounds__(256)
    willmore_rhs_init_kernel(const double* __restrict__ phi, const double* __restrict__ u,
                             const double* __restrict__ omega, double* __restrict__ tau,
                             double* __restrict__ xv, double* __restrict__ r,
                             double* __restrict__ zv, double* __restrict__ dir, KernelParams p,
                             Recip64 rc) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >=
        static_cast<unsigned>(p.Nx) * static_cast<unsigned>(p.Ny) * static_cast<unsigned>(p.Nz))
        return;
    xv[tid] = zv[tid] = dir[tid] = 0.0;
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c)) {
        r[tid] = tau[tid] = 0.0;
        return;
    }
    const double W2 = p.W0 * p.W0;
    const double f = phi[c], w = omega[c];
    const double M = W2 * lap7(omega, c, p.Ny * p.Nz, p.Nz, rc) + (1.0 - 3.0 * f * f) * w;
    tau[c] = allen_cahn_tau(phi, x, y, z, p);
    r[c] = allen_cahn_force(phi, u, x, y, z, p) - p.willmore_beta * M;
}

__global__ void __launch_bounds__(256)
    laplacian_interior_kernel(const double* __restrict__ src, double* __restrict__ dst,
                              KernelParams p, Recip64 rc) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    dst[c] = lap7(src, c, p.Ny * p.Nz, p.Nz, rc);
}

__global__ void __launch_bounds__(256)
    willmore_apply_kernel(const double* __restrict__ dir, const double* __restrict__ t,
                          const double* __restrict__ tau, double* __restrict__ Ap, KernelParams p,
                          Recip64 rc) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    const double W2 = p.W0 * p.W0;
    const double lap_t = lap7(t, c, p.Ny * p.Nz, p.Nz, rc);
    Ap[c] = tau[c] * rc.inv_dt * dir[c] +
            p.willmore_beta * (W2 * W2 * lap_t - 4.0 * W2 * t[c] + 4.0 * dir[c]);
}

__global__ void __launch_bounds__(256)
    cg_update_xr_kernel(double* __restrict__ xv, double* __restrict__ r,
                        const double* __restrict__ dir, const double* __restrict__ Ap, double alpha,
                        KernelParams p) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    xv[c] += alpha * dir[c];
    r[c] -= alpha * Ap[c];
}

__global__ void __launch_bounds__(256)
    cg_direction_kernel(const double* __restrict__ zv, double* __restrict__ dir, double beta,
                        KernelParams p) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    dir[c] = zv[c] + beta * dir[c];
}

__global__ void __launch_bounds__(256)
    willmore_finish_kernel(const double* __restrict__ phi_old, const double* __restrict__ xv,
                           double* __restrict__ phi_new, KernelParams p) {
    int x, y, z, c;
    if (!interior_cell(p, x, y, z, c))
        return;
    phi_new[c] = phi_old[c] + xv[c];
}

constexpr int kDotThreads = 256;

/// One block per X plane: fixed-order strided partial sums, then a fixed
/// shared-memory tree.
__global__ void __launch_bounds__(kDotThreads)
    plane_dot_kernel(const double* __restrict__ a, const double* __restrict__ b,
                     double* __restrict__ out, int x_begin, int Ny, int Nz) {
    __shared__ double s[kDotThreads];
    const int x = x_begin + static_cast<int>(blockIdx.x);
    const int ny = Ny - 2, nz = Nz - 2;
    const int n = ny * nz;
    double sum = 0.0;
    for (int k = static_cast<int>(threadIdx.x); k < n; k += kDotThreads) {
        const int y = 1 + k / nz, z = 1 + k % nz;
        const int c = idx3d(x, y, z, Ny, Nz);
        sum += a[c] * b[c];
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kDotThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = s[0];
}

} // namespace

void launch_willmore_omega(const double* phi, double* omega, const KernelParams& p,
                           cudaStream_t stream) {
    auto cfg = cells(p);
    willmore_omega_kernel<<<cfg.grid, cfg.block, 0, stream>>>(phi, omega, p, recip64(p));
    CUDA_CHECK(cudaGetLastError());
}

void launch_willmore_rhs_init(const double* phi, const double* u, const double* omega, double* tau,
                              double* x, double* r, double* z, double* dir, const KernelParams& p,
                              cudaStream_t stream) {
    auto cfg = cells(p);
    willmore_rhs_init_kernel<<<cfg.grid, cfg.block, 0, stream>>>(phi, u, omega, tau, x, r, z, dir,
                                                                 p, recip64(p));
    CUDA_CHECK(cudaGetLastError());
}

void launch_laplacian_interior(const double* src, double* dst, const KernelParams& p,
                               cudaStream_t stream) {
    auto cfg = cells(p);
    laplacian_interior_kernel<<<cfg.grid, cfg.block, 0, stream>>>(src, dst, p, recip64(p));
    CUDA_CHECK(cudaGetLastError());
}

void launch_willmore_apply(const double* dir, const double* t, const double* tau, double* Ap,
                           const KernelParams& p, cudaStream_t stream) {
    auto cfg = cells(p);
    willmore_apply_kernel<<<cfg.grid, cfg.block, 0, stream>>>(dir, t, tau, Ap, p, recip64(p));
    CUDA_CHECK(cudaGetLastError());
}

void launch_cg_update_xr(double* x, double* r, const double* dir, const double* Ap, double alpha,
                         const KernelParams& p, cudaStream_t stream) {
    auto cfg = cells(p);
    cg_update_xr_kernel<<<cfg.grid, cfg.block, 0, stream>>>(x, r, dir, Ap, alpha, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cg_direction(const double* z, double* dir, double beta, const KernelParams& p,
                         cudaStream_t stream) {
    auto cfg = cells(p);
    cg_direction_kernel<<<cfg.grid, cfg.block, 0, stream>>>(z, dir, beta, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_willmore_finish(const double* phi_old, const double* x, double* phi_new,
                            const KernelParams& p, cudaStream_t stream) {
    auto cfg = cells(p);
    willmore_finish_kernel<<<cfg.grid, cfg.block, 0, stream>>>(phi_old, x, phi_new, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_plane_dot(const double* a, const double* b, double* out, int x_begin, int x_end,
                      const KernelParams& p, cudaStream_t stream) {
    if (x_end <= x_begin)
        return;
    plane_dot_kernel<<<static_cast<unsigned>(x_end - x_begin), kDotThreads, 0, stream>>>(
        a, b, out, x_begin, p.Ny, p.Nz);
    CUDA_CHECK(cudaGetLastError());
}

// ── Single-precision Chebyshev preconditioner ──────────────────────────────

namespace {

// Interior cells are covered by a (z, y) thread grid and a grid-stride loop
// over the X planes, so no index division is needed.
constexpr int kPcBlockZ = 32;
constexpr int kPcBlockY = 8;
constexpr int kPcThreads = kPcBlockZ * kPcBlockY;
// Largest gridDim.y / gridDim.z of a launch.
constexpr unsigned kMaxGridYZ = 65535;

struct PcLaunch {
    dim3 grid, block;
};

PcLaunch pc_launch(const KernelParams& p) {
    const auto nz = static_cast<unsigned>(p.Nz - 2), ny = static_cast<unsigned>(p.Ny - 2),
               nx = static_cast<unsigned>(p.Nx - 2);
    PcLaunch l;
    l.block = dim3(kPcBlockZ, kPcBlockY, 1);
    l.grid = dim3((nz + kPcBlockZ - 1) / kPcBlockZ, (ny + kPcBlockY - 1) / kPcBlockY,
                  std::min(nx, kMaxGridYZ));
    return l;
}

__device__ __forceinline__ float lap32(const float* __restrict__ v, int c, int sx, int sy,
                                       const Operator32& op) {
    const float two_vc = 2.0f * v[c];
    return (v[c + sx] + v[c - sx] - two_vc) * op.inv_hx2 +
           (v[c + sy] + v[c - sy] - two_vc) * op.inv_hy2 +
           (v[c + 1] + v[c - 1] - two_vc) * op.inv_hz2;
}

/// Ghost of face `f` next to an interior cell holding v (non-periodic rule).
__device__ __forceinline__ float ghost_value(const GhostRules32& g, int f, float v) {
    return g.kind[f] == kGhostZero ? 0.0f : g.coef[f] * v;
}

/// Write the face ghosts that depend on interior cell (x, y, z) with value v:
/// the adjacent ghost of a non-periodic face, or, on a periodic axis, the
/// ghost at the opposite face (ghost at 0 = cell N-2, ghost at N-1 = cell 1).
/// Every face ghost has exactly one writer.
__device__ __forceinline__ void write_face_ghosts(float* __restrict__ f, int x, int y, int z,
                                                  float v, const GhostRules32& g,
                                                  const KernelParams& p) {
    const int sx = p.Ny * p.Nz, sy = p.Nz;
    const int c = idx3d(x, y, z, p.Ny, p.Nz);
    if (x == 1) {
        if (g.kind[0] != kGhostPeriodic)
            f[c - sx] = ghost_value(g, 0, v);
        else
            f[idx3d(p.Nx - 1, y, z, p.Ny, p.Nz)] = v;
    }
    if (x == p.Nx - 2) {
        if (g.kind[1] != kGhostPeriodic)
            f[c + sx] = ghost_value(g, 1, v);
        else
            f[idx3d(0, y, z, p.Ny, p.Nz)] = v;
    }
    if (y == 1) {
        if (g.kind[2] != kGhostPeriodic)
            f[c - sy] = ghost_value(g, 2, v);
        else
            f[idx3d(x, p.Ny - 1, z, p.Ny, p.Nz)] = v;
    }
    if (y == p.Ny - 2) {
        if (g.kind[3] != kGhostPeriodic)
            f[c + sy] = ghost_value(g, 3, v);
        else
            f[idx3d(x, 0, z, p.Ny, p.Nz)] = v;
    }
    if (z == 1) {
        if (g.kind[4] != kGhostPeriodic)
            f[c - 1] = ghost_value(g, 4, v);
        else
            f[idx3d(x, y, p.Nz - 1, p.Ny, p.Nz)] = v;
    }
    if (z == p.Nz - 2) {
        if (g.kind[5] != kGhostPeriodic)
            f[c + 1] = ghost_value(g, 5, v);
        else
            f[idx3d(x, y, 0, p.Ny, p.Nz)] = v;
    }
}

__global__ void __launch_bounds__(kPcThreads)
    precond_tau_scan_kernel(const double* __restrict__ tau, float* __restrict__ tdt, double inv_dt,
                            unsigned long long* __restrict__ range_bits, KernelParams p) {
    unsigned long long lo = ~0ULL, hi = 0ULL;
    const int z = 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (z <= p.Nz - 2 && y <= p.Ny - 2) {
        for (int x = 1 + static_cast<int>(blockIdx.z); x <= p.Nx - 2;
             x += static_cast<int>(gridDim.z)) {
            const int c = idx3d(x, y, z, p.Ny, p.Nz);
            const double t = tau[c];
            tdt[c] = static_cast<float>(t * inv_dt);
            const auto bits = static_cast<unsigned long long>(__double_as_longlong(t));
            lo = bits < lo ? bits : lo;
            hi = bits > hi ? bits : hi;
        }
    }
    __shared__ unsigned long long s_lo[kPcThreads], s_hi[kPcThreads];
    const int tid = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    s_lo[tid] = lo;
    s_hi[tid] = hi;
    __syncthreads();
    for (int stride = kPcThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_lo[tid] = s_lo[tid + stride] < s_lo[tid] ? s_lo[tid + stride] : s_lo[tid];
            s_hi[tid] = s_hi[tid + stride] > s_hi[tid] ? s_hi[tid + stride] : s_hi[tid];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicMin(&range_bits[0], s_lo[0]);
        atomicMax(&range_bits[1], s_hi[0]);
    }
}

__global__ void __launch_bounds__(kPcThreads)
    precond_first_kernel(const double* __restrict__ r, double scale, float* __restrict__ f,
                         float* __restrict__ x1, const float* __restrict__ tdt, Operator32 op,
                         float s0, GhostRules32 g, double* __restrict__ z_final, double inv_scale,
                         KernelParams p) {
    const int z = 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (z > p.Nz - 2 || y > p.Ny - 2)
        return;
    for (int x = 1 + static_cast<int>(blockIdx.z); x <= p.Nx - 2;
         x += static_cast<int>(gridDim.z)) {
        const int c = idx3d(x, y, z, p.Ny, p.Nz);
        const float fc = static_cast<float>(scale * r[c]);
        const float xc = s0 * fc / (tdt[c] + op.beta_dL);
        if (z_final != nullptr) {
            z_final[c] = static_cast<double>(xc) * inv_scale;
            continue;
        }
        f[c] = fc;
        x1[c] = xc;
        write_face_ghosts(x1, x, y, z, xc, g, p);
    }
}

__global__ void __launch_bounds__(kPcThreads)
    precond_laplacian_kernel(const float* __restrict__ xv, float* __restrict__ t, Operator32 op,
                             GhostRules32 g, KernelParams p) {
    const int z = 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (z > p.Nz - 2 || y > p.Ny - 2)
        return;
    const int sx = p.Ny * p.Nz, sy = p.Nz;
    for (int x = 1 + static_cast<int>(blockIdx.z); x <= p.Nx - 2;
         x += static_cast<int>(gridDim.z)) {
        const int c = idx3d(x, y, z, p.Ny, p.Nz);
        const float tc = lap32(xv, c, sx, sy, op);
        t[c] = tc;
        write_face_ghosts(t, x, y, z, tc, g, p);
    }
}

__global__ void __launch_bounds__(kPcThreads)
    precond_step_kernel(const float* __restrict__ t, const float* __restrict__ xv,
                        float* __restrict__ x_prev_next, bool prev_is_zero,
                        const float* __restrict__ f, const float* __restrict__ tdt, Operator32 op,
                        float a, float b, GhostRules32 g, double* __restrict__ z_final,
                        double inv_scale, KernelParams p) {
    const int z = 1 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 1 + static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (z > p.Nz - 2 || y > p.Ny - 2)
        return;
    const int sx = p.Ny * p.Nz, sy = p.Nz;
    for (int x = 1 + static_cast<int>(blockIdx.z); x <= p.Nx - 2;
         x += static_cast<int>(gridDim.z)) {
        const int c = idx3d(x, y, z, p.Ny, p.Nz);
        const float xc = xv[c], tc = t[c], tdtc = tdt[c];
        const float Ax =
            tdtc * xc + op.beta * (op.W4 * lap32(t, c, sx, sy, op) - op.four_W2 * tc + 4.0f * xc);
        const float xp = prev_is_zero ? 0.0f : x_prev_next[c];
        const float xn = xc + a * (xc - xp) + b * (f[c] - Ax) / (tdtc + op.beta_dL);
        if (z_final != nullptr) {
            z_final[c] = static_cast<double>(xn) * inv_scale;
            continue;
        }
        x_prev_next[c] = xn;
        write_face_ghosts(x_prev_next, x, y, z, xn, g, p);
    }
}

} // namespace

GhostRules32 make_ghost_rules32(const PerFaceBoundary& faces, const KernelParams& p) {
    const double spacing[3] = {p.dx, p.dy, p.dz};
    GhostRules32 g{};
    for (int f = 0; f < 6; ++f) {
        const BoundaryConfig& bc = faces.faces[static_cast<std::size_t>(f)];
        switch (bc.type) {
        case BCType::Dirichlet:
            g.kind[f] = kGhostZero;
            g.coef[f] = 0.0f;
            break;
        case BCType::Neumann:
            g.kind[f] = kGhostScaled;
            g.coef[f] = 1.0f;
            break;
        case BCType::Robin:
            g.kind[f] = kGhostScaled;
            g.coef[f] =
                static_cast<float>(robin_ghost_coefficient(bc.alpha, bc.beta, spacing[f / 2]));
            break;
        case BCType::Periodic:
            g.kind[f] = kGhostPeriodic;
            g.coef[f] = 0.0f;
            break;
        }
    }
    return g;
}

Operator32 make_operator32(const KernelParams& p, double d_L) {
    const double W2 = p.W0 * p.W0;
    Operator32 op{};
    op.inv_hx2 = static_cast<float>(1.0 / (p.dx * p.dx));
    op.inv_hy2 = static_cast<float>(1.0 / (p.dy * p.dy));
    op.inv_hz2 = static_cast<float>(1.0 / (p.dz * p.dz));
    op.beta = static_cast<float>(p.willmore_beta);
    op.W4 = static_cast<float>(W2 * W2);
    op.four_W2 = static_cast<float>(4.0 * W2);
    op.beta_dL = static_cast<float>(p.willmore_beta * d_L);
    return op;
}

void launch_precond_tau_scan(const double* tau, float* tdt, double inv_dt,
                             unsigned long long* range_bits, const KernelParams& p,
                             cudaStream_t stream) {
    // range_bits[0] = all ones (identity of min), range_bits[1] = 0 (of max).
    CUDA_CHECK(cudaMemsetAsync(range_bits, 0xFF, sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(range_bits + 1, 0, sizeof(unsigned long long), stream));
    const PcLaunch l = pc_launch(p);
    precond_tau_scan_kernel<<<l.grid, l.block, 0, stream>>>(tau, tdt, inv_dt, range_bits, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_precond_first(const double* r, double scale, float* f, float* x1, const float* tdt,
                          Operator32 op, float s0, GhostRules32 ghosts, double* z_final,
                          double inv_scale, const KernelParams& p, cudaStream_t stream) {
    const PcLaunch l = pc_launch(p);
    precond_first_kernel<<<l.grid, l.block, 0, stream>>>(r, scale, f, x1, tdt, op, s0, ghosts,
                                                         z_final, inv_scale, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_precond_laplacian(const float* x, float* t, Operator32 op, GhostRules32 ghosts,
                              const KernelParams& p, cudaStream_t stream) {
    const PcLaunch l = pc_launch(p);
    precond_laplacian_kernel<<<l.grid, l.block, 0, stream>>>(x, t, op, ghosts, p);
    CUDA_CHECK(cudaGetLastError());
}

void launch_precond_step(const float* t, const float* x, float* x_prev_next, bool prev_is_zero,
                         const float* f, const float* tdt, Operator32 op, float a, float b,
                         GhostRules32 ghosts, double* z_final, double inv_scale,
                         const KernelParams& p, cudaStream_t stream) {
    const PcLaunch l = pc_launch(p);
    precond_step_kernel<<<l.grid, l.block, 0, stream>>>(t, x, x_prev_next, prev_is_zero, f, tdt, op,
                                                        a, b, ghosts, z_final, inv_scale, p);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
