#include "cuda/CudaSolver.cuh"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ac::cuda {

// ── Helper kernels for RK stages (axpy-like operations) ────────────────────

/// y = a + dt * b  (element-wise)
__global__ void __launch_bounds__(256)
    axpy_kernel(double* __restrict__ y, const double* __restrict__ a, const double* __restrict__ b,
                double dt, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(N)) {
        y[i] = a[i] + dt * b[i];
    }
}

/// y = a + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)  (RK4 combination)
__global__ void __launch_bounds__(256)
    rk4_combine_kernel(double* __restrict__ y, const double* __restrict__ a,
                       const double* __restrict__ k1, const double* __restrict__ k2,
                       const double* __restrict__ k3, const double* __restrict__ k4, double dt,
                       int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(N)) {
        y[i] = a[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// Add latent heat coupling: u[i] += 0.5 * (phi_new[i] - phi_old[i])
__global__ void __launch_bounds__(256)
    add_latent_heat_kernel(double* __restrict__ u, const double* __restrict__ phi_new,
                           const double* __restrict__ phi_old, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(N)) {
        u[i] += 0.5 * (phi_new[i] - phi_old[i]);
    }
}

/// Jacobi iteration kernel for IMEX: solve (I - dt*D*Laplacian) u = rhs
/// Dispatches to 7-point or 27-point stencil based on p.stencil_type.
__global__ void __launch_bounds__(256)
    jacobi_step_kernel(const double* __restrict__ u_old, double* __restrict__ u_new,
                       const double* __restrict__ rhs, KernelParams p, double alpha) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = static_cast<std::size_t>(p.Nx) * p.Ny * p.Nz;
    if (tid >= total)
        return;

    int x, y, z;
    linear_to_3d(static_cast<int>(tid), p.Ny, p.Nz, x, y, z);

    if (x < 1 || x >= p.Nx - 1 || y < 1 || y >= p.Ny - 1 || z < 1 || z >= p.Nz - 1) {
        return;
    }

    int c = idx3d(x, y, z, p.Ny, p.Nz);

    if (p.stencil_type == 1) {
        // 27-point isotropic stencil Jacobi iteration (Patra-Karttunen)
        // Laplacian = (14*face + 3*edge + 1*corner - 128*center) / (30*h^2)
        double h2 = p.dx * p.dx;
        double coeff = alpha * p.D / (30.0 * h2);

        double face =
            u_old[idx3d(x + 1, y, z, p.Ny, p.Nz)] + u_old[idx3d(x - 1, y, z, p.Ny, p.Nz)] +
            u_old[idx3d(x, y + 1, z, p.Ny, p.Nz)] + u_old[idx3d(x, y - 1, z, p.Ny, p.Nz)] +
            u_old[idx3d(x, y, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x, y, z - 1, p.Ny, p.Nz)];

        double edge =
            u_old[idx3d(x + 1, y + 1, z, p.Ny, p.Nz)] + u_old[idx3d(x + 1, y - 1, z, p.Ny, p.Nz)] +
            u_old[idx3d(x - 1, y + 1, z, p.Ny, p.Nz)] + u_old[idx3d(x - 1, y - 1, z, p.Ny, p.Nz)] +
            u_old[idx3d(x + 1, y, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x + 1, y, z - 1, p.Ny, p.Nz)] +
            u_old[idx3d(x - 1, y, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x - 1, y, z - 1, p.Ny, p.Nz)] +
            u_old[idx3d(x, y + 1, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x, y + 1, z - 1, p.Ny, p.Nz)] +
            u_old[idx3d(x, y - 1, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x, y - 1, z - 1, p.Ny, p.Nz)];

        double corner = u_old[idx3d(x + 1, y + 1, z + 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x + 1, y + 1, z - 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x + 1, y - 1, z + 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x + 1, y - 1, z - 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x - 1, y + 1, z + 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x - 1, y + 1, z - 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x - 1, y - 1, z + 1, p.Ny, p.Nz)] +
                        u_old[idx3d(x - 1, y - 1, z - 1, p.Ny, p.Nz)];

        double off_diag = coeff * (14.0 * face + 3.0 * edge + 1.0 * corner);
        double diag = 1.0 + coeff * 128.0;

        u_new[c] = (rhs[c] + off_diag) / diag;
    } else {
        // Standard 7-point stencil Jacobi iteration
        double inv_dx2 = 1.0 / (p.dx * p.dx);
        double inv_dy2 = 1.0 / (p.dy * p.dy);
        double inv_dz2 = 1.0 / (p.dz * p.dz);

        double neighbors =
            (u_old[idx3d(x + 1, y, z, p.Ny, p.Nz)] + u_old[idx3d(x - 1, y, z, p.Ny, p.Nz)]) *
                inv_dx2 +
            (u_old[idx3d(x, y + 1, z, p.Ny, p.Nz)] + u_old[idx3d(x, y - 1, z, p.Ny, p.Nz)]) *
                inv_dy2 +
            (u_old[idx3d(x, y, z + 1, p.Ny, p.Nz)] + u_old[idx3d(x, y, z - 1, p.Ny, p.Nz)]) *
                inv_dz2;

        double diag = 1.0 + alpha * p.D * 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);

        u_new[c] = (rhs[c] + alpha * p.D * neighbors) / diag;
    }
}

// ── CudaSolver implementation ──────────────────────────────────────────────

int CudaSolver::activate_device(const SimulationConfig& config) {
    if (config.gpu.device_ids.empty())
        throw std::invalid_argument("CudaSolver requires at least one entry in gpu.device_ids");
    const int device = config.gpu.device_ids.front();
    CUDA_CHECK(cudaSetDevice(device));
    return device;
}

CudaSolver::CudaSolver(const SimulationConfig& config)
    : device_id_(activate_device(config)), config_(config),
      params_(KernelParams::from_config(config)), scheme_(config.time.scheme),
      total_points_(static_cast<std::size_t>(config.grid.Nx) * config.grid.Ny * config.grid.Nz) {
    // Allocate primary field buffers
    phi_old_ = DeviceField<double>(total_points_);
    phi_new_ = DeviceField<double>(total_points_);
    u_old_ = DeviceField<double>(total_points_);
    u_new_ = DeviceField<double>(total_points_);

    // Reduction scratch
    d_reduction_result_ = DeviceField<double>(1);
    {
        constexpr int BLOCK_SIZE = 256;
        int num_blocks = static_cast<int>((total_points_ + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
        num_blocks = std::max(num_blocks, 1);
        reduction_scratch_ = DeviceField<double>(static_cast<std::size_t>(num_blocks));
    }

    // Allocate temporaries based on time integration scheme
    if (scheme_ == TimeScheme::Heun || scheme_ == TimeScheme::RK4 || scheme_ == TimeScheme::IMEX) {
        phi_tmp_ = DeviceField<double>(total_points_);
        u_tmp_ = DeviceField<double>(total_points_);
    }

    if (scheme_ == TimeScheme::RK4) {
        k1_phi_ = DeviceField<double>(total_points_);
        k2_phi_ = DeviceField<double>(total_points_);
        k3_phi_ = DeviceField<double>(total_points_);
        k4_phi_ = DeviceField<double>(total_points_);
        k1_u_ = DeviceField<double>(total_points_);
        k2_u_ = DeviceField<double>(total_points_);
        k3_u_ = DeviceField<double>(total_points_);
        k4_u_ = DeviceField<double>(total_points_);
        // Force fields for non-fused kernel path
        Fx_ = DeviceField<double>(total_points_);
        Fy_ = DeviceField<double>(total_points_);
        Fz_ = DeviceField<double>(total_points_);
    }

    spdlog::info("CudaSolver initialized: {} total points, scheme={}", total_points_,
                 scheme_ == TimeScheme::Euler  ? "Euler"
                 : scheme_ == TimeScheme::Heun ? "Heun"
                 : scheme_ == TimeScheme::RK4  ? "RK4"
                                               : "IMEX");
}

void CudaSolver::initialize(const FieldData& phi0, const FieldData& u0) {
    phi_old_.copy_from_host(phi0.data(), compute_stream_);
    u_old_.copy_from_host(u0.data(), compute_stream_);
    phi_new_.zero_async(compute_stream_);
    u_new_.zero_async(compute_stream_);
    apply_boundary_conditions();
    compute_stream_.synchronize();
    spdlog::debug("Initial conditions uploaded to GPU");
}

void CudaSolver::step(double dt) {
    params_.dt = dt;

    switch (scheme_) {
    case TimeScheme::Euler:
        step_euler(dt);
        break;
    case TimeScheme::Heun:
        step_heun(dt);
        break;
    case TimeScheme::RK4:
        step_rk4(dt);
        break;
    case TimeScheme::IMEX:
        step_imex(dt);
        break;
    }
}

// ── Euler step ─────────────────────────────────────────────────────────────

void CudaSolver::step_euler(double dt) {
    params_.dt = dt;

    // Phase field update (fused kernel)
    launch_allen_cahn_fused(phi_old_.data(), phi_new_.data(), u_old_.data(), params_,
                            compute_stream_);

    // Apply BCs to phi
    apply_phi_bc(phi_new_.data());

    // Thermal update
    launch_thermal_equation(u_old_.data(), u_new_.data(), phi_new_.data(), phi_old_.data(), params_,
                            compute_stream_);

    // Apply BCs to u
    apply_u_bc(u_new_.data());

    // Swap old <-> new (O(1) pointer swap, zero GPU cost)
    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── Heun's method (RK2) ───────────────────────────────────────────────────
// Correct Heun formula: y_{n+1} = 0.5*(y_n + y_tilde)
// where y_tilde = y_n + dt*f(y_n) (Euler predictor), then
// y_{n+1} = 0.5*(y_n + (y_tilde + dt*f(y_tilde)))
// Equivalently: y_{n+1} = y_n + (dt/2)*(f(y_n) + f(y_tilde))

/// avg[i] = 0.5 * (a[i] + b[i])
/// Note: out may alias b (Heun average), so no __restrict__ on out/b.
__global__ void __launch_bounds__(256)
    average_kernel(double* out, const double* __restrict__ a, const double* b, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(N)) {
        out[i] = 0.5 * (a[i] + b[i]);
    }
}

void CudaSolver::step_heun(double dt) {
    step_heun_stage1(dt);
    step_heun_stage2(dt);
}

// ── Heun stages (split so MultiGPUSolver can exchange halos in between) ──

void CudaSolver::step_heun_stage1(double dt) {
    params_.dt = dt;

    // Euler predictor -> phi_tmp_, u_tmp_
    // phi_tmp_ = phi_old + dt*f_phi(phi_old, u_old)
    launch_allen_cahn_fused(phi_old_.data(), phi_tmp_.data(), u_old_.data(), params_,
                            compute_stream_);
    apply_phi_bc(phi_tmp_.data());

    // u_tmp_ = u_old + latent_heat + dt*D*lap(u_old)
    launch_thermal_equation(u_old_.data(), u_tmp_.data(), phi_tmp_.data(), phi_old_.data(), params_,
                            compute_stream_);
    apply_u_bc(u_tmp_.data());
}

void CudaSolver::step_heun_stage2(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Euler from predicted state -> phi_new_, u_new_
    // phi_new_ = phi_tmp_ + dt*f_phi(phi_tmp_, u_tmp_)
    launch_allen_cahn_fused(phi_tmp_.data(), phi_new_.data(), u_tmp_.data(), params_,
                            compute_stream_);
    apply_phi_bc(phi_new_.data());

    // u_new_ = u_tmp_ + latent_heat + dt*D*lap(u_tmp_)
    launch_thermal_equation(u_tmp_.data(), u_new_.data(), phi_new_.data(), phi_tmp_.data(), params_,
                            compute_stream_);
    apply_u_bc(u_new_.data());

    // Heun average: y_{n+1} = 0.5*(y_n + y_tilde + dt*f(y_tilde))
    //             = 0.5*(phi_old + phi_new)  where phi_new = phi_tmp + dt*f(phi_tmp)
    //             = phi_old + 0.5*dt*(f1 + f2)  — correct RK2 formula
    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_new_.data(), phi_old_.data(), phi_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_new_.data(), u_old_.data(),
                                                                      u_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    apply_phi_bc(phi_new_.data());
    apply_u_bc(u_new_.data());

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── RK4 step ───────────────────────────────────────────────────────────────

void CudaSolver::step_rk4(double dt) {
    for (int stage = 1; stage <= 4; ++stage)
        rk4_stage(stage, dt);
}

void CudaSolver::rk4_stage(int stage, double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);
    DeviceField<double>* k_phi[4] = {&k1_phi_, &k2_phi_, &k3_phi_, &k4_phi_};
    DeviceField<double>* k_u[4] = {&k1_u_, &k2_u_, &k3_u_, &k4_u_};
    const int s = stage - 1;

    // k_s = f(stage state): y_n for stage 1, the tmp state built by the
    // previous stage otherwise.
    const double* phi_in = (stage == 1) ? phi_old_.data() : phi_tmp_.data();
    const double* u_in = (stage == 1) ? u_old_.data() : u_tmp_.data();
    compute_force_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_in, Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());
    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_in, k_phi[s]->data(), u_in, Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());
    thermal_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        u_in, k_u[s]->data(), k_phi[s]->data(), params_);
    CUDA_CHECK(cudaGetLastError());

    if (stage < 4) {
        // Next stage state: y_n + c*dt*k_s with c = 1/2, 1/2, 1.
        const double c = (stage == 3) ? dt : 0.5 * dt;
        axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
            phi_tmp_.data(), phi_old_.data(), k_phi[s]->data(), c, N);
        CUDA_CHECK(cudaGetLastError());
        axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(), u_old_.data(),
                                                                       k_u[s]->data(), c, N);
        CUDA_CHECK(cudaGetLastError());
        apply_phi_bc(phi_tmp_.data());
        apply_u_bc(u_tmp_.data());
        return;
    }

    // Combine: y_{n+1} = y_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    rk4_combine_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_new_.data(), phi_old_.data(), k1_phi_.data(), k2_phi_.data(), k3_phi_.data(),
        k4_phi_.data(), dt, N);
    CUDA_CHECK(cudaGetLastError());
    rk4_combine_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        u_new_.data(), u_old_.data(), k1_u_.data(), k2_u_.data(), k3_u_.data(), k4_u_.data(), dt,
        N);
    CUDA_CHECK(cudaGetLastError());

    apply_phi_bc(phi_new_.data());
    apply_u_bc(u_new_.data());

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── IMEX step ──────────────────────────────────────────────────────────────

void CudaSolver::step_imex(double dt) {
    imex_begin(dt);
    for (int iter = 0; iter < kJacobiMaxIters; ++iter) {
        imex_sweep();
        if ((iter + 1) % kJacobiCheckFreq == 0 && imex_residual(0, params_.Nx) < kJacobiTol)
            break;
    }
    imex_finish();
}

void CudaSolver::imex_begin(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Explicit step for Allen-Cahn (reaction + anisotropy are explicit)
    launch_allen_cahn_fused(phi_old_.data(), phi_new_.data(), u_old_.data(), params_,
                            compute_stream_);
    apply_phi_bc(phi_new_.data());

    // Implicit step for thermal diffusion
    // Solve: (I - dt*D*Laplacian) u_new = u_old + 0.5*(phi_new - phi_old)
    // RHS is built directly in phi_tmp_ so that neither Jacobi buffer is used
    // as scratch.
    phi_tmp_.copy_from(u_old_, compute_stream_);
    add_latent_heat_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), phi_new_.data(), phi_old_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    // Jacobi iterations to solve (I - dt*D*Lap) u_new = phi_tmp_.
    // jacobi_step_kernel writes interior cells only, so both iterate buffers
    // start from u_old and the BC is re-applied to every new iterate: boundary
    // cells then always hold values consistent with the current interior
    // (instead of leftover scratch data), and the residual measures real
    // convergence. Adaptive iteration count: check residual every
    // kJacobiCheckFreq sweeps, exit early when converged; at most
    // kJacobiMaxIters to handle large dt*D/h^2 ratios.
    u_new_.copy_from(u_old_, compute_stream_);
    u_tmp_.copy_from(u_old_, compute_stream_);
}

void CudaSolver::imex_sweep() {
    auto cfg = LaunchConfig::for_1d(total_points_, 256);
    jacobi_step_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        u_new_.data(), u_tmp_.data(), phi_tmp_.data(), params_, params_.dt);
    CUDA_CHECK(cudaGetLastError());
    apply_u_bc(u_tmp_.data());
    swap(u_new_, u_tmp_);
}

double CudaSolver::imex_residual(int x_begin, int x_end) const {
    const std::size_t plane = static_cast<std::size_t>(params_.Ny) * params_.Nz;
    const std::size_t offset = static_cast<std::size_t>(x_begin) * plane;
    const std::size_t count = static_cast<std::size_t>(x_end - x_begin) * plane;
    launch_max_abs_diff(u_new_.data() + offset, u_tmp_.data() + offset, d_reduction_result_.data(),
                        count, reduction_scratch_.data(),
                        static_cast<int>(reduction_scratch_.size()), compute_stream_);
    double residual = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&residual, d_reduction_result_.data(), sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return residual;
}

void CudaSolver::imex_finish() {
    apply_u_bc(u_new_.data());

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── Helper methods ─────────────────────────────────────────────────────────

void CudaSolver::apply_bc(double* field, const BoundaryConfig& bc) {
    launch_boundary_conditions(field, params_, bc.type, bc.value, bc.flux, bc.alpha, bc.beta,
                               bc.gamma, compute_stream_);
}

void CudaSolver::apply_bc_per_face(double* field, const PerFaceBoundary& face_bcs) {
    launch_boundary_conditions_per_face(field, params_, face_bcs, compute_stream_);
}

// Every time-integration stage must honour per-face BCs, not just initialize():
// the uniform phi_bc/u_bc are left at their defaults when a config specifies
// faces individually, so dispatching on per_face here is what makes the
// per-face specification effective for the whole run.
void CudaSolver::apply_phi_bc(double* field) {
    if (config_.boundary.per_face)
        apply_bc_per_face(field, config_.boundary.phi_faces);
    else
        apply_bc(field, config_.boundary.phi_bc);
}

void CudaSolver::apply_u_bc(double* field) {
    if (config_.boundary.per_face)
        apply_bc_per_face(field, config_.boundary.u_faces);
    else
        apply_bc(field, config_.boundary.u_bc);
}

void CudaSolver::apply_boundary_conditions() {
    apply_phi_bc(phi_old_.data());
    apply_u_bc(u_old_.data());
}

/// Warp-level max reduction (reused from ReductionKernels pattern).
__device__ double warp_reduce_max(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = nan_max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

/// Per-block reduction: max(phi) over the physical boundary cells of the
/// X range [x_begin, x_end): the Y and Z faces restricted to that range, plus
/// the X planes x_begin / x_end-1 when they are walls (not inter-GPU halos).
/// NaN propagates, so a diverged field cannot read as unsaturated.
/// Writes one per-block result (no cross-block atomics — avoids the broken
/// atomicMax-for-negative-doubles pattern).
__global__ void __launch_bounds__(256)
    boundary_max_kernel(const double* __restrict__ phi, double* __restrict__ block_results, int Ny,
                        int Nz, int x_begin, int x_end, int x_lo_face, int x_hi_face) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nx = static_cast<std::size_t>(x_end - x_begin);
    const std::size_t face_yz = static_cast<std::size_t>(Ny) * Nz;
    const std::size_t n_xlo = x_lo_face ? face_yz : 0;
    const std::size_t n_xhi = x_hi_face ? face_yz : 0;
    const std::size_t face_xz = nx * Nz;
    const std::size_t face_xy = nx * Ny;
    const std::size_t total_boundary = n_xlo + n_xhi + 2 * face_xz + 2 * face_xy;
    double local_max = -1e30;

    for (std::size_t i = gid; i < total_boundary; i += gridDim.x * blockDim.x) {
        int x, y, z;
        std::size_t off = i;
        if (off < n_xlo) {
            x = x_begin;
            y = static_cast<int>(off / Nz);
            z = static_cast<int>(off % Nz);
        } else if ((off -= n_xlo) < n_xhi) {
            x = x_end - 1;
            y = static_cast<int>(off / Nz);
            z = static_cast<int>(off % Nz);
        } else if ((off -= n_xhi) < face_xz) {
            x = x_begin + static_cast<int>(off / Nz);
            y = 0;
            z = static_cast<int>(off % Nz);
        } else if ((off -= face_xz) < face_xz) {
            x = x_begin + static_cast<int>(off / Nz);
            y = Ny - 1;
            z = static_cast<int>(off % Nz);
        } else if ((off -= face_xz) < face_xy) {
            x = x_begin + static_cast<int>(off / Ny);
            y = static_cast<int>(off % Ny);
            z = 0;
        } else {
            off -= face_xy;
            x = x_begin + static_cast<int>(off / Ny);
            y = static_cast<int>(off % Ny);
            z = Nz - 1;
        }
        double v = phi[(static_cast<std::size_t>(x) * Ny + y) * Nz + z];
        local_max = nan_max(local_max, v);
    }

    sdata[tid] = local_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = nan_max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        double val = sdata[tid];
        if (blockDim.x >= 64)
            val = nan_max(val, sdata[tid + 32]);
        val = warp_reduce_max(val);
        if (tid == 0)
            block_results[blockIdx.x] = val;
    }
}

/// Single-block final pass: reduce per-block maxima to a scalar.
__global__ void __launch_bounds__(256)
    boundary_final_max_kernel(const double* __restrict__ block_results, double* __restrict__ result,
                              int num_blocks) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    double val = -1e30;
    for (unsigned int i = tid; i < static_cast<unsigned>(num_blocks); i += blockDim.x)
        val = nan_max(val, block_results[i]);
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] = nan_max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid < 32) {
        val = sdata[tid];
        if (blockDim.x >= 64)
            val = nan_max(val, sdata[tid + 32]);
        val = warp_reduce_max(val);
        if (tid == 0)
            result[0] = val;
    }
}

double CudaSolver::compute_boundary_max_phi() const {
    return compute_boundary_max_phi(0, params_.Nx, true, true);
}

double CudaSolver::compute_boundary_max_phi(int x_begin, int x_end, bool x_lo_face,
                                            bool x_hi_face) const {
    const std::size_t nx = static_cast<std::size_t>(x_end - x_begin);
    std::size_t total_boundary = static_cast<std::size_t>(2) * nx * params_.Nz +
                                 static_cast<std::size_t>(2) * nx * params_.Ny;
    if (x_lo_face)
        total_boundary += static_cast<std::size_t>(params_.Ny) * params_.Nz;
    if (x_hi_face)
        total_boundary += static_cast<std::size_t>(params_.Ny) * params_.Nz;
    auto cfg = LaunchConfig::for_1d(total_boundary, 256);
    int num_blocks = static_cast<int>(cfg.grid.x);

    DeviceField<double> block_results(static_cast<std::size_t>(num_blocks));
    boundary_max_kernel<<<cfg.grid, cfg.block, 256 * sizeof(double), compute_stream_.get()>>>(
        phi_old_.data(), block_results.data(), params_.Ny, params_.Nz, x_begin, x_end,
        x_lo_face ? 1 : 0, x_hi_face ? 1 : 0);
    CUDA_CHECK(cudaGetLastError());

    boundary_final_max_kernel<<<1, 256, 256 * sizeof(double), compute_stream_.get()>>>(
        block_results.data(), d_reduction_result_.data(), num_blocks);
    CUDA_CHECK(cudaGetLastError());

    double result = -1e30;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_reduction_result_.data(), sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return result;
}

double CudaSolver::compute_max_dphi() const {
    return compute_max_dphi(0, params_.Nx);
}

double CudaSolver::compute_max_dphi(int x_begin, int x_end) const {
    // The layout is x-major, so an X range is one contiguous slab.
    const std::size_t plane = static_cast<std::size_t>(params_.Ny) * params_.Nz;
    const std::size_t offset = static_cast<std::size_t>(x_begin) * plane;
    const std::size_t count = static_cast<std::size_t>(x_end - x_begin) * plane;
    launch_max_abs_diff(phi_old_.data() + offset, phi_new_.data() + offset,
                        d_reduction_result_.data(), count, reduction_scratch_.data(),
                        static_cast<int>(reduction_scratch_.size()), compute_stream_);

    double result = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&result, d_reduction_result_.data(), sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return result;
}

void CudaSolver::copy_phi_to_host(FieldData& out) const {
    // Ensure all in-flight kernel writes on compute_stream_ are visible
    // before issuing the device->host copy on transfer_stream_.
    compute_stream_.synchronize();
    phi_old_.copy_to_host(out.data(), transfer_stream_);
    transfer_stream_.synchronize();
}

void CudaSolver::copy_u_to_host(FieldData& out) const {
    compute_stream_.synchronize();
    u_old_.copy_to_host(out.data(), transfer_stream_);
    transfer_stream_.synchronize();
}

void CudaSolver::synchronize() const {
    compute_stream_.synchronize();
}

} // namespace ac::cuda
