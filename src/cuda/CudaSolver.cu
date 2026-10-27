#include "cuda/CudaSolver.cuh"
#include "cuda/WillmoreKernels.cuh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
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
    // Kernels index cells with 32-bit int (idx3d, linear_to_3d).
    if (static_cast<long long>(config.grid.Nx) * config.grid.Ny * config.grid.Nz >
        std::numeric_limits<int>::max())
        throw std::invalid_argument("CudaSolver: grid exceeds 2^31-1 points (32-bit indexing)");
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
    // Two results and two partial maxima per block: imex_change reduces the
    // change and the magnitude in one pass.
    d_reduction_result_ = DeviceField<double>(2);
    {
        constexpr int BLOCK_SIZE = 256;
        int num_blocks = static_cast<int>((total_points_ + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));
        num_blocks = std::max(num_blocks, 1);
        reduction_scratch_ = DeviceField<double>(2 * static_cast<std::size_t>(num_blocks));
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
    }

    if (params_.willmore_beta > 0.0) {
        if (scheme_ != TimeScheme::Euler && scheme_ != TimeScheme::IMEX)
            throw std::invalid_argument(
                "CudaSolver: physics.willmore_beta > 0 requires the euler or imex scheme");
        // Homogeneous form of phi's BCs for the Willmore operator's fields.
        hom_phi_faces_ = config_.boundary.per_face
                             ? config_.boundary.phi_faces
                             : PerFaceBoundary::uniform(config_.boundary.phi_bc);
        for (auto& f : hom_phi_faces_.faces) {
            f.value = 0.0;
            f.flux = 0.0;
            f.gamma = 0.0;
        }
        for (auto* f : {&w_tau_, &cg_x_, &cg_r_, &cg_z_, &cg_p_, &cg_Ap_, &cg_t_})
            *f = DeviceField<double>(total_points_);
        plane_dot_scratch_ = DeviceField<double>(static_cast<std::size_t>(config.grid.Nx));
        // Preconditioner fields. Zeroed once: their edge and corner ghost cells
        // are never written (no stencil reads them), so they stay defined.
        pc_ghosts_ = make_ghost_rules32(hom_phi_faces_, params_);
        for (auto* f : {&pc_tdt_, &pc_f_, &pc_x_[0], &pc_x_[1], &pc_t_}) {
            *f = DeviceField<float>(total_points_);
            f->zero_async(compute_stream_);
        }
        pc_tau_range_ = DeviceField<unsigned long long>(2);
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
    advance_phi();
    euler_thermal_and_swap();
}

void CudaSolver::advance_phi() {
    if (params_.willmore_beta > 0.0) {
        if (in_decomposition_)
            throw std::logic_error("CudaSolver::advance_phi: a Willmore step of a domain-"
                                   "decomposed run must be driven by MultiGPUSolver");
        last_willmore_stats_ =
            solve_willmore_step({WillmoreDomain{this, device_id_, 1, params_.Nx - 1, 1}}, config_,
                                [](WillmoreBuffer) {});
        return;
    }
    launch_allen_cahn_fused(phi_old_.data(), phi_new_.data(), u_old_.data(), params_,
                            compute_stream_);
    apply_phi_bc(phi_new_.data());
}

void CudaSolver::euler_thermal_and_swap() {
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
    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_in, k_phi[s]->data(), u_in, params_);
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
    // The rule throws instead of returning an iterate it cannot certify, and
    // the change it requires to decrease between checks ends the loop.
    JacobiStoppingRule rule = jacobi_stopping_rule();
    for (int sweeps = 1;; ++sweeps) {
        imex_sweep();
        if (sweeps % kJacobiCheckFreq == 0) {
            if (rule.converged(sweeps, imex_change(0, params_.Nx)))
                break;
        }
    }
    imex_finish();
}

JacobiStoppingRule CudaSolver::jacobi_stopping_rule() const {
    // s = the diagonal of jacobi_step_kernel minus one, computed as there.
    const double alpha = params_.dt;
    if (params_.stencil_type == 1) {
        const double coeff = alpha * params_.D / (30.0 * (params_.dx * params_.dx));
        return {coeff * 128.0, kJacobiRoundingUnits27, kJacobiTol};
    }
    const double inv_dx2 = 1.0 / (params_.dx * params_.dx);
    const double inv_dy2 = 1.0 / (params_.dy * params_.dy);
    const double inv_dz2 = 1.0 / (params_.dz * params_.dz);
    return {alpha * params_.D * 2.0 * (inv_dx2 + inv_dy2 + inv_dz2), kJacobiRoundingUnits7,
            kJacobiTol};
}

void CudaSolver::imex_begin(double dt) {
    params_.dt = dt;

    // Phase field (explicit, or linearly implicit with the Willmore term)
    advance_phi();
    imex_begin_thermal();
}

void CudaSolver::imex_begin_thermal() {
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

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
    // (instead of leftover scratch data), and the change between iterates is
    // that of the iteration (docs/THEORY.md 10.2).
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

JacobiChange CudaSolver::imex_change(int x_begin, int x_end) const {
    // After imex_sweep's swap u_new_ holds u_{k+1} and u_tmp_ holds u_k.
    const std::size_t plane = static_cast<std::size_t>(params_.Ny) * params_.Nz;
    const std::size_t offset = static_cast<std::size_t>(x_begin) * plane;
    const std::size_t count = static_cast<std::size_t>(x_end - x_begin) * plane;
    launch_max_abs_diff_and_max_abs(u_new_.data() + offset, u_tmp_.data() + offset,
                                    d_reduction_result_.data(), count, reduction_scratch_.data(),
                                    static_cast<int>(reduction_scratch_.size()), compute_stream_);
    double result[2] = {0.0, 0.0};
    CUDA_CHECK(cudaMemcpyAsync(result, d_reduction_result_.data(), 2 * sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return {result[0], result[1]};
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

void CudaSolver::apply_homogeneous_phi_bc(double* field) {
    apply_bc_per_face(field, hom_phi_faces_);
}

template <typename Solver> auto CudaSolver::cg_vector_of(Solver& self, CgVector v) {
    switch (v) {
    case CgVector::R:
        return self.cg_r_.data();
    case CgVector::Z:
        return self.cg_z_.data();
    case CgVector::P:
        return self.cg_p_.data();
    case CgVector::AP:
        return self.cg_Ap_.data();
    case CgVector::X:
        return self.cg_x_.data();
    case CgVector::Tau:
        return self.w_tau_.data();
    }
    throw std::logic_error("unknown CgVector");
}

const double* CudaSolver::cg_vector(CgVector v) const {
    return cg_vector_of(*this, v);
}

double* CudaSolver::cg_vector(CgVector v) {
    return cg_vector_of(*this, v);
}

// ── Willmore step building blocks ──────────────────────────────────────────

void CudaSolver::willmore_omega() {
    launch_willmore_omega(phi_old_.data(), cg_t_.data(), params_, compute_stream_);
    apply_homogeneous_phi_bc(cg_t_.data());
}

void CudaSolver::willmore_rhs_init() {
    launch_willmore_rhs_init(phi_old_.data(), u_old_.data(), cg_t_.data(), w_tau_.data(),
                             cg_x_.data(), cg_r_.data(), cg_z_.data(), cg_p_.data(), params_,
                             compute_stream_);
    launch_precond_tau_scan(w_tau_.data(), pc_tdt_.data(), 1.0 / params_.dt, pc_tau_range_.data(),
                            params_, compute_stream_);
}

std::pair<double, double> CudaSolver::willmore_tau_range() const {
    unsigned long long bits[2];
    CUDA_CHECK(cudaMemcpyAsync(bits, pc_tau_range_.data(), sizeof(bits), cudaMemcpyDeviceToHost,
                               compute_stream_));
    compute_stream_.synchronize();
    return {std::bit_cast<double>(bits[0]), std::bit_cast<double>(bits[1])};
}

void CudaSolver::willmore_precondition_first(const ChebyshevPlan& plan, double scale,
                                             double inv_scale, bool last) {
    pc_current_ = 0;
    pc_op_ = make_operator32(params_, plan.d_L);
    launch_precond_first(cg_r_.data(), scale, pc_f_.data(), pc_x_[0].data(), pc_tdt_.data(), pc_op_,
                         static_cast<float>(plan.s0), pc_ghosts_, last ? cg_z_.data() : nullptr,
                         inv_scale, params_, compute_stream_);
}

void CudaSolver::willmore_precondition_laplacian() {
    launch_precond_laplacian(pc_x_[pc_current_].data(), pc_t_.data(), pc_op_, pc_ghosts_, params_,
                             compute_stream_);
}

void CudaSolver::willmore_precondition_step(const ChebyshevPlan& plan, int k, double inv_scale,
                                            bool last) {
    const auto i = static_cast<std::size_t>(k - 1);
    const int next = 1 - pc_current_;
    launch_precond_step(pc_t_.data(), pc_x_[pc_current_].data(), pc_x_[next].data(), k == 1,
                        pc_f_.data(), pc_tdt_.data(), pc_op_, static_cast<float>(plan.step_a[i]),
                        static_cast<float>(plan.step_b[i]), pc_ghosts_,
                        last ? cg_z_.data() : nullptr, inv_scale, params_, compute_stream_);
    if (!last)
        pc_current_ = next;
}

void CudaSolver::willmore_direction_bc() {
    apply_homogeneous_phi_bc(cg_p_.data());
}

void CudaSolver::willmore_direction_laplacian() {
    launch_laplacian_interior(cg_p_.data(), cg_t_.data(), params_, compute_stream_);
    apply_homogeneous_phi_bc(cg_t_.data());
}

void CudaSolver::willmore_apply() {
    launch_willmore_apply(cg_p_.data(), cg_t_.data(), w_tau_.data(), cg_Ap_.data(), params_,
                          compute_stream_);
}

void CudaSolver::willmore_update_xr(double alpha) {
    launch_cg_update_xr(cg_x_.data(), cg_r_.data(), cg_p_.data(), cg_Ap_.data(), alpha, params_,
                        compute_stream_);
}

void CudaSolver::willmore_update_direction(double beta) {
    launch_cg_direction(cg_z_.data(), cg_p_.data(), beta, params_, compute_stream_);
}

void CudaSolver::willmore_finish() {
    launch_willmore_finish(phi_old_.data(), cg_x_.data(), phi_new_.data(), params_,
                           compute_stream_);
    apply_phi_bc(phi_new_.data());
}

std::vector<double> CudaSolver::cg_plane_dot(CgVector a, CgVector b, int x_begin, int x_end) const {
    std::vector<double> out(static_cast<std::size_t>(std::max(0, x_end - x_begin)));
    if (out.empty())
        return out;
    launch_plane_dot(cg_vector(a), cg_vector(b), plane_dot_scratch_.data(), x_begin, x_end, params_,
                     compute_stream_);
    CUDA_CHECK(cudaMemcpyAsync(out.data(), plane_dot_scratch_.data(), out.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return out;
}

void CudaSolver::willmore_upload(CgVector v, const std::vector<double>& host) {
    if (host.size() != total_points_)
        throw std::invalid_argument("willmore_upload: size mismatch");
    CUDA_CHECK(cudaMemcpyAsync(cg_vector(v), host.data(), total_points_ * sizeof(double),
                               cudaMemcpyHostToDevice, compute_stream_));
    compute_stream_.synchronize();
}

std::vector<double> CudaSolver::willmore_download(CgVector v) const {
    std::vector<double> host(total_points_);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), cg_vector(v), total_points_ * sizeof(double),
                               cudaMemcpyDeviceToHost, compute_stream_));
    compute_stream_.synchronize();
    return host;
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

/// Per-block reduction: max(phi) over the interior cells next to the walls in
/// the X range [x_begin, x_end): the cells of the first and last interior Y
/// and Z layers (y = 1, Ny-2; z = 1, Nz-2) in the range's interior planes,
/// plus the planes x_begin+1 / x_end-2 next to X planes x_begin / x_end-1 that
/// are walls (not inter-GPU halos). These are the solution's values next to
/// each wall whatever the boundary condition: the boundary cells themselves
/// hold the BC (a Dirichlet value, a Robin combination), or for zero-flux and
/// periodic faces a copy of these cells. NaN propagates, so a diverged field
/// cannot read as unsaturated. Writes one per-block result (no cross-block
/// atomics: avoids the broken atomicMax-for-negative-doubles pattern).
__global__ void __launch_bounds__(256)
    boundary_max_kernel(const double* __restrict__ phi, double* __restrict__ block_results, int Ny,
                        int Nz, int x_begin, int x_end, int x_lo_face, int x_hi_face) {
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int xs = x_begin + (x_lo_face ? 1 : 0), xe = x_end - (x_hi_face ? 1 : 0);
    const std::size_t nyi = static_cast<std::size_t>(Ny - 2),
                      nzi = static_cast<std::size_t>(Nz - 2);
    const std::size_t nxi = xe > xs ? static_cast<std::size_t>(xe - xs) : 0;
    const std::size_t layer_yz = nyi * nzi;
    const std::size_t n_xlo = x_lo_face ? layer_yz : 0;
    const std::size_t n_xhi = x_hi_face ? layer_yz : 0;
    const std::size_t layer_xz = nxi * nzi;
    const std::size_t layer_xy = nxi * nyi;
    const std::size_t total = n_xlo + n_xhi + 2 * layer_xz + 2 * layer_xy;
    double local_max = -1e30;

    for (std::size_t i = gid; i < total; i += gridDim.x * blockDim.x) {
        int x, y, z;
        std::size_t off = i;
        if (off < n_xlo) {
            x = x_begin + 1;
            y = 1 + static_cast<int>(off / nzi);
            z = 1 + static_cast<int>(off % nzi);
        } else if ((off -= n_xlo) < n_xhi) {
            x = x_end - 2;
            y = 1 + static_cast<int>(off / nzi);
            z = 1 + static_cast<int>(off % nzi);
        } else if ((off -= n_xhi) < layer_xz) {
            x = xs + static_cast<int>(off / nzi);
            y = 1;
            z = 1 + static_cast<int>(off % nzi);
        } else if ((off -= layer_xz) < layer_xz) {
            x = xs + static_cast<int>(off / nzi);
            y = Ny - 2;
            z = 1 + static_cast<int>(off % nzi);
        } else if ((off -= layer_xz) < layer_xy) {
            x = xs + static_cast<int>(off / nyi);
            y = 1 + static_cast<int>(off % nyi);
            z = 1;
        } else {
            off -= layer_xy;
            x = xs + static_cast<int>(off / nyi);
            y = 1 + static_cast<int>(off % nyi);
            z = Nz - 2;
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
    // Cells read by boundary_max_kernel: the interior layers next to the walls.
    const int xs = x_begin + (x_lo_face ? 1 : 0), xe = x_end - (x_hi_face ? 1 : 0);
    const std::size_t nyi = static_cast<std::size_t>(params_.Ny - 2),
                      nzi = static_cast<std::size_t>(params_.Nz - 2);
    const std::size_t nxi = xe > xs ? static_cast<std::size_t>(xe - xs) : 0;
    std::size_t total = 2 * nxi * nzi + 2 * nxi * nyi;
    if (x_lo_face)
        total += nyi * nzi;
    if (x_hi_face)
        total += nyi * nzi;
    auto cfg = LaunchConfig::for_1d(total, 256);
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
