#include "cuda/CudaSolver.cuh"

#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>

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
    int total = p.Nx * p.Ny * p.Nz;
    if (tid >= static_cast<unsigned>(total))
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

CudaSolver::CudaSolver(const SimulationConfig& config)
    : config_(config), params_(KernelParams::from_config(config)), scheme_(config.time.scheme),
      total_points_(static_cast<std::size_t>(config.grid.Nx) * config.grid.Ny * config.grid.Nz) {
    CUDA_CHECK(cudaSetDevice(config.gpu.device_ids.front()));

    // Allocate primary field buffers
    phi_old_ = DeviceField<double>(total_points_);
    phi_new_ = DeviceField<double>(total_points_);
    u_old_ = DeviceField<double>(total_points_);
    u_new_ = DeviceField<double>(total_points_);

    // Reduction scratch
    d_reduction_result_ = DeviceField<double>(1);

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
    apply_bc(phi_new_.data(), config_.boundary.phi_bc);

    // Thermal update
    launch_thermal_equation(u_old_.data(), u_new_.data(), phi_new_.data(), phi_old_.data(), params_,
                            compute_stream_);

    // Apply BCs to u
    apply_bc(u_new_.data(), config_.boundary.u_bc);

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
__global__ void __launch_bounds__(256)
    average_kernel(double* __restrict__ out, const double* __restrict__ a,
                   const double* __restrict__ b, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < static_cast<unsigned>(N)) {
        out[i] = 0.5 * (a[i] + b[i]);
    }
}

void CudaSolver::step_heun(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Stage 1: Euler predictor -> phi_tmp_, u_tmp_
    // phi_tmp_ = phi_old + dt*f_phi(phi_old, u_old)
    launch_allen_cahn_fused(phi_old_.data(), phi_tmp_.data(), u_old_.data(), params_,
                            compute_stream_);
    apply_bc(phi_tmp_.data(), config_.boundary.phi_bc);

    // u_tmp_ = u_old + latent_heat + dt*D*lap(u_old)
    launch_thermal_equation(u_old_.data(), u_tmp_.data(), phi_tmp_.data(), phi_old_.data(), params_,
                            compute_stream_);
    apply_bc(u_tmp_.data(), config_.boundary.u_bc);

    // Stage 2: Euler from predicted state -> phi_new_, u_new_
    // phi_new_ = phi_tmp_ + dt*f_phi(phi_tmp_, u_tmp_)
    launch_allen_cahn_fused(phi_tmp_.data(), phi_new_.data(), u_tmp_.data(), params_,
                            compute_stream_);
    apply_bc(phi_new_.data(), config_.boundary.phi_bc);

    // u_new_ = u_tmp_ + latent_heat + dt*D*lap(u_tmp_)
    launch_thermal_equation(u_tmp_.data(), u_new_.data(), phi_new_.data(), phi_tmp_.data(), params_,
                            compute_stream_);
    apply_bc(u_new_.data(), config_.boundary.u_bc);

    // Heun average: y_{n+1} = 0.5*(y_n + y_tilde + dt*f(y_tilde))
    //             = 0.5*(phi_old + phi_new)  where phi_new = phi_tmp + dt*f(phi_tmp)
    //             = phi_old + 0.5*dt*(f1 + f2)  — correct RK2 formula
    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_new_.data(), phi_old_.data(), phi_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_new_.data(), u_old_.data(),
                                                                      u_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    apply_bc(phi_new_.data(), config_.boundary.phi_bc);
    apply_bc(u_new_.data(), config_.boundary.u_bc);

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── Heun stage 2 (for multi-GPU inter-stage halo exchange) ────────────────

void CudaSolver::step_heun_stage2(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Stage 2: Euler from predicted state -> phi_new_, u_new_
    launch_allen_cahn_fused(phi_tmp_.data(), phi_new_.data(), u_tmp_.data(), params_,
                            compute_stream_);
    apply_bc(phi_new_.data(), config_.boundary.phi_bc);

    launch_thermal_equation(u_tmp_.data(), u_new_.data(), phi_new_.data(), phi_tmp_.data(), params_,
                            compute_stream_);
    apply_bc(u_new_.data(), config_.boundary.u_bc);

    // Heun average: y_{n+1} = 0.5*(y_n + y_tilde2)
    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_new_.data(), phi_old_.data(), phi_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    average_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_new_.data(), u_old_.data(),
                                                                      u_new_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    apply_bc(phi_new_.data(), config_.boundary.phi_bc);
    apply_bc(u_new_.data(), config_.boundary.u_bc);

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── RK4 step ───────────────────────────────────────────────────────────────

void CudaSolver::step_rk4(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Stage 1: k1 = f(t_n, y_n)
    // Compute forces and RHS for phi
    compute_force_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_old_.data(), Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());

    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_old_.data(), k1_phi_.data(), u_old_.data(), Fx_.data(), Fy_.data(), Fz_.data(),
        params_);
    CUDA_CHECK(cudaGetLastError());

    thermal_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_old_.data(),
                                                                          k1_u_.data(), params_);
    CUDA_CHECK(cudaGetLastError());

    // Stage 2: k2 = f(t_n + dt/2, y_n + dt/2 * k1)
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(phi_tmp_.data(), phi_old_.data(),
                                                                   k1_phi_.data(), 0.5 * dt, N);
    CUDA_CHECK(cudaGetLastError());
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(), u_old_.data(),
                                                                   k1_u_.data(), 0.5 * dt, N);
    CUDA_CHECK(cudaGetLastError());
    apply_bc(phi_tmp_.data(), config_.boundary.phi_bc);
    apply_bc(u_tmp_.data(), config_.boundary.u_bc);

    compute_force_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());
    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), k2_phi_.data(), u_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(),
        params_);
    CUDA_CHECK(cudaGetLastError());
    thermal_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(),
                                                                          k2_u_.data(), params_);
    CUDA_CHECK(cudaGetLastError());

    // Stage 3: k3 = f(t_n + dt/2, y_n + dt/2 * k2)
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(phi_tmp_.data(), phi_old_.data(),
                                                                   k2_phi_.data(), 0.5 * dt, N);
    CUDA_CHECK(cudaGetLastError());
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(), u_old_.data(),
                                                                   k2_u_.data(), 0.5 * dt, N);
    CUDA_CHECK(cudaGetLastError());
    apply_bc(phi_tmp_.data(), config_.boundary.phi_bc);
    apply_bc(u_tmp_.data(), config_.boundary.u_bc);

    compute_force_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());
    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), k3_phi_.data(), u_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(),
        params_);
    CUDA_CHECK(cudaGetLastError());
    thermal_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(),
                                                                          k3_u_.data(), params_);
    CUDA_CHECK(cudaGetLastError());

    // Stage 4: k4 = f(t_n + dt, y_n + dt * k3)
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(phi_tmp_.data(), phi_old_.data(),
                                                                   k3_phi_.data(), dt, N);
    CUDA_CHECK(cudaGetLastError());
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(), u_old_.data(),
                                                                   k3_u_.data(), dt, N);
    CUDA_CHECK(cudaGetLastError());
    apply_bc(phi_tmp_.data(), config_.boundary.phi_bc);
    apply_bc(u_tmp_.data(), config_.boundary.u_bc);

    compute_force_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(), params_);
    CUDA_CHECK(cudaGetLastError());
    allen_cahn_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_tmp_.data(), k4_phi_.data(), u_tmp_.data(), Fx_.data(), Fy_.data(), Fz_.data(),
        params_);
    CUDA_CHECK(cudaGetLastError());
    thermal_rhs_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(),
                                                                          k4_u_.data(), params_);
    CUDA_CHECK(cudaGetLastError());

    // Combine: y_{n+1} = y_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    rk4_combine_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        phi_new_.data(), phi_old_.data(), k1_phi_.data(), k2_phi_.data(), k3_phi_.data(),
        k4_phi_.data(), dt, N);
    CUDA_CHECK(cudaGetLastError());
    rk4_combine_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        u_new_.data(), u_old_.data(), k1_u_.data(), k2_u_.data(), k3_u_.data(), k4_u_.data(), dt,
        N);
    CUDA_CHECK(cudaGetLastError());

    // Operator splitting for latent heat: the thermal RHS kernel computes only
    // D·∇²u (diffusion). The latent heat coupling 0.5·∂φ/∂t is added as a
    // separate post-step correction. This is first-order accurate in the
    // splitting error but maintains the RK4 accuracy for each sub-operator.
    add_latent_heat_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
        u_new_.data(), phi_new_.data(), phi_old_.data(), N);
    CUDA_CHECK(cudaGetLastError());

    apply_bc(phi_new_.data(), config_.boundary.phi_bc);
    apply_bc(u_new_.data(), config_.boundary.u_bc);

    swap(phi_old_, phi_new_);
    swap(u_old_, u_new_);
}

// ── IMEX step ──────────────────────────────────────────────────────────────

void CudaSolver::step_imex(double dt) {
    params_.dt = dt;
    int N = static_cast<int>(total_points_);
    auto cfg = LaunchConfig::for_1d(total_points_, 256);

    // Explicit step for Allen-Cahn (reaction + anisotropy are explicit)
    launch_allen_cahn_fused(phi_old_.data(), phi_new_.data(), u_old_.data(), params_,
                            compute_stream_);
    apply_bc(phi_new_.data(), config_.boundary.phi_bc);

    // Implicit step for thermal diffusion
    // Solve: (I - dt*D*Laplacian) u_new = u_old + 0.5*(phi_new - phi_old)
    // Compute RHS in phi_tmp_: first u_tmp_ = phi_new - phi_old, then phi_tmp_ = u_old + 0.5*u_tmp_
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(u_tmp_.data(), phi_new_.data(),
                                                                   phi_old_.data(), -1.0, N);
    CUDA_CHECK(cudaGetLastError());
    axpy_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(phi_tmp_.data(), u_old_.data(),
                                                                   u_tmp_.data(), 0.5, N);
    CUDA_CHECK(cudaGetLastError());

    // Jacobi iterations to solve (I - dt*D*Lap) u_new = phi_tmp_
    // Start from u_old as initial guess. Use 50 iterations for reliable convergence
    // of the Jacobi iteration; spectral radius rho < 1 when dt*D/h^2 is bounded.
    u_new_.copy_from(u_old_, compute_stream_);
    constexpr int JACOBI_ITERS = 50;
    for (int iter = 0; iter < JACOBI_ITERS; ++iter) {
        // Read from u_new_, write to u_tmp_, then swap so u_new_ holds latest result
        jacobi_step_kernel<<<cfg.grid, cfg.block, 0, compute_stream_.get()>>>(
            u_new_.data(), u_tmp_.data(), phi_tmp_.data(), params_, dt);
        CUDA_CHECK(cudaGetLastError());
        swap(u_new_, u_tmp_);
    }
    // After each iteration, swap puts the output into u_new_.
    // After JACOBI_ITERS swaps, u_new_ holds the final result. No extra swap needed.

    apply_bc(u_new_.data(), config_.boundary.u_bc);

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

void CudaSolver::apply_boundary_conditions() {
    if (config_.boundary.per_face) {
        apply_bc_per_face(phi_old_.data(), config_.boundary.phi_faces);
        apply_bc_per_face(u_old_.data(), config_.boundary.u_faces);
    } else {
        apply_bc(phi_old_.data(), config_.boundary.phi_bc);
        apply_bc(u_old_.data(), config_.boundary.u_bc);
    }
}

double CudaSolver::compute_max_dphi() const {
    launch_max_abs_diff(phi_old_.data(), phi_new_.data(), d_reduction_result_.data(), total_points_,
                        compute_stream_);

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
