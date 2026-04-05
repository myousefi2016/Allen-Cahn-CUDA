#pragma once

#include "core/SimulationConfig.hpp"
#include "core/FieldData.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"

#include <memory>

namespace ac::cuda {

/// GPU solver facade. Owns all device memory and launches kernels.
/// Supports Euler, Heun (RK2), RK4, and IMEX time integration.
class CudaSolver {
public:
    explicit CudaSolver(const SimulationConfig& config);
    ~CudaSolver() = default;  // RAII handles all cleanup

    /// Upload initial conditions to the GPU.
    void initialize(const FieldData& phi0, const FieldData& u0);

    /// Perform one complete time step (dispatches to selected scheme).
    void step(double dt);

    /// Explicit Euler step.
    void step_euler(double dt);

    /// Heun's method (RK2, 2nd order).
    void step_heun(double dt);

    /// Classical Runge-Kutta (RK4, 4th order).
    void step_rk4(double dt);

    /// IMEX: implicit diffusion, explicit reaction.
    void step_imex(double dt);

    /// Compute max |phi_new - phi_old| for adaptive time stepping.
    [[nodiscard]] double compute_max_dphi() const;

    /// Copy phi from device to host.
    void copy_phi_to_host(FieldData& out) const;

    /// Copy u from device to host.
    void copy_u_to_host(FieldData& out) const;

    /// Apply boundary conditions to both fields.
    void apply_boundary_conditions();

    /// Get the kernel parameters (for testing).
    [[nodiscard]] const KernelParams& params() const { return params_; }

    /// Synchronize the compute stream.
    void synchronize() const;

    /// Get compute stream handle.
    [[nodiscard]] cudaStream_t stream() const { return compute_stream_.get(); }

private:
    /// Apply BCs to a single field.
    void apply_bc(double* field, const BoundaryConfig& bc);

    /// One Euler sub-step for phi: phi_out = phi_in + dt * RHS(phi_in, u_in)
    void euler_substep_phi(const double* phi_in, double* phi_out,
                            const double* u_in, double dt);

    /// One Euler sub-step for u: u_out = u_in + 0.5*(phi_new - phi_old) + dt*D*lap(u_in)
    void euler_substep_u(const double* u_in, double* u_out,
                          const double* phi_new, const double* phi_old, double dt);

    SimulationConfig config_;
    KernelParams params_;

    // Primary field buffers (swapped via pointer swap, zero GPU cost)
    DeviceField<double> phi_old_, phi_new_;
    DeviceField<double> u_old_, u_new_;

    // Temporary buffers for higher-order time integration
    DeviceField<double> phi_tmp_, u_tmp_;     // Heun: predictor stage
    DeviceField<double> k1_phi_, k2_phi_, k3_phi_, k4_phi_;  // RK4
    DeviceField<double> k1_u_, k2_u_, k3_u_, k4_u_;          // RK4

    // Force field buffers (needed for non-fused RK stages)
    DeviceField<double> Fx_, Fy_, Fz_;

    // Reduction scratch
    DeviceField<double> d_reduction_result_;

    // CUDA streams
    Stream compute_stream_;
    Stream transfer_stream_;

    TimeScheme scheme_;
    std::size_t total_points_;
};

} // namespace ac::cuda
