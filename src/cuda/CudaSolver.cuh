#pragma once

#include "core/SimulationConfig.hpp"
#include "core/FieldData.hpp"
#include "cuda/ISolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"

#include <memory>

namespace ac::cuda {

/// GPU solver facade. Owns all device memory and launches kernels.
/// Supports Euler, Heun (RK2), RK4, and IMEX time integration.
class CudaSolver : public ISolver {
public:
    explicit CudaSolver(const SimulationConfig& config);
    ~CudaSolver() override = default;

    // ISolver interface
    void initialize(const FieldData& phi0, const FieldData& u0) override;
    void step(double dt) override;
    [[nodiscard]] double compute_max_dphi() const override;
    void copy_phi_to_host(FieldData& out) const override;
    void copy_u_to_host(FieldData& out) const override;
    void apply_boundary_conditions() override;
    void synchronize() const override;
    [[nodiscard]] cudaStream_t stream() const override { return compute_stream_.get(); }

    // Time integration methods
    void step_euler(double dt);
    void step_heun(double dt);
    void step_rk4(double dt);
    void step_imex(double dt);

    /// Get the kernel parameters (for testing).
    [[nodiscard]] const KernelParams& params() const { return params_; }

    /// Direct access to device field pointers (for multi-GPU halo exchange).
    [[nodiscard]] double* phi_data() { return phi_old_.data(); }
    [[nodiscard]] double* u_data() { return u_old_.data(); }
    [[nodiscard]] const double* phi_data() const { return phi_old_.data(); }
    [[nodiscard]] const double* u_data() const { return u_old_.data(); }

    /// Get total number of grid points.
    [[nodiscard]] std::size_t total_points() const { return total_points_; }

    /// Heun stage 2: corrector + average. Called by MultiGPUSolver after inter-stage halo exchange.
    void step_heun_stage2(double dt);

    /// Mutable access to kernel params (for multi-GPU dt updates).
    KernelParams& mutable_params() { return params_; }

    // Grant MultiGPUSolver access to internal fields for halo exchange
    friend class MultiGPUSolver;

private:
    /// Apply uniform BCs to a single field.
    void apply_bc(double* field, const BoundaryConfig& bc);

    /// Apply per-face BCs to a single field.
    void apply_bc_per_face(double* field, const PerFaceBoundary& face_bcs);

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

    // Reduction scratch (mutable: logically const methods use it as temporary)
    mutable DeviceField<double> d_reduction_result_;

    // CUDA streams (mutable: synchronize() and copy_to_host are logically const)
    mutable Stream compute_stream_;
    mutable Stream transfer_stream_;

    TimeScheme scheme_;
    std::size_t total_points_;
};

} // namespace ac::cuda
