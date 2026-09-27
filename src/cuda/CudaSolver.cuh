#pragma once

#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/ISolver.cuh"
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
    [[nodiscard]] double compute_boundary_max_phi() const override;
    void copy_phi_to_host(FieldData& out) const override;
    void copy_u_to_host(FieldData& out) const override;
    void apply_boundary_conditions() override;
    void synchronize() const override;
    /// Compute stream (created on device_ids.front()).
    [[nodiscard]] cudaStream_t stream() const { return compute_stream_.get(); }

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

    /// Heun stage 1 (Euler predictor into phi_tmp_/u_tmp_) and stage 2
    /// (corrector + average + swap). step_heun runs both back to back;
    /// MultiGPUSolver exchanges the predictor's halos between them.
    void step_heun_stage1(double dt);
    void step_heun_stage2(double dt);

    /// RK4 stage 1..4: evaluate k_stage at the stage state (y_n for stage 1,
    /// phi_tmp_/u_tmp_ otherwise), then build the next stage state in the tmp
    /// buffers, or (stage 4) combine into y_{n+1} and swap. step_rk4 runs the
    /// four stages back to back; MultiGPUSolver exchanges tmp halos between them.
    void rk4_stage(int stage, double dt);

    /// IMEX building blocks (step_imex = begin, sweeps with periodic residual
    /// checks, finish), exposed so MultiGPUSolver can exchange the Jacobi
    /// iterate's halos before every sweep and test a global residual.
    void imex_begin(double dt);
    void imex_sweep();
    [[nodiscard]] double imex_residual(int x_begin, int x_end) const;
    void imex_finish();
    static constexpr int kJacobiMaxIters = 200;
    static constexpr int kJacobiCheckFreq = 10;
    static constexpr double kJacobiTol = 1e-10;

    /// max|phi_new - phi_old| over the x-slab [x_begin, x_end).
    [[nodiscard]] double compute_max_dphi(int x_begin, int x_end) const;

    /// max(phi) over the physical boundary cells of the x-slab [x_begin, x_end):
    /// Y/Z faces restricted to the slab plus the X planes x_begin / x_end-1
    /// when x_lo_face / x_hi_face say they are walls.
    [[nodiscard]] double compute_boundary_max_phi(int x_begin, int x_end, bool x_lo_face,
                                                  bool x_hi_face) const;

    /// Mutable access to kernel params (for multi-GPU dt updates).
    KernelParams& mutable_params() { return params_; }

    // Grant MultiGPUSolver access to internal fields for halo exchange
    friend class MultiGPUSolver;

private:
    /// Make device_ids.front() current before any other member is built:
    /// streams and events belong to the device that is current when they are
    /// created, so this must run before compute_stream_/transfer_stream_.
    static int activate_device(const SimulationConfig& config);
    /// Apply the configured phi BCs (per-face when config.boundary.per_face, else uniform).
    void apply_phi_bc(double* field);

    /// Apply the configured u BCs (per-face when config.boundary.per_face, else uniform).
    void apply_u_bc(double* field);

    /// Apply uniform BCs to a single field.
    void apply_bc(double* field, const BoundaryConfig& bc);

    /// Apply per-face BCs to a single field.
    void apply_bc_per_face(double* field, const PerFaceBoundary& face_bcs);

    int device_id_; // first data member: initialised (device selected) before the streams
    SimulationConfig config_;
    KernelParams params_;

    // Primary field buffers (swapped via pointer swap, zero GPU cost)
    DeviceField<double> phi_old_, phi_new_;
    DeviceField<double> u_old_, u_new_;

    // Temporary buffers for higher-order time integration
    DeviceField<double> phi_tmp_, u_tmp_;                   // Heun: predictor stage
    DeviceField<double> k1_phi_, k2_phi_, k3_phi_, k4_phi_; // RK4
    DeviceField<double> k1_u_, k2_u_, k3_u_, k4_u_;         // RK4

    // Force field buffers (needed for non-fused RK stages)
    DeviceField<double> Fx_, Fy_, Fz_;

    // Reduction scratch (mutable: logically const methods use it as temporary)
    mutable DeviceField<double> d_reduction_result_;
    mutable DeviceField<double> reduction_scratch_;

    // CUDA streams (mutable: synchronize() and copy_to_host are logically const)
    mutable Stream compute_stream_;
    mutable Stream transfer_stream_;

    TimeScheme scheme_;
    std::size_t total_points_;
};

} // namespace ac::cuda
