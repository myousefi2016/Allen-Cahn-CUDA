#pragma once

#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/ISolver.cuh"
#include "cuda/Kernels.cuh"
#include "cuda/WillmoreKernels.cuh"
#include "cuda/WillmoreSolve.cuh"

#include <memory>
#include <utility>
#include <vector>

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

    /// Phase-field part of an Euler / IMEX step: phi_new from phi_old, u_old
    /// (explicit, or the linearly implicit Willmore solve when
    /// physics.willmore_beta > 0), with phi's BCs applied. Requires params().dt.
    void advance_phi();
    /// Rest of an Euler step after advance_phi(): explicit thermal update,
    /// u's BCs, and the old/new swaps.
    void euler_thermal_and_swap();
    /// Thermal part of imex_begin() after advance_phi().
    void imex_begin_thermal();

    // ── Willmore step building blocks (driven by solve_willmore_step) ──────
    /// Double-precision vectors of the Willmore solve: residual, preconditioned
    /// residual, direction, A times direction, increment (dphi), and tau.
    enum class CgVector { R, Z, P, AP, X, Tau };
    void willmore_omega();               ///< omega from phi_old + homogeneous BCs
    void willmore_rhs_init();            ///< tau, rhs; x = z = p = 0, r = rhs; tau/dt (single)
    void willmore_direction_bc();        ///< homogeneous BCs on p
    void willmore_direction_laplacian(); ///< t = lap p + homogeneous BCs on t
    void willmore_apply();               ///< Ap
    void willmore_update_xr(double alpha);
    void willmore_update_direction(double beta); ///< p = z + beta p
    void willmore_finish();                      ///< phi_new = phi_old + x, phi's BCs
    /// Exact min and max of tau over the interior cells (from willmore_rhs_init).
    [[nodiscard]] std::pair<double, double> willmore_tau_range() const;
    /// Chebyshev preconditioner z = M r (see apply_willmore_preconditioner):
    /// first step from r scaled by `scale` (it also sets the single-precision
    /// operator constants for the plan); the last step (or the first, with
    /// `last`) writes z = x * inv_scale. Steps k = 1 .. degree - 1 each need
    /// the current iterate's Laplacian first (willmore_precondition_laplacian).
    void willmore_precondition_first(const ChebyshevPlan& plan, double scale, double inv_scale,
                                     bool last);
    void willmore_precondition_laplacian();
    void willmore_precondition_step(const ChebyshevPlan& plan, int k, double inv_scale, bool last);
    /// Per-plane interior sums of a*b for local X planes [x_begin, x_end).
    [[nodiscard]] std::vector<double> cg_plane_dot(CgVector a, CgVector b, int x_begin,
                                                   int x_end) const;
    /// A CG vector on every cell, to and from the host (for testing).
    void willmore_upload(CgVector v, const std::vector<double>& host);
    [[nodiscard]] std::vector<double> willmore_download(CgVector v) const;
    /// CG statistics of the last Willmore solve run by this solver's own
    /// advance_phi() (single-domain runs).
    [[nodiscard]] const WillmoreSolveStats& last_willmore_stats() const {
        return last_willmore_stats_;
    }

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

    /// Apply the homogeneous form of phi's BCs (Dirichlet 0, zero flux,
    /// periodic, Robin with gamma = 0): the boundary rule of the Willmore
    /// operator's fields (omega, CG direction and its Laplacian).
    void apply_homogeneous_phi_bc(double* field);
    [[nodiscard]] const double* cg_vector(CgVector v) const;
    [[nodiscard]] double* cg_vector(CgVector v);
    /// Shared body of both cg_vector overloads (const-ness follows `self`).
    template <typename Solver> static auto cg_vector_of(Solver& self, CgVector v);

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

    // Reduction scratch (mutable: logically const methods use it as temporary)
    mutable DeviceField<double> d_reduction_result_;
    mutable DeviceField<double> reduction_scratch_;

    // CUDA streams (mutable: synchronize() and copy_to_host are logically const)
    mutable Stream compute_stream_;
    mutable Stream transfer_stream_;

    TimeScheme scheme_;
    std::size_t total_points_;

    // Willmore regularisation (allocated only when physics.willmore_beta > 0).
    PerFaceBoundary hom_phi_faces_;
    DeviceField<double> w_tau_;                     // tau0 A^2
    DeviceField<double> cg_x_, cg_r_, cg_z_, cg_p_; // increment, residual, z, direction
    DeviceField<double> cg_Ap_, cg_t_;              // A p; lap p (also omega)
    mutable DeviceField<double> plane_dot_scratch_; // one double per X plane
    // Single-precision Chebyshev preconditioner: tau/dt, the scaled residual,
    // two iterates (current and previous/next) and the current one's Laplacian.
    GhostRules32 pc_ghosts_{};
    Operator32 pc_op_{}; // operator constants of the plan being applied
    DeviceField<float> pc_tdt_, pc_f_, pc_x_[2], pc_t_;
    int pc_current_ = 0;                                   // index of the current iterate in pc_x_
    mutable DeviceField<unsigned long long> pc_tau_range_; // min/max tau as double bits
    WillmoreSolveStats last_willmore_stats_;
    bool in_decomposition_ = false; // set by MultiGPUSolver: the solve spans domains
};

} // namespace ac::cuda
