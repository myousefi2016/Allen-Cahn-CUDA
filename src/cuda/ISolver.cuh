#pragma once

#include "core/FieldData.hpp"

namespace ac::cuda {

/// Abstract interface for GPU solvers (single-GPU and multi-GPU).
class ISolver {
public:
    virtual ~ISolver() = default;

    /// Upload initial conditions to the GPU.
    virtual void initialize(const FieldData& phi0, const FieldData& u0) = 0;

    /// Perform one complete time step.
    virtual void step(double dt) = 0;

    /// Compute max |phi_new - phi_old| for adaptive time stepping.
    [[nodiscard]] virtual double compute_max_dphi() const = 0;

    /// Copy phi from device to host.
    virtual void copy_phi_to_host(FieldData& out) const = 0;

    /// Copy u from device to host.
    virtual void copy_u_to_host(FieldData& out) const = 0;

    /// Apply boundary conditions to both fields.
    virtual void apply_boundary_conditions() = 0;

    /// Synchronize all GPU work.
    virtual void synchronize() const = 0;

    /// Get compute stream handle (primary GPU for multi-GPU).
    [[nodiscard]] virtual cudaStream_t stream() const = 0;
};

} // namespace ac::cuda
