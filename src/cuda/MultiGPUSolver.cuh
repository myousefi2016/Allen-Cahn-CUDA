#pragma once

#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/ISolver.cuh"

#include <memory>
#include <vector>

namespace ac::cuda {

/// Multi-GPU solver using domain decomposition along the X axis.
/// Each GPU owns a sub-domain with halo regions for stencil overlap.
class MultiGPUSolver : public ISolver {
public:
    explicit MultiGPUSolver(const SimulationConfig& config);
    ~MultiGPUSolver() override = default;

    // ISolver interface
    void initialize(const FieldData& phi0, const FieldData& u0) override;
    void step(double dt) override;
    [[nodiscard]] double compute_max_dphi() const override;
    [[nodiscard]] double compute_boundary_max_phi() const override;
    void copy_phi_to_host(FieldData& out) const override;
    void copy_u_to_host(FieldData& out) const override;
    void apply_boundary_conditions() override;
    void synchronize() const override;
    [[nodiscard]] cudaStream_t stream() const override;

private:
    struct GPUDomain {
        int device_id = 0;
        int x_start = 0;  // Global X start index (exclusive of halo)
        int x_end = 0;    // Global X end index (exclusive of halo)
        int local_Nx = 0; // Including halo on both sides
        int halo = 0;
        std::unique_ptr<CudaSolver> solver;
        Stream halo_stream; // Dedicated stream for halo exchange
        Event compute_done; // Signalled when compute_stream_ finishes a step
    };

    /// Extract sub-domain data from global field for initialization.
    void extract_subdomain(const FieldData& global, FieldData& local,
                           const GPUDomain& domain) const;

    /// Exchange halo data between neighboring GPUs for both phi and u (phi_old_, u_old_).
    void exchange_halos();

    /// Exchange halo data for the temporary fields (phi_tmp_, u_tmp_) used in Heun stage 2.
    void exchange_halos_for_tmp();

    /// Copy a YZ-slab from one device to another.
    void copy_slab(double* dst, int dst_device, const double* src, int src_device, int x_dst,
                   int x_src, int slab_count, int Ny, int Nz, cudaStream_t stream);

    SimulationConfig config_;
    std::vector<GPUDomain> domains_;
    int halo_width_ = 0;
    bool multi_stage_warned_ = false;
};

} // namespace ac::cuda
