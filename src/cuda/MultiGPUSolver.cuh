#pragma once

#include "core/SimulationConfig.hpp"
#include "core/FieldData.hpp"
#include "cuda/ISolver.cuh"
#include "cuda/CudaSolver.cuh"
#include "cuda/DeviceField.cuh"

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
    void copy_phi_to_host(FieldData& out) const override;
    void copy_u_to_host(FieldData& out) const override;
    void apply_boundary_conditions() override;
    void synchronize() const override;
    [[nodiscard]] cudaStream_t stream() const override;

private:
    struct GPUDomain {
        int device_id = 0;
        int x_start = 0;       // Global X start index (exclusive of halo)
        int x_end = 0;         // Global X end index (exclusive of halo)
        int local_Nx = 0;      // Including halo on both sides
        int halo = 0;
        std::unique_ptr<CudaSolver> solver;
        Stream halo_stream;     // Dedicated stream for halo exchange
    };

    /// Extract sub-domain data from global field for initialization.
    void extract_subdomain(const FieldData& global, FieldData& local,
                           const GPUDomain& domain) const;

    /// Exchange halo data between neighboring GPUs for both phi and u.
    void exchange_halos();

    /// Copy a YZ-slab from one device to another.
    void copy_slab(double* dst, int dst_device,
                   const double* src, int src_device,
                   int x_dst, int x_src, int slab_count,
                   int Ny, int Nz, int dst_Nx, int src_Nx,
                   cudaStream_t stream);

    SimulationConfig config_;
    std::vector<GPUDomain> domains_;
    int halo_width_ = 0;
};

} // namespace ac::cuda
