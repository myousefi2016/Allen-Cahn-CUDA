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
///
/// Each domain owns a contiguous range of global X planes and carries halo
/// planes only on sides that face another domain, so the first and last
/// domains hold the real global X boundary planes and apply the configured BCs
/// there. Halos (and, for periodic X, the global ghost planes) are refreshed
/// before every stencil evaluation, which makes every scheme produce exactly
/// the single-GPU result. Several domains may share one device.
class MultiGPUSolver : public ISolver {
public:
    explicit MultiGPUSolver(const SimulationConfig& config);
    ~MultiGPUSolver() override;

    MultiGPUSolver(const MultiGPUSolver&) = delete;
    MultiGPUSolver& operator=(const MultiGPUSolver&) = delete;

    // ISolver interface
    void initialize(const FieldData& phi0, const FieldData& u0) override;
    void step(double dt) override;
    [[nodiscard]] double compute_max_dphi() const override;
    [[nodiscard]] double compute_boundary_max_phi() const override;
    void copy_phi_to_host(FieldData& out) const override;
    void copy_u_to_host(FieldData& out) const override;
    void apply_boundary_conditions() override;
    void synchronize() const override;

private:
    struct GPUDomain {
        int device_id = 0;
        int x_start = 0;    // first owned global X plane
        int x_end = 0;      // one past the last owned global X plane
        int left_halo = 0;  // halo planes below the owned range (0 for the first domain)
        int right_halo = 0; // halo planes above the owned range (0 for the last domain)
        int local_Nx = 0;   // owned planes + halos
        std::unique_ptr<CudaSolver> solver;
        Stream halo_stream; // dedicated stream for halo copies
        Event compute_done; // recorded on the solver's compute stream before a copy

        [[nodiscard]] int owned_begin() const { return left_halo; }
        [[nodiscard]] int owned_end() const { return left_halo + (x_end - x_start); }
    };

    /// Which device buffers an exchange refreshes.
    enum class Buffers {
        Current,       ///< phi_old_ and u_old_
        Stage,         ///< phi_tmp_ and u_tmp_ (Heun / RK4 intermediate state)
        JacobiIterate, ///< u_new_ (IMEX iterate)
    };

    /// Build every domain (device, slab, sub-solver); called by the constructor.
    void build_domains();

    /// Destroy the domains, each with its own device current.
    void release_domains() noexcept;

    /// Copy global planes into this domain's local sub-field (initialization).
    void extract_subdomain(const FieldData& global, FieldData& local,
                           const GPUDomain& domain) const;

    /// Gather the owned planes of every domain into a global field.
    void gather(FieldData& out, bool phi) const;

    /// Refresh periodic X ghost planes, then inter-domain halos, for `which`.
    void exchange(Buffers which);

    /// Copy `count` YZ planes between (possibly different) devices.
    static void copy_planes(double* dst, int dst_device, int x_dst, const double* src,
                            int src_device, int x_src, int count, int Ny, int Nz,
                            cudaStream_t stream);

    SimulationConfig config_;
    std::vector<GPUDomain> domains_;
    int halo_width_ = 2;
    bool wrap_phi_lo_ = false, wrap_phi_hi_ = false; // periodic X for phi
    bool wrap_u_lo_ = false, wrap_u_hi_ = false;     // periodic X for u
};

} // namespace ac::cuda
