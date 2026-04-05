#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"

#include <spdlog/spdlog.h>
#include <memory>
#include <vector>

namespace ac::cuda {

/// Multi-GPU solver using domain decomposition along the X axis.
/// Each GPU owns a sub-domain with halo regions for stencil overlap.
class MultiGPUSolver {
public:
    explicit MultiGPUSolver(const SimulationConfig& config)
        : config_(config)
    {
        const auto& device_ids = config.gpu.device_ids;
        int num_gpus = static_cast<int>(device_ids.size());

        if (num_gpus < 2) {
            spdlog::warn("MultiGPUSolver created with {} GPUs, falling back to single GPU",
                         num_gpus);
        }

        // Check peer access
        for (int i = 0; i < num_gpus; ++i) {
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    int can_access = 0;
                    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access,
                                                        device_ids[i],
                                                        device_ids[j]));
                    if (can_access) {
                        CUDA_CHECK(cudaSetDevice(device_ids[i]));
                        cudaDeviceEnablePeerAccess(device_ids[j], 0);
                        // Ignore error if already enabled
                    }
                }
            }
        }

        // Decompose domain along X axis
        int total_Nx = config.grid.Nx;
        int halo = (config.stencil == StencilType::Isotropic27Point) ? 1 : 1;

        for (int g = 0; g < num_gpus; ++g) {
            GPUDomain domain;
            domain.device_id = device_ids[g];
            domain.x_start = g * (total_Nx / num_gpus);
            domain.x_end = (g == num_gpus - 1) ? total_Nx
                                                 : (g + 1) * (total_Nx / num_gpus);
            domain.local_Nx = domain.x_end - domain.x_start + 2 * halo;
            domain.halo = halo;

            // Create a sub-config for this domain
            SimulationConfig sub_config = config;
            sub_config.grid.Nx = domain.local_Nx;
            sub_config.gpu.device_ids = {domain.device_id};
            sub_config.gpu.multi_gpu = false;

            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver = std::make_unique<CudaSolver>(sub_config);

            // Allocate halo exchange buffers
            std::size_t halo_size = static_cast<std::size_t>(halo) *
                                    config.grid.Ny * config.grid.Nz;
            domain.halo_send_lo = DeviceField<double>(halo_size);
            domain.halo_send_hi = DeviceField<double>(halo_size);
            domain.halo_recv_lo = DeviceField<double>(halo_size);
            domain.halo_recv_hi = DeviceField<double>(halo_size);

            domains_.push_back(std::move(domain));
        }

        spdlog::info("MultiGPUSolver initialized with {} GPUs, halo={}", num_gpus, halo);
    }

    void initialize(const FieldData& phi0, const FieldData& u0)
    {
        for (auto& domain : domains_) {
            // Extract sub-domain data from full fields
            Grid sub_grid(
                Dim3{domain.local_Nx, config_.grid.Ny, config_.grid.Nz},
                Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz}
            );
            FieldData phi_sub(sub_grid, "phi_sub");
            FieldData u_sub(sub_grid, "u_sub");

            int src_start = domain.x_start - domain.halo;
            for (int lx = 0; lx < domain.local_Nx; ++lx) {
                int gx = src_start + lx;
                gx = std::max(0, std::min(gx, config_.grid.Nx - 1));
                for (int y = 0; y < config_.grid.Ny; ++y) {
                    for (int z = 0; z < config_.grid.Nz; ++z) {
                        phi_sub(lx, y, z) = phi0(gx, y, z);
                        u_sub(lx, y, z) = u0(gx, y, z);
                    }
                }
            }

            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->initialize(phi_sub, u_sub);
        }
    }

    void step(double dt)
    {
        // Exchange halos between neighboring GPUs
        exchange_halos();

        // Step each sub-domain
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step(dt);
        }
    }

    void synchronize()
    {
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->synchronize();
        }
    }

private:
    struct GPUDomain {
        int device_id = 0;
        int x_start = 0;
        int x_end = 0;
        int local_Nx = 0;
        int halo = 1;
        std::unique_ptr<CudaSolver> solver;
        DeviceField<double> halo_send_lo, halo_send_hi;
        DeviceField<double> halo_recv_lo, halo_recv_hi;
    };

    void exchange_halos()
    {
        // For each pair of neighboring GPUs, exchange boundary data.
        // Uses cudaMemcpyPeerAsync for NVLink or PCIe transfers.
        for (std::size_t g = 0; g + 1 < domains_.size(); ++g) {
            auto& left = domains_[g];
            auto& right = domains_[g + 1];

            // Left sends its high boundary to right's low halo
            // Right sends its low boundary to left's high halo
            // This is a simplified placeholder; full implementation would
            // extract boundary slices from the solver's internal fields.
            CUDA_CHECK(cudaMemcpyPeerAsync(
                right.halo_recv_lo.data(), right.device_id,
                left.halo_send_hi.data(), left.device_id,
                left.halo_send_hi.bytes(), nullptr));

            CUDA_CHECK(cudaMemcpyPeerAsync(
                left.halo_recv_hi.data(), left.device_id,
                right.halo_send_lo.data(), right.device_id,
                right.halo_send_lo.bytes(), nullptr));
        }

        // Synchronize all devices
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    SimulationConfig config_;
    std::vector<GPUDomain> domains_;
};

} // namespace ac::cuda
