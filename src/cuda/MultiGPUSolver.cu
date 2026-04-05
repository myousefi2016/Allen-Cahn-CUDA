#include "cuda/MultiGPUSolver.cuh"
#include "cuda/CudaUtils.cuh"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace ac::cuda {

MultiGPUSolver::MultiGPUSolver(const SimulationConfig& config)
    : config_(config)
{
    const auto& device_ids = config.gpu.device_ids;
    int num_gpus = static_cast<int>(device_ids.size());

    if (num_gpus < 2) {
        throw std::runtime_error(
            "MultiGPUSolver requires at least 2 GPUs, got " + std::to_string(num_gpus));
    }

    // Fused Allen-Cahn kernel computes gradients at neighbor points,
    // so effective stencil reach is ±2 in each direction.
    halo_width_ = 2;

    // Enable peer access between all GPU pairs
    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access,
                                                device_ids[i], device_ids[j]));
            if (can_access) {
                CUDA_CHECK(cudaSetDevice(device_ids[i]));
                auto err = cudaDeviceEnablePeerAccess(device_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(err);
                }
            } else {
                spdlog::warn("Peer access not available between GPU {} and GPU {}; "
                             "halo exchange will use staged copies via host",
                             device_ids[i], device_ids[j]);
            }
        }
    }

    // Decompose domain along X axis
    int total_Nx = config.grid.Nx;
    int base_chunk = total_Nx / num_gpus;
    int remainder = total_Nx % num_gpus;

    int x_offset = 0;
    for (int g = 0; g < num_gpus; ++g) {
        GPUDomain domain;
        domain.device_id = device_ids[g];
        domain.halo = halo_width_;

        // Distribute remainder across first 'remainder' GPUs
        int chunk = base_chunk + (g < remainder ? 1 : 0);
        domain.x_start = x_offset;
        domain.x_end = x_offset + chunk;
        domain.local_Nx = chunk + 2 * halo_width_;
        x_offset += chunk;

        // Create sub-config for this domain
        SimulationConfig sub_config = config;
        sub_config.grid.Nx = domain.local_Nx;
        sub_config.gpu.device_ids = {domain.device_id};
        sub_config.gpu.multi_gpu = false;

        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver = std::make_unique<CudaSolver>(sub_config);
        domain.halo_stream = Stream();

        domains_.push_back(std::move(domain));
    }

    spdlog::info("MultiGPUSolver initialized: {} GPUs, halo_width={}, total_Nx={}",
                 num_gpus, halo_width_, total_Nx);
    for (const auto& d : domains_) {
        spdlog::info("  GPU {}: x=[{}, {}), local_Nx={} (with halo)",
                     d.device_id, d.x_start, d.x_end, d.local_Nx);
    }
}

void MultiGPUSolver::extract_subdomain(const FieldData& global, FieldData& local,
                                        const GPUDomain& domain) const
{
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;
    int global_Nx = config_.grid.Nx;

    for (int lx = 0; lx < domain.local_Nx; ++lx) {
        // Map local x to global x (with halo offset)
        int gx = domain.x_start - domain.halo + lx;
        // Clamp to valid global range
        gx = std::clamp(gx, 0, global_Nx - 1);

        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                local(lx, y, z) = global(gx, y, z);
            }
        }
    }
}

void MultiGPUSolver::initialize(const FieldData& phi0, const FieldData& u0)
{
    for (auto& domain : domains_) {
        Grid sub_grid(
            Dim3{domain.local_Nx, config_.grid.Ny, config_.grid.Nz},
            Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz}
        );
        FieldData phi_sub(sub_grid, "phi_sub");
        FieldData u_sub(sub_grid, "u_sub");

        extract_subdomain(phi0, phi_sub, domain);
        extract_subdomain(u0, u_sub, domain);

        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->initialize(phi_sub, u_sub);
    }
    spdlog::debug("MultiGPUSolver: initial conditions distributed to all GPUs");
}

void MultiGPUSolver::copy_slab(double* dst, int dst_device,
                                const double* src, int src_device,
                                int x_dst, int x_src, int slab_count,
                                int Ny, int Nz, int dst_Nx, int src_Nx,
                                cudaStream_t stream)
{
    // Copy 'slab_count' YZ-planes from src to dst
    // Each YZ-plane is contiguous if memory is laid out as [x][y][z]
    for (int s = 0; s < slab_count; ++s) {
        std::size_t dst_offset = static_cast<std::size_t>(x_dst + s) * Ny * Nz;
        std::size_t src_offset = static_cast<std::size_t>(x_src + s) * Ny * Nz;
        std::size_t bytes = static_cast<std::size_t>(Ny) * Nz * sizeof(double);

        CUDA_CHECK(cudaMemcpyPeerAsync(
            dst + dst_offset, dst_device,
            src + src_offset, src_device,
            bytes, stream));
    }
}

void MultiGPUSolver::exchange_halos()
{
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    // Exchange between each pair of neighboring GPUs
    for (std::size_t g = 0; g + 1 < domains_.size(); ++g) {
        auto& left = domains_[g];
        auto& right = domains_[g + 1];

        int left_Nx = left.local_Nx;
        int right_Nx = right.local_Nx;

        // For phi field:
        // Left's right boundary -> Right's left halo
        // Left interior ends at x = left_Nx - halo_width_ - 1
        // Right halo starts at x = 0
        int left_src_x = left_Nx - 2 * halo_width_;  // Start of left's right interior boundary
        int right_dst_x = 0;                           // Start of right's left halo

        // Right's left boundary -> Left's right halo
        int right_src_x = halo_width_;                 // Start of right's left interior boundary
        int left_dst_x = left_Nx - halo_width_;        // Start of left's right halo

        // Exchange phi
        CUDA_CHECK(cudaSetDevice(left.device_id));
        copy_slab(right.solver->phi_data(), right.device_id,
                  left.solver->phi_data(), left.device_id,
                  right_dst_x, left_src_x, halo_width_,
                  Ny, Nz, right_Nx, left_Nx,
                  left.halo_stream.get());

        copy_slab(left.solver->phi_data(), left.device_id,
                  right.solver->phi_data(), right.device_id,
                  left_dst_x, right_src_x, halo_width_,
                  Ny, Nz, left_Nx, right_Nx,
                  left.halo_stream.get());

        // Exchange u
        copy_slab(right.solver->u_data(), right.device_id,
                  left.solver->u_data(), left.device_id,
                  right_dst_x, left_src_x, halo_width_,
                  Ny, Nz, right_Nx, left_Nx,
                  left.halo_stream.get());

        copy_slab(left.solver->u_data(), left.device_id,
                  right.solver->u_data(), right.device_id,
                  left_dst_x, right_src_x, halo_width_,
                  Ny, Nz, left_Nx, right_Nx,
                  left.halo_stream.get());
    }

    // Synchronize all halo streams
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.halo_stream.synchronize();
    }
}

void MultiGPUSolver::step(double dt)
{
    // Exchange halos before stepping
    exchange_halos();

    // Step each sub-domain
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->step(dt);
    }
}

double MultiGPUSolver::compute_max_dphi() const
{
    double global_max = 0.0;
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        global_max = std::max(global_max, domain.solver->compute_max_dphi());
    }
    return global_max;
}

void MultiGPUSolver::copy_phi_to_host(FieldData& out) const
{
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));

        // Create temporary for sub-domain
        Grid sub_grid(
            Dim3{domain.local_Nx, Ny, Nz},
            Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz}
        );
        FieldData sub(sub_grid, "phi_sub");
        domain.solver->copy_phi_to_host(sub);

        // Copy interior (skip halo) to global field
        for (int lx = domain.halo; lx < domain.local_Nx - domain.halo; ++lx) {
            int gx = domain.x_start + (lx - domain.halo);
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    out(gx, y, z) = sub(lx, y, z);
                }
            }
        }
    }
}

void MultiGPUSolver::copy_u_to_host(FieldData& out) const
{
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));

        Grid sub_grid(
            Dim3{domain.local_Nx, Ny, Nz},
            Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz}
        );
        FieldData sub(sub_grid, "u_sub");
        domain.solver->copy_u_to_host(sub);

        for (int lx = domain.halo; lx < domain.local_Nx - domain.halo; ++lx) {
            int gx = domain.x_start + (lx - domain.halo);
            for (int y = 0; y < Ny; ++y) {
                for (int z = 0; z < Nz; ++z) {
                    out(gx, y, z) = sub(lx, y, z);
                }
            }
        }
    }
}

void MultiGPUSolver::apply_boundary_conditions()
{
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->apply_boundary_conditions();
    }
}

void MultiGPUSolver::synchronize() const
{
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->synchronize();
    }
}

cudaStream_t MultiGPUSolver::stream() const
{
    if (domains_.empty()) return nullptr;
    return domains_.front().solver->stream();
}

} // namespace ac::cuda
