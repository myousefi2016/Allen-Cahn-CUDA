#include "cuda/CudaUtils.cuh"
#include "cuda/MultiGPUSolver.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <spdlog/spdlog.h>

namespace ac::cuda {

MultiGPUSolver::MultiGPUSolver(const SimulationConfig& config) : config_(config) {
    const auto& device_ids = config.gpu.device_ids;
    int num_gpus = static_cast<int>(device_ids.size());

    if (num_gpus < 2) {
        throw std::runtime_error("MultiGPUSolver requires at least 2 GPUs, got " +
                                 std::to_string(num_gpus));
    }

    // Fused Allen-Cahn kernel computes gradients at neighbor points,
    // so effective stencil reach is ±2 in each direction.
    halo_width_ = 2;

    // Enable peer access between all GPU pairs
    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j)
                continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]));
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

        // For inter-GPU boundaries, use Neumann (zero-flux) BCs on X faces
        // that abut a neighbor GPU. The halo exchange provides the real data;
        // Neumann just copies the adjacent interior value, which is benign.
        BoundaryConfig neumann_bc;
        neumann_bc.type = BCType::Neumann;
        neumann_bc.flux = 0.0;
        if (g > 0) {
            sub_config.boundary.phi_faces.faces[0] = neumann_bc;  // x_lo
            sub_config.boundary.u_faces.faces[0] = neumann_bc;
        }
        if (g < num_gpus - 1) {
            sub_config.boundary.phi_faces.faces[1] = neumann_bc;  // x_hi
            sub_config.boundary.u_faces.faces[1] = neumann_bc;
        }
        sub_config.boundary.per_face = true;

        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver = std::make_unique<CudaSolver>(sub_config);
        domain.halo_stream = Stream();

        domains_.push_back(std::move(domain));
    }

    spdlog::info("MultiGPUSolver initialized: {} GPUs, halo_width={}, total_Nx={}", num_gpus,
                 halo_width_, total_Nx);
    for (const auto& d : domains_) {
        spdlog::info("  GPU {}: x=[{}, {}), local_Nx={} (with halo)", d.device_id, d.x_start,
                     d.x_end, d.local_Nx);
    }
}

void MultiGPUSolver::extract_subdomain(const FieldData& global, FieldData& local,
                                       const GPUDomain& domain) const {
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

void MultiGPUSolver::initialize(const FieldData& phi0, const FieldData& u0) {
    for (auto& domain : domains_) {
        Grid sub_grid(Dim3{domain.local_Nx, config_.grid.Ny, config_.grid.Nz},
                      Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz});
        FieldData phi_sub(sub_grid, "phi_sub");
        FieldData u_sub(sub_grid, "u_sub");

        extract_subdomain(phi0, phi_sub, domain);
        extract_subdomain(u0, u_sub, domain);

        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->initialize(phi_sub, u_sub);
    }
    spdlog::debug("MultiGPUSolver: initial conditions distributed to all GPUs");
}

void MultiGPUSolver::copy_slab(double* dst, int dst_device, const double* src, int src_device,
                               int x_dst, int x_src, int slab_count, int Ny, int Nz,
                               cudaStream_t stream) {
    // Copy 'slab_count' YZ-planes from src to dst
    // Each YZ-plane is contiguous if memory is laid out as [x][y][z]
    for (int s = 0; s < slab_count; ++s) {
        std::size_t dst_offset = static_cast<std::size_t>(x_dst + s) * Ny * Nz;
        std::size_t src_offset = static_cast<std::size_t>(x_src + s) * Ny * Nz;
        std::size_t bytes = static_cast<std::size_t>(Ny) * Nz * sizeof(double);

        CUDA_CHECK(cudaMemcpyPeerAsync(dst + dst_offset, dst_device, src + src_offset, src_device,
                                       bytes, stream));
    }
}

void MultiGPUSolver::exchange_halos() {
    // Ensure all compute kernels have finished writing field data before
    // halo_stream reads it.  Without this, cudaMemcpyPeerAsync on the
    // halo stream could read stale/in-flight data from compute_stream_.
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.compute_done.record(domain.solver->stream());
        CUDA_CHECK(cudaStreamWaitEvent(domain.halo_stream.get(), domain.compute_done.get(), 0));
    }

    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    // Exchange between each pair of neighboring GPUs
    for (std::size_t g = 0; g + 1 < domains_.size(); ++g) {
        auto& left = domains_[g];
        auto& right = domains_[g + 1];

        // Left's halo_stream reads from right's device memory, so it must
        // wait for right's compute to finish (and vice versa).
        CUDA_CHECK(cudaStreamWaitEvent(left.halo_stream.get(), right.compute_done.get(), 0));
        CUDA_CHECK(cudaStreamWaitEvent(right.halo_stream.get(), left.compute_done.get(), 0));

        int left_Nx = left.local_Nx;

        int left_src_x = left_Nx - 2 * halo_width_;
        int right_dst_x = 0;

        int right_src_x = halo_width_;
        int left_dst_x = left_Nx - halo_width_;

        // Exchange phi
        CUDA_CHECK(cudaSetDevice(left.device_id));
        copy_slab(right.solver->phi_data(), right.device_id, left.solver->phi_data(),
                  left.device_id, right_dst_x, left_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());

        copy_slab(left.solver->phi_data(), left.device_id, right.solver->phi_data(),
                  right.device_id, left_dst_x, right_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());

        // Exchange u
        copy_slab(right.solver->u_data(), right.device_id, left.solver->u_data(), left.device_id,
                  right_dst_x, left_src_x, halo_width_, Ny, Nz, left.halo_stream.get());

        copy_slab(left.solver->u_data(), left.device_id, right.solver->u_data(), right.device_id,
                  left_dst_x, right_src_x, halo_width_, Ny, Nz, left.halo_stream.get());
    }

    // Synchronize all halo streams
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.halo_stream.synchronize();
    }
}

void MultiGPUSolver::exchange_halos_for_tmp() {
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.compute_done.record(domain.solver->stream());
        CUDA_CHECK(cudaStreamWaitEvent(domain.halo_stream.get(), domain.compute_done.get(), 0));
    }

    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    for (std::size_t g = 0; g + 1 < domains_.size(); ++g) {
        auto& left = domains_[g];
        auto& right = domains_[g + 1];

        CUDA_CHECK(cudaStreamWaitEvent(left.halo_stream.get(), right.compute_done.get(), 0));
        CUDA_CHECK(cudaStreamWaitEvent(right.halo_stream.get(), left.compute_done.get(), 0));

        int left_Nx = left.local_Nx;

        int left_src_x = left_Nx - 2 * halo_width_;
        int right_dst_x = 0;
        int right_src_x = halo_width_;
        int left_dst_x = left_Nx - halo_width_;

        CUDA_CHECK(cudaSetDevice(left.device_id));

        // Exchange phi_tmp_
        copy_slab(right.solver->phi_tmp_.data(), right.device_id, left.solver->phi_tmp_.data(),
                  left.device_id, right_dst_x, left_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());
        copy_slab(left.solver->phi_tmp_.data(), left.device_id, right.solver->phi_tmp_.data(),
                  right.device_id, left_dst_x, right_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());

        // Exchange u_tmp_
        copy_slab(right.solver->u_tmp_.data(), right.device_id, left.solver->u_tmp_.data(),
                  left.device_id, right_dst_x, left_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());
        copy_slab(left.solver->u_tmp_.data(), left.device_id, right.solver->u_tmp_.data(),
                  right.device_id, left_dst_x, right_src_x, halo_width_, Ny, Nz,
                  left.halo_stream.get());
    }

    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.halo_stream.synchronize();
    }
}

void MultiGPUSolver::step(double dt) {
    auto scheme = config_.time.scheme;

    if (scheme == TimeScheme::Euler) {
        // Euler: single stage, one halo exchange suffices
        exchange_halos();
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step(dt);
        }
    } else if (scheme == TimeScheme::Heun) {
        // Heun: 2 stages. Need halo exchange before each stage.
        // Stage 1: Euler predictor (uses phi_old/u_old -> writes phi_tmp/u_tmp)
        exchange_halos();
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            auto& s = *domain.solver;
            s.mutable_params().dt = dt;
            launch_allen_cahn_fused(s.phi_old_.data(), s.phi_tmp_.data(), s.u_old_.data(),
                                    s.params_, s.compute_stream_);
            s.apply_bc(s.phi_tmp_.data(), s.config_.boundary.phi_bc);
            launch_thermal_equation(s.u_old_.data(), s.u_tmp_.data(), s.phi_tmp_.data(),
                                    s.phi_old_.data(), s.params_, s.compute_stream_);
            s.apply_bc(s.u_tmp_.data(), s.config_.boundary.u_bc);
        }

        // Exchange halos for the predictor fields (phi_tmp_, u_tmp_)
        exchange_halos_for_tmp();

        // Stage 2: Euler from predicted + Heun average
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step_heun_stage2(dt);
        }
    } else {
        // RK4/IMEX with multi-GPU: fall back to single exchange + warning.
        // Full inter-stage exchange for RK4 (4 stages) is complex.
        // The error is confined to ~2 cells at each inter-GPU boundary.
        if (!multi_stage_warned_) {
            spdlog::warn("Multi-GPU with {} scheme: inter-stage halo exchange not fully "
                         "implemented. Results near GPU boundaries may have reduced accuracy. "
                         "Use Euler or Heun for full multi-GPU correctness.",
                         scheme == TimeScheme::RK4 ? "RK4" : "IMEX");
            multi_stage_warned_ = true;
        }
        exchange_halos();
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step(dt);
        }
    }
}

double MultiGPUSolver::compute_max_dphi() const {
    double global_max = 0.0;
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        global_max = std::max(global_max, domain.solver->compute_max_dphi());
    }
    return global_max;
}

double MultiGPUSolver::compute_boundary_max_phi() const {
    double global_max = -1e30;
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        global_max = std::max(global_max, domain.solver->compute_boundary_max_phi());
    }
    return global_max;
}

void MultiGPUSolver::copy_phi_to_host(FieldData& out) const {
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));

        // Create temporary for sub-domain
        Grid sub_grid(Dim3{domain.local_Nx, Ny, Nz},
                      Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz});
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

void MultiGPUSolver::copy_u_to_host(FieldData& out) const {
    int Ny = config_.grid.Ny;
    int Nz = config_.grid.Nz;

    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));

        Grid sub_grid(Dim3{domain.local_Nx, Ny, Nz},
                      Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz});
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

void MultiGPUSolver::apply_boundary_conditions() {
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->apply_boundary_conditions();
    }
}

void MultiGPUSolver::synchronize() const {
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->synchronize();
    }
}

cudaStream_t MultiGPUSolver::stream() const {
    if (domains_.empty())
        return nullptr;
    return domains_.front().solver->stream();
}

} // namespace ac::cuda
