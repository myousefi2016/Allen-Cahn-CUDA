#include "cuda/CudaUtils.cuh"
#include "cuda/MultiGPUSolver.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace ac::cuda {

namespace {

/// Effective BC of `face` (0 = x_lo, 1 = x_hi) for one field of the global config.
const BoundaryConfig& global_face(const BoundaryParams& b, bool phi, int face) {
    if (b.per_face)
        return (phi ? b.phi_faces : b.u_faces).faces[static_cast<std::size_t>(face)];
    return phi ? b.phi_bc : b.u_bc;
}

/// max over domains that propagates NaN (std::max drops a NaN operand).
double nan_aware_max(double a, double b) {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    return std::max(a, b);
}

} // namespace

MultiGPUSolver::MultiGPUSolver(const SimulationConfig& config) : config_(config) {
    const auto& device_ids = config.gpu.device_ids;
    const int num_domains = static_cast<int>(device_ids.size());

    if (num_domains < 2) {
        throw std::runtime_error("MultiGPUSolver requires at least 2 domains, got " +
                                 std::to_string(num_domains));
    }

    // Halos are as wide as the stencils reach (Allen-Cahn face fluxes,
    // thermal Laplacian, Jacobi sweep).
    halo_width_ = kStencilReach;

    const int total_Nx = config.grid.Nx;
    const int base_chunk = total_Nx / num_domains;
    // Every domain must own at least kMinOwnedPlanes planes: neighbour halos
    // are copied from owned planes (>= halo_width_), and the BC at an X wall
    // reads the plane next to it, which the first/last domain must own, as
    // must be the periodic wrap sources, global planes 1 and Nx-2 (>= 2).
    if (base_chunk < kMinOwnedPlanes) {
        throw std::invalid_argument("grid.Nx=" + std::to_string(total_Nx) + " is too small for " +
                                    std::to_string(num_domains) +
                                    " domains: each domain needs at least " +
                                    std::to_string(kMinOwnedPlanes) + " owned X planes");
    }

    // Peer access between distinct devices (several domains may share one).
    for (int i = 0; i < num_domains; ++i) {
        for (int j = 0; j < num_domains; ++j) {
            if (device_ids[i] == device_ids[j])
                continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]));
            if (can_access) {
                CUDA_CHECK(cudaSetDevice(device_ids[i]));
                auto err = cudaDeviceEnablePeerAccess(device_ids[j], 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    (void)cudaGetLastError(); // clear the sticky-free "already enabled" status
                } else {
                    CUDA_CHECK(err);
                }
            } else {
                spdlog::warn("Peer access not available between GPU {} and GPU {}; "
                             "cudaMemcpyPeerAsync will stage halo copies through the host",
                             device_ids[i], device_ids[j]);
            }
        }
    }

    // Periodic X cannot be applied inside a sub-domain (its opposite face lives
    // on another domain): the wrap is done by exchange() instead.
    wrap_phi_lo_ = global_face(config.boundary, true, 0).type == BCType::Periodic;
    wrap_phi_hi_ = global_face(config.boundary, true, 1).type == BCType::Periodic;
    wrap_u_lo_ = global_face(config.boundary, false, 0).type == BCType::Periodic;
    wrap_u_hi_ = global_face(config.boundary, false, 1).type == BCType::Periodic;

    try {
        build_domains();
    } catch (...) {
        // The destructor does not run for a partially constructed object.
        release_domains();
        throw;
    }

    spdlog::info("MultiGPUSolver initialized: {} domains, halo_width={}, total_Nx={}", num_domains,
                 halo_width_, total_Nx);
    for (const auto& d : domains_) {
        spdlog::info("  GPU {}: owns x=[{}, {}), local_Nx={} (halos {}+{})", d.device_id, d.x_start,
                     d.x_end, d.local_Nx, d.left_halo, d.right_halo);
    }
}

void MultiGPUSolver::build_domains() {
    const auto& device_ids = config_.gpu.device_ids;
    const int num_domains = static_cast<int>(device_ids.size());
    const int total_Nx = config_.grid.Nx;
    const int base_chunk = total_Nx / num_domains;
    const int remainder = total_Nx % num_domains;

    BoundaryConfig neumann_bc;
    neumann_bc.type = BCType::Neumann;
    neumann_bc.flux = 0.0;

    int x_offset = 0;
    for (int g = 0; g < num_domains; ++g) {
        // Select the device first: GPUDomain's stream and event are created on
        // the current device.
        CUDA_CHECK(cudaSetDevice(device_ids[g]));
        GPUDomain domain;
        domain.device_id = device_ids[g];

        // Distribute remainder across first 'remainder' domains
        const int chunk = base_chunk + (g < remainder ? 1 : 0);
        domain.x_start = x_offset;
        domain.x_end = x_offset + chunk;
        domain.left_halo = (g > 0) ? halo_width_ : 0;
        domain.right_halo = (g < num_domains - 1) ? halo_width_ : 0;
        domain.local_Nx = chunk + domain.left_halo + domain.right_halo;
        x_offset += chunk;

        SimulationConfig sub_config = config_;
        sub_config.grid.Nx = domain.local_Nx;
        sub_config.gpu.device_ids = {domain.device_id};
        sub_config.gpu.multi_gpu = false;

        // Sub-solvers run in per-face mode; a uniform global config only
        // carries phi_bc/u_bc, so seed every face from it first.
        if (!config_.boundary.per_face) {
            sub_config.boundary.phi_faces = PerFaceBoundary::uniform(config_.boundary.phi_bc);
            sub_config.boundary.u_faces = PerFaceBoundary::uniform(config_.boundary.u_bc);
        }
        // X faces that face another domain are halo planes, and a periodic
        // global X face is filled by the wrap: in both cases the sub-solver
        // must not impose anything there, and zero-flux Neumann is a benign
        // placeholder that exchange() overwrites before the plane is read.
        auto& pf = sub_config.boundary.phi_faces.faces;
        auto& uf = sub_config.boundary.u_faces.faces;
        if (g > 0 || wrap_phi_lo_)
            pf[0] = neumann_bc;
        if (g < num_domains - 1 || wrap_phi_hi_)
            pf[1] = neumann_bc;
        if (g > 0 || wrap_u_lo_)
            uf[0] = neumann_bc;
        if (g < num_domains - 1 || wrap_u_hi_)
            uf[1] = neumann_bc;
        sub_config.boundary.per_face = true;

        domain.solver = std::make_unique<CudaSolver>(sub_config);
        domains_.push_back(std::move(domain));
    }
}

void MultiGPUSolver::extract_subdomain(const FieldData& global, FieldData& local,
                                       const GPUDomain& domain) const {
    const int Ny = config_.grid.Ny;
    const int Nz = config_.grid.Nz;
    const std::size_t plane = static_cast<std::size_t>(Ny) * static_cast<std::size_t>(Nz);

    for (int lx = 0; lx < domain.local_Nx; ++lx) {
        // Halos only exist on interior sides, so every local plane maps to a
        // real global plane.
        const int gx = domain.x_start - domain.left_halo + lx;
        std::memcpy(local.data() + static_cast<std::size_t>(lx) * plane,
                    global.data() + static_cast<std::size_t>(gx) * plane, plane * sizeof(Real));
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
    // Sub-solvers put Neumann placeholders on halo / periodic planes; restore
    // them so the invariant "halos current after initialize() and step()" holds.
    exchange(Buffers::Current);
    spdlog::debug("MultiGPUSolver: initial conditions distributed to all domains");
}

void MultiGPUSolver::copy_planes(double* dst, int dst_device, int x_dst, const double* src,
                                 int src_device, int x_src, int count, int Ny, int Nz,
                                 cudaStream_t stream) {
    // x-major layout: `count` consecutive YZ planes are one contiguous block.
    const std::size_t plane = static_cast<std::size_t>(Ny) * static_cast<std::size_t>(Nz);
    CUDA_CHECK(cudaMemcpyPeerAsync(dst + static_cast<std::size_t>(x_dst) * plane, dst_device,
                                   src + static_cast<std::size_t>(x_src) * plane, src_device,
                                   static_cast<std::size_t>(count) * plane * sizeof(double),
                                   stream));
}

void MultiGPUSolver::exchange(Buffers which) {
    struct Field {
        double* (*ptr)(CudaSolver&);
        bool wrap_lo;
        bool wrap_hi;
    };
    std::vector<Field> fields;
    switch (which) {
    case Buffers::Current:
        fields.push_back(
            {[](CudaSolver& s) { return s.phi_old_.data(); }, wrap_phi_lo_, wrap_phi_hi_});
        fields.push_back({[](CudaSolver& s) { return s.u_old_.data(); }, wrap_u_lo_, wrap_u_hi_});
        break;
    case Buffers::Stage:
        fields.push_back(
            {[](CudaSolver& s) { return s.phi_tmp_.data(); }, wrap_phi_lo_, wrap_phi_hi_});
        fields.push_back({[](CudaSolver& s) { return s.u_tmp_.data(); }, wrap_u_lo_, wrap_u_hi_});
        break;
    case Buffers::JacobiIterate:
        fields.push_back({[](CudaSolver& s) { return s.u_new_.data(); }, wrap_u_lo_, wrap_u_hi_});
        break;
    }

    const int Ny = config_.grid.Ny;
    const int Nz = config_.grid.Nz;
    const int Nx = config_.grid.Nx;

    // Every halo stream waits until every domain's compute stream has finished
    // writing the source planes (copies read other domains' memory).
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.compute_done.record(domain.solver->stream());
    }
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        for (auto& other : domains_)
            CUDA_CHECK(cudaStreamWaitEvent(domain.halo_stream.get(), other.compute_done.get(), 0));
    }

    auto sync_halo_streams = [&] {
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.halo_stream.synchronize();
        }
    };

    // Phase 1 — periodic X wrap, matching the single-domain periodic BC
    // (field[0] = field[Nx-2], field[Nx-1] = field[1]). It completes before
    // phase 2 so that phase 2 never forwards a stale ghost plane: that could
    // only happen for a first/last domain owning no more than halo_width_
    // planes, which kMinOwnedPlanes excludes today, but the order keeps the
    // exchange correct for any kStencilReach <= kMinOwnedPlanes.
    auto& first = domains_.front();
    auto& last = domains_.back();
    for (const auto& f : fields) {
        if (f.wrap_lo) {
            CUDA_CHECK(cudaSetDevice(first.device_id));
            copy_planes(f.ptr(*first.solver), first.device_id, 0, f.ptr(*last.solver),
                        last.device_id, (Nx - 2) - last.x_start + last.left_halo, 1, Ny, Nz,
                        first.halo_stream.get());
        }
        if (f.wrap_hi) {
            CUDA_CHECK(cudaSetDevice(last.device_id));
            copy_planes(f.ptr(*last.solver), last.device_id, last.local_Nx - 1,
                        f.ptr(*first.solver), first.device_id, 1 - first.x_start + first.left_halo,
                        1, Ny, Nz, last.halo_stream.get());
        }
    }
    sync_halo_streams();

    // Phase 2 — each pair of neighbours swaps its outermost owned planes.
    for (std::size_t g = 0; g + 1 < domains_.size(); ++g) {
        auto& left = domains_[g];
        auto& right = domains_[g + 1];
        CUDA_CHECK(cudaSetDevice(left.device_id));
        for (const auto& f : fields) {
            // left's last halo_width_ owned planes -> right's low halo
            copy_planes(f.ptr(*right.solver), right.device_id, 0, f.ptr(*left.solver),
                        left.device_id, left.owned_end() - halo_width_, halo_width_, Ny, Nz,
                        left.halo_stream.get());
            // right's first halo_width_ owned planes -> left's high halo
            copy_planes(f.ptr(*left.solver), left.device_id, left.owned_end(), f.ptr(*right.solver),
                        right.device_id, right.owned_begin(), halo_width_, Ny, Nz,
                        left.halo_stream.get());
        }
    }
    sync_halo_streams();
}

void MultiGPUSolver::step(double dt) {
    // Invariant on entry: halos and periodic ghost planes of phi_old_/u_old_
    // are current (established by initialize() and by the previous step()).
    switch (config_.time.scheme) {
    case TimeScheme::Euler:
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step_euler(dt);
        }
        break;
    case TimeScheme::Heun:
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step_heun_stage1(dt);
        }
        exchange(Buffers::Stage);
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->step_heun_stage2(dt);
        }
        break;
    case TimeScheme::RK4:
        for (int stage = 1; stage <= 4; ++stage) {
            if (stage > 1)
                exchange(Buffers::Stage);
            for (auto& domain : domains_) {
                CUDA_CHECK(cudaSetDevice(domain.device_id));
                domain.solver->rk4_stage(stage, dt);
            }
        }
        break;
    case TimeScheme::IMEX:
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->imex_begin(dt);
        }
        for (int iter = 0; iter < CudaSolver::kJacobiMaxIters; ++iter) {
            for (auto& domain : domains_) {
                CUDA_CHECK(cudaSetDevice(domain.device_id));
                domain.solver->imex_sweep();
            }
            // Refresh the iterate's halos/ghosts before it is read: by the next
            // sweep's stencil and by the residual on the periodic ghost planes.
            exchange(Buffers::JacobiIterate);
            if ((iter + 1) % CudaSolver::kJacobiCheckFreq == 0) {
                double residual = 0.0;
                for (auto& domain : domains_) {
                    CUDA_CHECK(cudaSetDevice(domain.device_id));
                    residual =
                        nan_aware_max(residual, domain.solver->imex_residual(domain.owned_begin(),
                                                                             domain.owned_end()));
                }
                if (residual < CudaSolver::kJacobiTol)
                    break;
            }
        }
        for (auto& domain : domains_) {
            CUDA_CHECK(cudaSetDevice(domain.device_id));
            domain.solver->imex_finish();
        }
        break;
    }
    exchange(Buffers::Current);
}

double MultiGPUSolver::compute_max_dphi() const {
    double global_max = 0.0;
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        global_max = nan_aware_max(
            global_max, domain.solver->compute_max_dphi(domain.owned_begin(), domain.owned_end()));
    }
    return global_max;
}

double MultiGPUSolver::compute_boundary_max_phi() const {
    // Only physical walls count: inter-domain halo planes are interior cells.
    double global_max = -1e30;
    for (std::size_t g = 0; g < domains_.size(); ++g) {
        const auto& domain = domains_[g];
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        global_max = nan_aware_max(global_max, domain.solver->compute_boundary_max_phi(
                                                   domain.owned_begin(), domain.owned_end(), g == 0,
                                                   g + 1 == domains_.size()));
    }
    return global_max;
}

void MultiGPUSolver::gather(FieldData& out, bool phi) const {
    const int Ny = config_.grid.Ny;
    const int Nz = config_.grid.Nz;
    const std::size_t plane = static_cast<std::size_t>(Ny) * static_cast<std::size_t>(Nz);

    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        Grid sub_grid(Dim3{domain.local_Nx, Ny, Nz},
                      Spacing{config_.grid.dx, config_.grid.dy, config_.grid.dz});
        FieldData sub(sub_grid, phi ? "phi_sub" : "u_sub");
        if (phi)
            domain.solver->copy_phi_to_host(sub);
        else
            domain.solver->copy_u_to_host(sub);

        const int owned = domain.x_end - domain.x_start;
        std::memcpy(out.data() + static_cast<std::size_t>(domain.x_start) * plane,
                    sub.data() + static_cast<std::size_t>(domain.owned_begin()) * plane,
                    static_cast<std::size_t>(owned) * plane * sizeof(Real));
    }
}

void MultiGPUSolver::copy_phi_to_host(FieldData& out) const {
    gather(out, true);
}

void MultiGPUSolver::copy_u_to_host(FieldData& out) const {
    gather(out, false);
}

void MultiGPUSolver::apply_boundary_conditions() {
    for (auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->apply_boundary_conditions();
    }
    exchange(Buffers::Current);
}

void MultiGPUSolver::synchronize() const {
    for (const auto& domain : domains_) {
        CUDA_CHECK(cudaSetDevice(domain.device_id));
        domain.solver->synchronize();
        domain.halo_stream.synchronize();
    }
}

MultiGPUSolver::~MultiGPUSolver() {
    release_domains();
}

void MultiGPUSolver::release_domains() noexcept {
    // Release each domain's buffers, streams and event with its own device
    // current, in reverse construction order. No CUDA_CHECK: this runs from
    // the destructor and from the constructor's error path.
    while (!domains_.empty()) {
        (void)cudaSetDevice(domains_.back().device_id);
        domains_.pop_back();
    }
}

} // namespace ac::cuda
