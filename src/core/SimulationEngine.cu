#include "core/SimulationEngine.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/ISolver.cuh"
#include "cuda/MultiGPUSolver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace ac {

std::atomic<bool> g_shutdown_requested{false};
std::atomic<int> g_shutdown_signal{0};
// Both are written from a signal handler, which is only async-signal-safe
// for lock-free atomics.
static_assert(std::atomic<bool>::is_always_lock_free && std::atomic<int>::is_always_lock_free);

namespace {

/// Throws if `field` holds a NaN or Inf: a diverged run must stop with an
/// error instead of writing output and checkpoints of garbage (a resume
/// would otherwise pick the corrupted checkpoint as the latest).
void ensure_finite(const FieldData& field, const char* name, int step) {
    const Real* d = field.data();
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (!std::isfinite(d[i])) {
            throw std::runtime_error(std::string(name) + " is not finite at step " +
                                     std::to_string(step) + " (first at linear index " +
                                     std::to_string(i) + "): the simulation diverged");
        }
    }
}

} // namespace

std::unique_ptr<cuda::ISolver> SimulationEngine::create_solver() {
    if (config_.gpu.multi_gpu && config_.gpu.device_ids.size() >= 2) {
        spdlog::info("Creating MultiGPUSolver with {} GPUs", config_.gpu.device_ids.size());
        return std::make_unique<cuda::MultiGPUSolver>(config_);
    }
    return std::make_unique<cuda::CudaSolver>(config_);
}

SimulationEngine::SimulationEngine(SimulationConfig config)
    : config_(std::move(config)), grid_(config_.make_grid()), phi_host_(grid_, "phi"),
      u_host_(grid_, "u") {
    // Create CUDA solver (single or multi-GPU based on config)
    solver_ = create_solver();

    // Create VTK writer
    vtk_writer_ = std::make_unique<VTKWriter>(grid_, config_.output);

    // Create checkpoint manager
    checkpoint_mgr_ = std::make_unique<CheckpointManager>(config_.checkpoint, grid_);

    spdlog::info("SimulationEngine created: {}x{}x{} grid, {} steps", grid_.Nx(), grid_.Ny(),
                 grid_.Nz(), config_.time.max_steps);
}

SimulationEngine::~SimulationEngine() {
    // run() already flushed and reported write errors; this only waits for
    // writes left by a run() that threw. A destructor must not throw.
    if (vtk_writer_) {
        try {
            vtk_writer_->flush();
        } catch (const std::exception& e) {
            spdlog::error("Output write failed: {}", e.what());
        }
    }
}

void SimulationEngine::run() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    if (checkpoint_mgr_->restart_requested()) {
        initialize_from_checkpoint();
    } else {
        initialize_fields();
    }

    // Upload to GPU
    solver_->initialize(phi_host_, u_host_);

    // Write initial state
    if (start_step_ == 0) {
        output_step(0, 0.0);
    }

    // Main time loop
    const int last_step = time_loop();

    // Final synchronization
    solver_->synchronize();

    // Leave the final state on the host (phi()/u()) and refuse to report a
    // diverged run as finished even when no output step caught it.
    copy_phi_if_needed(last_step);
    copy_u_if_needed(last_step);
    ensure_finite(phi_host_, "phi", last_step);
    ensure_finite(u_host_, "u", last_step);
    if (vtk_writer_)
        vtk_writer_->flush();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    spdlog::info("Simulation completed in {:.2f} seconds ({:.2f} minutes)", wall_s, wall_s / 60.0);
}

void SimulationEngine::initialize_fields() {
    const Real r0 = config_.initial.seed_radius; // physical length, same units as W0 and dx
    const Real delta = config_.physics.delta;
    const Real W0 = config_.physics.W0;
    const int Nx = grid_.Nx(), Ny = grid_.Ny(), Nz = grid_.Nz();
    const Real dx = grid_.dx(), dy = grid_.dy(), dz = grid_.dz();
    // Grid points sit at i*dx for i in [0, N-1], so the domain midpoint is
    // 0.5*(N-1) in index units; centring the seed there keeps the problem
    // mirror-symmetric about every axis.
    const Real cx = 0.5 * (Nx - 1), cy = 0.5 * (Ny - 1), cz = 0.5 * (Nz - 1);
    const Real inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * W0);

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                // Distances in physical units: the PDE's lengths (W0, dx) are
                // physical, and the equilibrium profile of
                // W0^2 phi'' + phi - phi^3 = 0 is -tanh(s / (sqrt(2) W0)).
                Real rx = (x - cx) * dx, ry = (y - cy) * dy, rz = (z - cz) * dz;
                Real r = std::sqrt(rx * rx + ry * ry + rz * rz);

                // Equilibrium tanh interface profile (Karma & Rappel 1998):
                // φ = +1 (solid) inside, φ = -1 (liquid) outside.
                phi_host_(x, y, z) = -std::tanh((r - r0) * inv_sqrt2_W0);

                // Uniform undercooling: u = -Δ everywhere.
                u_host_(x, y, z) = -delta;
            }
        }
    }

    spdlog::info("Initial conditions: tanh-profile seed, r0={}, W0={}, u=-{:.3f}", r0, W0, delta);
}

void SimulationEngine::initialize_from_checkpoint() {
    auto data = checkpoint_mgr_->restore();

    if (data.grid.Nx() != grid_.Nx() || data.grid.Ny() != grid_.Ny() ||
        data.grid.Nz() != grid_.Nz()) {
        throw std::runtime_error("Checkpoint grid dimensions (" + std::to_string(data.grid.Nx()) +
                                 "x" + std::to_string(data.grid.Ny()) + "x" +
                                 std::to_string(data.grid.Nz()) + ") do not match config (" +
                                 std::to_string(grid_.Nx()) + "x" + std::to_string(grid_.Ny()) +
                                 "x" + std::to_string(grid_.Nz()) + ")");
    }
    // The spacing is written from, and parsed into, the same double, so a
    // checkpoint of this configuration matches exactly; any difference is a
    // different physical problem.
    if (data.grid.dx() != grid_.dx() || data.grid.dy() != grid_.dy() ||
        data.grid.dz() != grid_.dz()) {
        throw std::runtime_error("Checkpoint grid spacing (" + std::to_string(data.grid.dx()) +
                                 ", " + std::to_string(data.grid.dy()) + ", " +
                                 std::to_string(data.grid.dz()) + ") does not match config (" +
                                 std::to_string(grid_.dx()) + ", " + std::to_string(grid_.dy()) +
                                 ", " + std::to_string(grid_.dz()) + ")");
    }
    ensure_finite(data.phi, "phi", data.step);
    ensure_finite(data.u, "u", data.step);

    phi_host_ = std::move(data.phi);
    u_host_ = std::move(data.u);
    start_step_ = data.step;
    start_time_ = data.time;
    config_.time.dt = data.dt;

    spdlog::info("Restarted from checkpoint: step={}, time={:.4f}", start_step_, start_time_);
}

int SimulationEngine::time_loop() {
    double dt = config_.time.dt;
    double time = start_time_;
    int max_steps = config_.time.max_steps;
    const bool sat_guard = config_.time.exit_on_saturation;
    const int sat_freq = std::max(1, config_.time.saturation_check_freq);

    int last_step = start_step_;
    for (int step = start_step_ + 1; step <= max_steps; ++step) {
        // Host clock around step() + synchronize(): a CUDA event can only be
        // recorded on a stream of the device it was created on, and a
        // multi-GPU step spans several devices.
        const auto step_start = std::chrono::steady_clock::now();

        // Adaptive time stepping
        if (config_.time.adaptive && step > start_step_ + 1) {
            dt = adapt_time_step(dt, step);
        }

        // Perform one time step
        solver_->step(dt);
        time += dt;
        last_step = step;

        solver_->synchronize();
        const double step_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
                .count();

        // Periodic logging
        if (step % 100 == 0 || step == max_steps) {
            spdlog::info("Step {}/{}: time={:.6f}, dt={:.6e}, step_time={:.2f}ms", step, max_steps,
                         time, dt, step_ms);
        }

        // Output
        if (step % config_.output.frequency == 0) {
            output_step(step, time);
        }

        // Checkpoint
        if (checkpoint_mgr_->should_checkpoint(step)) {
            checkpoint_step(step, time, dt);
        }

        // Graceful shutdown on SIGINT/SIGTERM
        if (g_shutdown_requested.load(std::memory_order_relaxed)) {
            spdlog::warn("Shutdown requested at step {}. Writing checkpoint and flushing output...",
                         step);
            if (!checkpoint_mgr_->should_checkpoint(step))
                checkpoint_step(step, time, dt);
            if (step % config_.output.frequency != 0)
                output_step(step, time);
            break;
        }

        // Saturation guard
        if (sat_guard && step % sat_freq == 0 && check_saturation(step)) {
            spdlog::warn("Saturation detected at step {} (phi > {:.3f} on a boundary slab). "
                         "Writing final checkpoint and exiting cleanly.",
                         step, config_.time.saturation_threshold);
            if (!checkpoint_mgr_->should_checkpoint(step))
                checkpoint_step(step, time, dt);
            if (step % config_.output.frequency != 0)
                output_step(step, time);
            break;
        }
    }
    return last_step;
}

bool SimulationEngine::check_saturation(int step) {
    double bmax = solver_->compute_boundary_max_phi();
    if (std::isnan(bmax))
        throw std::runtime_error("phi is NaN on the boundary at step " + std::to_string(step) +
                                 ": the simulation diverged");
    return bmax > config_.time.saturation_threshold;
}

void SimulationEngine::copy_phi_if_needed(int step) {
    if (last_phi_d2h_step_ != step) {
        solver_->copy_phi_to_host(phi_host_);
        last_phi_d2h_step_ = step;
    }
}

void SimulationEngine::copy_u_if_needed(int step) {
    if (last_u_d2h_step_ != step) {
        solver_->copy_u_to_host(u_host_);
        last_u_d2h_step_ = step;
    }
}

void SimulationEngine::output_step(int step, double time) {
    copy_phi_if_needed(step);
    copy_u_if_needed(step);
    ensure_finite(phi_host_, "phi", step);
    ensure_finite(u_host_, "u", step);
    vtk_writer_->write_async(step, time, phi_host_, u_host_);
}

void SimulationEngine::checkpoint_step(int step, double time, double dt) {
    copy_phi_if_needed(step);
    copy_u_if_needed(step);
    ensure_finite(phi_host_, "phi", step);
    ensure_finite(u_host_, "u", step);
    checkpoint_mgr_->save(step, time, dt, phi_host_, u_host_);
}

double SimulationEngine::adapt_time_step(double current_dt, int step) {
    double max_dphi = solver_->compute_max_dphi();
    if (!std::isfinite(max_dphi))
        throw std::runtime_error("max|dphi| is not finite after step " + std::to_string(step - 1) +
                                 ": the simulation diverged");
    if (max_dphi < 1e-30)
        return current_dt;

    double target = config_.time.adaptive_tolerance;
    double ratio = target / max_dphi;
    double new_dt = current_dt * std::min(1.5, std::max(0.5, 0.9 * ratio));

    // CFL constraint: both thermal diffusivity D and phase-field effective
    // diffusivity D_phi = W0^2*A_max^2/tau0 must be stable.
    Real min_dx = std::min({config_.grid.dx, config_.grid.dy, config_.grid.dz});
    Real inv_h2_sum = 1.0 / (min_dx * min_dx) * 3.0;
    Real thermal_cfl = config_.time.cfl_safety / (2.0 * config_.physics.D * inv_h2_sum);
    Real A_max = 1.0 + config_.physics.epsilon;
    Real D_phi = config_.physics.W0 * config_.physics.W0 * A_max * A_max / config_.physics.tau0();
    Real phi_cfl = config_.time.cfl_safety / (2.0 * D_phi * inv_h2_sum);
    Real cfl_dt = std::min(thermal_cfl, phi_cfl);
    new_dt = std::min(new_dt, cfl_dt);

    new_dt = std::clamp(new_dt, config_.time.dt_min, config_.time.dt_max);

    if (std::abs(new_dt - current_dt) / current_dt > 0.1) {
        spdlog::debug("Adaptive dt: {:.6e} -> {:.6e} (max_dphi={:.6e})", current_dt, new_dt,
                      max_dphi);
    }

    return new_dt;
}

} // namespace ac
