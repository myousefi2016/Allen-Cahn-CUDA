#include "core/SimulationEngine.hpp"
#include "core/StabilityLimits.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/ISolver.cuh"
#include "cuda/MultiGPUSolver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ac {

std::atomic<bool> g_shutdown_requested{false};
std::atomic<int> g_shutdown_signal{0};
// Both are written from a signal handler, which is only async-signal-safe
// for lock-free atomics.
static_assert(std::atomic<bool>::is_always_lock_free && std::atomic<int>::is_always_lock_free);

namespace {

/// The state a field check sees, for the cause its error names.
enum class FieldOrigin {
    Restored, ///< a checkpoint's fields, as read
    Start,    ///< the state the run starts from, after its boundary fill, before any step
    Stepped,  ///< the state after one or more time steps
};

/// The cell of linear index i (FieldData's layout: z fastest) and where it lies: an
/// interior cell, or a boundary cell, whose value the boundary fill sets, on one face, on
/// the edge of two faces or at the corner of three. `boundary` tells which.
std::string describe_cell(const FieldData& field, std::size_t i, bool& boundary) {
    const auto ny = static_cast<std::size_t>(field.Ny());
    const auto nz = static_cast<std::size_t>(field.Nz());
    const int at[3] = {static_cast<int>(i / (ny * nz)), static_cast<int>((i / nz) % ny),
                       static_cast<int>(i % nz)};
    const int n[3] = {field.Nx(), field.Ny(), field.Nz()};
    static constexpr const char* lo[3] = {"x_lo", "y_lo", "z_lo"};
    static constexpr const char* hi[3] = {"x_hi", "y_hi", "z_hi"};
    std::vector<std::string> faces;
    for (int a = 0; a < 3; ++a) {
        if (at[a] == 0)
            faces.emplace_back(lo[a]);
        else if (at[a] == n[a] - 1)
            faces.emplace_back(hi[a]);
    }
    boundary = !faces.empty();
    std::string cell = "(" + std::to_string(at[0]) + ", " + std::to_string(at[1]) + ", " +
                       std::to_string(at[2]) + "), ";
    switch (faces.size()) {
    case 0:
        return cell + "an interior cell";
    case 1:
        return cell + "a boundary cell on face " + faces[0];
    case 2:
        return cell + "a boundary cell on the edge of " + faces[0] + " and " + faces[1];
    default:
        return cell + "a boundary cell at the corner of " + faces[0] + ", " + faces[1] + " and " +
               faces[2];
    }
}

/// Throws if `field` holds a NaN or Inf: a run must stop with an error instead of writing
/// output and checkpoints of garbage (a resume would otherwise pick the corrupted
/// checkpoint as the latest). The error names the first such cell and, from `origin`,
/// what put it there.
void ensure_finite(const FieldData& field, const char* name, int step, FieldOrigin origin) {
    const Real* d = field.data();
    for (std::size_t i = 0; i < field.size(); ++i) {
        if (std::isfinite(d[i]))
            continue;
        bool boundary = false;
        const std::string cell = describe_cell(field, i, boundary);
        std::string cause;
        switch (origin) {
        case FieldOrigin::Restored:
            cause = "the checkpoint holds a non-finite value";
            break;
        case FieldOrigin::Start:
            // No step has run: an interior value is the initial or restored one, a boundary
            // value the boundary fill of it. A face cell takes its face's fill of the
            // adjacent interior value, c inner + gamma/(alpha + beta/h) for Robin, and an
            // edge or corner cell the fills of its faces in turn, so their constants add up
            // (validate() requires each to be finite; docs/THEORY.md 9.1).
            cause = boundary ? "the boundary fill of the start state overflows the double range "
                               "(an edge or corner cell composes the fills of its faces, whose "
                               "constants add up)"
                             : "the start state is not finite";
            break;
        case FieldOrigin::Stepped:
            cause = "the simulation diverged";
            break;
        }
        throw std::runtime_error(std::string(name) + " is not finite at step " +
                                 std::to_string(step) + " in cell " + cell + ": " + cause);
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
      u_host_(grid_, "u"),
      adaptive_dt_cap_(config_.time.cfl_safety *
                       std::min(thermal_dt_limit(config_), phase_field_dt_limit(config_))) {
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

void SimulationEngine::run(Start start) {
    auto wall_start = std::chrono::high_resolution_clock::now();

    if (start == Start::ResumeLatest && checkpoint_mgr_->restart_requested()) {
        throw std::invalid_argument("--resume restores the newest intact checkpoint itself: "
                                    "checkpoint.restart_file must not be set as well");
    }
    std::optional<CheckpointIO::RestoreData> data;
    if (checkpoint_mgr_->restart_requested()) {
        data = checkpoint_mgr_->restore();
    } else if (start == Start::ResumeLatest) {
        data = checkpoint_mgr_->restore_latest();
        if (!data)
            spdlog::info("No checkpoint of this run's grid in {} reads back intact: starting at "
                         "step 0",
                         config_.checkpoint.checkpoint_dir.string());
    }
    if (data)
        initialize_from_checkpoint(std::move(*data));
    else
        initialize_fields();

    // Upload to GPU
    solver_->initialize(phi_host_, u_host_);

    // Write initial state
    if (start_step_ == 0) {
        write_step(0, 0.0, config_.time.dt, true, false);
    }

    // A restored state the saturation guard stopped (its final checkpoint) is not advanced
    // again: the guard checks only every saturation_check_freq steps, so the steps up to
    // its next check would grow the crystal further into the wall, write another frame
    // and checkpoint, and rotate away the checkpoints from before wall contact.
    const bool saturated =
        config_.time.exit_on_saturation && restored_ && check_saturation(start_step_);
    if (saturated) {
        spdlog::warn("The restored state at step {} is saturated (phi > {:.3f} next to a wall): "
                     "nothing to compute",
                     start_step_, config_.time.saturation_threshold);
    }

    // Main time loop
    const int last_step = saturated ? start_step_ : time_loop();

    // Final synchronization
    solver_->synchronize();

    // Leave the final state on the host (phi()/u()) and refuse to report a
    // diverged run as finished even when no output step caught it.
    copy_phi_if_needed(last_step);
    copy_u_if_needed(last_step);
    const FieldOrigin end_state =
        last_step == start_step_ ? FieldOrigin::Start : FieldOrigin::Stepped;
    ensure_finite(phi_host_, "phi", last_step, end_state);
    ensure_finite(u_host_, "u", last_step, end_state);
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

void SimulationEngine::initialize_from_checkpoint(CheckpointIO::RestoreData data) {
    restored_ = true;

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
    ensure_finite(data.phi, "phi", data.step, FieldOrigin::Restored);
    ensure_finite(data.u, "u", data.step, FieldOrigin::Restored);

    phi_host_ = std::move(data.phi);
    u_host_ = std::move(data.u);
    start_step_ = data.step;
    start_time_ = data.time;
    // A fixed-dt run continues with the configured dt: it is the step the
    // configuration chose and validate() checked, and the output, checkpoint
    // and saturation-check frequencies count steps of it. An adaptive run
    // continues from the step it had adapted to when the checkpoint was
    // written, within the bounds every adaptive step obeys (adapt_time_step):
    // [dt_min, dt_max], then the stability cap.
    if (config_.time.adaptive) {
        config_.time.dt = std::min(std::clamp(data.dt, config_.time.dt_min, config_.time.dt_max),
                                   adaptive_dt_cap_);
        if (config_.time.dt != data.dt) {
            spdlog::warn("The checkpoint was written with dt={:.17g}, outside [dt_min, dt_max] or "
                         "above the stability cap {:.17g}; continuing with dt={:.17g}",
                         data.dt, adaptive_dt_cap_, config_.time.dt);
        }
    } else if (data.dt != config_.time.dt) {
        spdlog::warn("The checkpoint was written with dt={:.6e}; continuing with the configured "
                     "dt={:.6e} (time.adaptive is false)",
                     data.dt, config_.time.dt);
    }

    spdlog::info("Restarted from checkpoint: step={}, time={:.4f}, dt={:.6e}", start_step_,
                 start_time_, config_.time.dt);
}

int SimulationEngine::time_loop() {
    double dt = config_.time.dt;
    double time = start_time_;
    int max_steps = config_.time.max_steps;
    const bool sat_guard = config_.time.exit_on_saturation;
    const int sat_freq = config_.time.saturation_check_freq; // >= 1 (validate)

    int last_step = start_step_;
    // Steps start_step_ + 1 .. max_steps. Testing step < max_steps before the increment keeps
    // step within int for every max_steps, INT_MAX included (step <= max_steps; ++step would
    // overflow after INT_MAX).
    for (int step = start_step_; step < max_steps;) {
        ++step;
        // Host clock around step() + synchronize(): a CUDA event can only be
        // recorded on a stream of the device it was created on, and a
        // multi-GPU step spans several devices.
        const auto step_start = std::chrono::steady_clock::now();

        // Adaptive time stepping
        if (config_.time.adaptive && step - start_step_ > 1) {
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

        // A step ends the run on SIGINT/SIGTERM (graceful shutdown) or when the solid
        // reaches a wall (saturation guard); its state is then written as output and as a
        // checkpoint, as on the output and checkpoint steps.
        bool last = false;
        if (g_shutdown_requested.load(std::memory_order_relaxed)) {
            spdlog::warn("Shutdown requested at step {}. Writing output and checkpoint...", step);
            last = true;
            interrupted_ = step < max_steps;
        } else if (sat_guard && step % sat_freq == 0 && check_saturation(step)) {
            spdlog::warn("Saturation detected at step {} (phi > {:.3f} next to a wall). "
                         "Writing final output and checkpoint and exiting cleanly.",
                         step, config_.time.saturation_threshold);
            last = true;
        }
        const bool frame = last || step % config_.output.frequency == 0;
        const bool checkpoint = last || checkpoint_mgr_->should_checkpoint(step);
        if (frame || checkpoint)
            write_step(step, time, dt, frame, checkpoint);
        if (last)
            break;
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

void SimulationEngine::write_step(int step, double time, double dt, bool frame, bool checkpoint) {
    copy_phi_if_needed(step);
    copy_u_if_needed(step);
    const FieldOrigin state = step == start_step_ ? FieldOrigin::Start : FieldOrigin::Stepped;
    ensure_finite(phi_host_, "phi", step, state);
    ensure_finite(u_host_, "u", step, state);
    VTKWriter::Checkpoint save;
    if (checkpoint) {
        save = [this, dt](int s, double t, std::span<const Real> phi, std::span<const Real> u) {
            checkpoint_mgr_->save(s, t, dt, phi, u);
        };
    }
    vtk_writer_->write_async(step, time, phi_host_, u_host_, frame, std::move(save));
}

double SimulationEngine::adapt_time_step(double current_dt, int step) {
    double max_dphi = solver_->compute_max_dphi();
    if (!std::isfinite(max_dphi))
        throw std::runtime_error("max|dphi| is not finite after step " + std::to_string(step - 1) +
                                 ": the simulation diverged");
    // tolerance / max|dphi| is +infinity when phi did not change
    // (adaptive_tolerance > 0 is validated): dt then grows by the largest
    // factor, 1.5, like any step far inside the tolerance.
    const double ratio = config_.time.adaptive_tolerance / max_dphi;
    double new_dt = current_dt * std::min(1.5, std::max(0.5, 0.9 * ratio));

    // The stability cap is applied last, so no other bound (dt_min) can
    // lift dt above it; validate() rejects dt_min above the cap.
    new_dt = std::clamp(new_dt, config_.time.dt_min, config_.time.dt_max);
    new_dt = std::min(new_dt, adaptive_dt_cap_);

    if (std::abs(new_dt - current_dt) / current_dt > 0.1) {
        spdlog::debug("Adaptive dt: {:.6e} -> {:.6e} (max_dphi={:.6e})", current_dt, new_dt,
                      max_dphi);
    }

    return new_dt;
}

} // namespace ac
