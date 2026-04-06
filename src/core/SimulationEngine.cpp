#include "core/SimulationEngine.hpp"

#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/ISolver.cuh"
#include "cuda/MultiGPUSolver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <spdlog/spdlog.h>

namespace ac {

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
    if (vtk_writer_)
        vtk_writer_->flush();
}

void SimulationEngine::run() {
    auto wall_start = std::chrono::high_resolution_clock::now();

    if (checkpoint_mgr_->has_restart_file()) {
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
    time_loop();

    // Final synchronization
    solver_->synchronize();
    if (vtk_writer_)
        vtk_writer_->flush();

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    spdlog::info("Simulation completed in {:.2f} seconds ({:.2f} minutes)", wall_s, wall_s / 60.0);
}

void SimulationEngine::initialize_fields() {
    Real r0 = config_.initial.seed_radius;
    Real delta = config_.physics.delta;
    int Nx = grid_.Nx(), Ny = grid_.Ny(), Nz = grid_.Nz();
    Real cx = 0.5 * Nx, cy = 0.5 * Ny, cz = 0.5 * Nz;

    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));

                phi_host_(x, y, z) = (r < r0) ? 1.0 : -1.0;
                u_host_(x, y, z) = (r < r0) ? 0.0 : -delta * (1.0 - std::exp(-(r - r0)));
            }
        }
    }

    spdlog::info("Initial conditions: spherical seed at center, r0={}", r0);
}

void SimulationEngine::initialize_from_checkpoint() {
    auto data = checkpoint_mgr_->restore();
    phi_host_ = std::move(data.phi);
    u_host_ = std::move(data.u);
    start_step_ = data.step;
    start_time_ = data.time;
    config_.time.dt = data.dt;

    spdlog::info("Restarted from checkpoint: step={}, time={:.4f}", start_step_, start_time_);
}

void SimulationEngine::time_loop() {
    double dt = config_.time.dt;
    double time = start_time_;
    int max_steps = config_.time.max_steps;

    cuda::Event timer_start(cudaEventDefault);
    cuda::Event timer_stop(cudaEventDefault);

    for (int step = start_step_ + 1; step <= max_steps; ++step) {
        timer_start.record(solver_->stream());

        // Adaptive time stepping
        if (config_.time.adaptive && step > start_step_ + 1) {
            dt = adapt_time_step(dt);
        }

        // Perform one time step
        solver_->step(dt);
        time += dt;

        timer_stop.record(solver_->stream());
        timer_stop.synchronize();
        float step_ms = timer_stop.elapsed_ms(timer_start);

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
    }
}

void SimulationEngine::output_step(int step, double time) {
    solver_->copy_phi_to_host(phi_host_);
    solver_->copy_u_to_host(u_host_);
    vtk_writer_->write_async(step, time, phi_host_, u_host_);
}

void SimulationEngine::checkpoint_step(int step, double time, double dt) {
    solver_->copy_phi_to_host(phi_host_);
    solver_->copy_u_to_host(u_host_);
    checkpoint_mgr_->save(step, time, dt, phi_host_, u_host_);
}

double SimulationEngine::adapt_time_step(double current_dt) {
    double max_dphi = solver_->compute_max_dphi();
    if (max_dphi < 1e-30)
        return current_dt;

    double target = config_.time.adaptive_tolerance;
    double ratio = target / max_dphi;
    double new_dt = current_dt * std::min(1.5, std::max(0.5, 0.9 * ratio));

    // CFL constraint
    Real min_dx = std::min({config_.grid.dx, config_.grid.dy, config_.grid.dz});
    Real cfl_dt = config_.time.cfl_safety * min_dx * min_dx / (2.0 * config_.physics.D * 3.0);
    new_dt = std::min(new_dt, cfl_dt);

    new_dt = std::clamp(new_dt, config_.time.dt_min, config_.time.dt_max);

    if (std::abs(new_dt - current_dt) / current_dt > 0.1) {
        spdlog::debug("Adaptive dt: {:.6e} -> {:.6e} (max_dphi={:.6e})", current_dt, new_dt,
                      max_dphi);
    }

    return new_dt;
}

} // namespace ac
