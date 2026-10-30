#pragma once

#include "core/CheckpointManager.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "io/VTKWriter.hpp"

#include <atomic>
#include <memory>

namespace ac::cuda {
class ISolver;
}

namespace ac {

/// Global flag set by SIGINT/SIGTERM handler to request graceful shutdown.
extern std::atomic<bool> g_shutdown_requested;

/// Number of the signal that requested the shutdown (0 if none); main()
/// exits with 128 + signal when the run stopped for it before max_steps
/// (SimulationEngine::exit_status), so callers see an interrupted, incomplete run.
extern std::atomic<int> g_shutdown_signal;

/// Top-level simulation orchestrator.
/// Owns config, grid, solver, I/O writers, and checkpoint manager.
class SimulationEngine {
public:
    explicit SimulationEngine(SimulationConfig config);
    ~SimulationEngine();

    /// Run the full simulation from start (or restart) to completion.
    void run();

    /// Whether run() stopped for a shutdown request before reaching max_steps: then a
    /// checkpoint of the step it stopped at was written and a resume continues the run.
    /// A run that reached max_steps or the saturation guard is complete, also when a
    /// signal arrived during its last step or after it.
    [[nodiscard]] bool interrupted() const { return interrupted_; }

    /// The process exit status after run(): 128 + signal for an interrupted run (the
    /// shell's convention for a run ended by that signal), 0 for a complete one.
    [[nodiscard]] int exit_status(int signal) const {
        return interrupted_ && signal > 0 ? 128 + signal : 0;
    }

    /// Fields on the host: the final state after run() (for testing).
    [[nodiscard]] const FieldData& phi() const { return phi_host_; }
    [[nodiscard]] const FieldData& u() const { return u_host_; }
    [[nodiscard]] const Grid& grid() const { return grid_; }

private:
    void initialize_fields();
    void initialize_from_checkpoint();
    /// Runs the steps; returns the last step executed (start_step_ if none).
    int time_loop();
    /// Enqueue the state of `step` with the writer: as an output frame when `frame`, then
    /// as a checkpoint when `checkpoint`. The writer saves a checkpoint only after every
    /// frame before it is on disk: a resume continues at the checkpoint's step + 1 and
    /// never writes those frames again.
    void write_step(int step, double time, double dt, bool frame, bool checkpoint);
    double adapt_time_step(double current_dt, int step);
    void copy_phi_if_needed(int step);
    void copy_u_if_needed(int step);

    /// Max of phi over the interior cells next to the walls: true if above
    /// config.time.saturation_threshold, i.e. the solid has reached a wall and
    /// the run no longer models growth into an unbounded melt. Throws if the
    /// maximum is NaN (diverged field).
    [[nodiscard]] bool check_saturation(int step);

    /// Create appropriate solver based on config (single-GPU or multi-GPU).
    std::unique_ptr<cuda::ISolver> create_solver();

    SimulationConfig config_;
    Grid grid_;
    FieldData phi_host_;
    FieldData u_host_;

    std::unique_ptr<cuda::ISolver> solver_;
    // Declared before the writer, which saves checkpoints through it on its thread: the
    // writer (and that thread) is destroyed first.
    std::unique_ptr<CheckpointManager> checkpoint_mgr_;
    std::unique_ptr<VTKWriter> vtk_writer_;

    int start_step_ = 0;
    bool interrupted_ = false;
    double start_time_ = 0.0;
    int last_phi_d2h_step_ = -1;
    int last_u_d2h_step_ = -1;
    /// Adaptive steps, the first after a restart included, never exceed
    /// cfl_safety times the explicit stability limits (docs/THEORY.md 10.1);
    /// +infinity when neither applies. The first step of a fresh run uses
    /// time.dt, which validate() requires to be below this cap.
    double adaptive_dt_cap_;
};

} // namespace ac
