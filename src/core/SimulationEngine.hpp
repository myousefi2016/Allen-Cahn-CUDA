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

/// Top-level simulation orchestrator.
/// Owns config, grid, solver, I/O writers, and checkpoint manager.
class SimulationEngine {
public:
    explicit SimulationEngine(SimulationConfig config);
    ~SimulationEngine();

    /// Run the full simulation from start (or restart) to completion.
    void run();

    /// Access current simulation state (for testing).
    [[nodiscard]] const FieldData& phi() const { return phi_host_; }
    [[nodiscard]] const FieldData& u() const { return u_host_; }
    [[nodiscard]] const Grid& grid() const { return grid_; }

private:
    void initialize_fields();
    void initialize_from_checkpoint();
    void time_loop();
    void output_step(int step, double time);
    void checkpoint_step(int step, double time, double dt);
    double adapt_time_step(double current_dt);
    void copy_phi_if_needed(int step);
    void copy_u_if_needed(int step);

    /// Inspect the six 1-cell-thick boundary slabs of phi and return true if
    /// any cell has phi > config.time.saturation_threshold. The Allen-Cahn
    /// kernel returns zero force at boundary cells (AllenCahnKernels.cu:52),
    /// which produces an unphysical force-divergence discontinuity at near-
    /// boundary cells once the solid touches the wall — this manifests as
    /// numerical blow-up. Detecting saturation lets us exit cleanly before
    /// that happens.
    [[nodiscard]] bool check_saturation();

    /// Create appropriate solver based on config (single-GPU or multi-GPU).
    std::unique_ptr<cuda::ISolver> create_solver();

    SimulationConfig config_;
    Grid grid_;
    FieldData phi_host_;
    FieldData u_host_;

    std::unique_ptr<cuda::ISolver> solver_;
    std::unique_ptr<VTKWriter> vtk_writer_;
    std::unique_ptr<CheckpointManager> checkpoint_mgr_;

    int start_step_ = 0;
    double start_time_ = 0.0;
    int last_phi_d2h_step_ = -1;
    int last_u_d2h_step_ = -1;
};

} // namespace ac
