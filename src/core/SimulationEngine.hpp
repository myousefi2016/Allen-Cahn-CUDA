#pragma once

#include "core/CheckpointManager.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "io/VTKWriter.hpp"

#include <memory>

namespace ac::cuda {
class ISolver;
}

namespace ac {

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
};

} // namespace ac
