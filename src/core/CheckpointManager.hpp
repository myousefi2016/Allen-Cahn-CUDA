#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "io/CheckpointIO.hpp"

#include <filesystem>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ac {

/// Manages rolling checkpoints with configurable retention policy.
class CheckpointManager {
public:
    explicit CheckpointManager(const CheckpointParams& params, const Grid& grid);

    /// Check if this step should trigger a checkpoint.
    [[nodiscard]] bool should_checkpoint(int step) const;

    /// Save a checkpoint, enforcing the rolling retention policy.
    void save(int step, double time, double dt, std::span<const Real> phi, std::span<const Real> u);
    void save(int step, double time, double dt, const FieldData& phi, const FieldData& u) {
        save(step, time, dt, std::span<const Real>(phi.data(), phi.size()),
             std::span<const Real>(u.data(), u.size()));
    }

    /// Restore from checkpoint.restart_file if it is set, else from the newest
    /// checkpoint in checkpoint_dir of this run's grid that reads back intact
    /// (older ones are tried past a damaged one or one of another grid).
    [[nodiscard]] CheckpointIO::RestoreData restore() const;

    /// The newest checkpoint in checkpoint_dir of this run's grid that reads back intact
    /// (older ones are tried past a damaged one or one of another grid), or none.
    [[nodiscard]] std::optional<CheckpointIO::RestoreData> restore_latest() const;

    /// True when checkpoint.restart_file is configured. The file is not
    /// checked here: restore() throws if it is missing, so a mistyped path
    /// stops the run instead of silently cold-starting from step 0.
    [[nodiscard]] bool restart_requested() const;

private:
    /// Checkpoint files in checkpoint_dir (checkpoint_<step>.acbin), by step.
    [[nodiscard]] std::vector<std::pair<int, std::filesystem::path>> list_checkpoints() const;

    /// After writing `newest_step`, among the checkpoints of this run's grid
    /// (header Nx, Ny, Nz, dx, dy, dz): keep it and the keep_last - 1 highest
    /// older steps; delete the rest, including any step above newest_step
    /// (left by an earlier run past this run's restart point). A checkpoint
    /// of another grid, or without a readable header, cannot belong to this
    /// trajectory and is left alone (with one warning per manager).
    void enforce_retention(int newest_step);
    /// Whether a checkpoint header has this run's grid (Nx, Ny, Nz, dx, dy, dz).
    [[nodiscard]] bool of_this_grid(const CheckpointIO::Header& header) const;
    bool warned_foreign_ = false;

    CheckpointParams params_;
    Grid grid_;
};

} // namespace ac
