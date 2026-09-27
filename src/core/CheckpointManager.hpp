#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "io/CheckpointIO.hpp"

#include <filesystem>
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
    void save(int step, double time, double dt, const FieldData& phi, const FieldData& u);

    /// Restore from the latest or specified checkpoint.
    [[nodiscard]] CheckpointIO::RestoreData restore() const;

    /// True when checkpoint.restart_file is configured. The file is not
    /// checked here: restore() throws if it is missing, so a mistyped path
    /// stops the run instead of silently cold-starting from step 0.
    [[nodiscard]] bool restart_requested() const;

private:
    /// Checkpoint files in checkpoint_dir (checkpoint_<step>.acbin), by step.
    [[nodiscard]] std::vector<std::pair<int, std::filesystem::path>> list_checkpoints() const;

    /// After writing `newest_step`: keep it and the keep_last - 1 highest
    /// older steps; delete the rest, including any step above newest_step
    /// (left by an earlier run past this run's restart point).
    void enforce_retention(int newest_step);

    CheckpointParams params_;
    Grid grid_;
};

} // namespace ac
