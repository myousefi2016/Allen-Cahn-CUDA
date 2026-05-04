#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "io/CheckpointIO.hpp"

#include <deque>
#include <filesystem>

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

    /// Check if a restart file is configured and exists.
    [[nodiscard]] bool has_restart_file() const;

private:
    void scan_existing_checkpoints();
    void enforce_retention();

    CheckpointParams params_;
    Grid grid_;
    std::deque<std::filesystem::path> checkpoint_files_;
};

} // namespace ac
