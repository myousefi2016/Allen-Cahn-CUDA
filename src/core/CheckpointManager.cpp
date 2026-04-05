#include "core/CheckpointManager.hpp"

#include <spdlog/spdlog.h>
#include <filesystem>

namespace ac {

CheckpointManager::CheckpointManager(const CheckpointParams& params, const Grid& grid)
    : params_(params), grid_(grid)
{
    std::filesystem::create_directories(params_.checkpoint_dir);
}

bool CheckpointManager::should_checkpoint(int step) const
{
    return params_.frequency > 0 && step > 0 && (step % params_.frequency == 0);
}

void CheckpointManager::save(int step, double time, double dt,
                              const FieldData& phi, const FieldData& u)
{
    auto path = params_.checkpoint_dir /
        ("checkpoint_" + std::to_string(step) + ".acbin");

    CheckpointIO::write(path, step, time, dt, grid_, phi, u);
    checkpoint_files_.push_back(path);
    enforce_retention();
}

CheckpointIO::RestoreData CheckpointManager::restore() const
{
    if (params_.restart_file.has_value()) {
        return CheckpointIO::read(params_.restart_file.value());
    }

    // Find latest checkpoint in directory
    std::filesystem::path latest;
    int latest_step = -1;

    if (std::filesystem::exists(params_.checkpoint_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(params_.checkpoint_dir)) {
            if (entry.path().extension() == ".acbin" &&
                CheckpointIO::is_valid_checkpoint(entry.path())) {
                // Extract step number from filename
                auto stem = entry.path().stem().string();
                auto pos = stem.rfind('_');
                if (pos != std::string::npos) {
                    int step = std::stoi(stem.substr(pos + 1));
                    if (step > latest_step) {
                        latest_step = step;
                        latest = entry.path();
                    }
                }
            }
        }
    }

    if (latest.empty()) {
        throw std::runtime_error("No checkpoint files found for restart");
    }

    return CheckpointIO::read(latest);
}

bool CheckpointManager::has_restart_file() const
{
    if (params_.restart_file.has_value()) {
        return std::filesystem::exists(params_.restart_file.value());
    }
    return false;
}

void CheckpointManager::enforce_retention()
{
    while (static_cast<int>(checkpoint_files_.size()) > params_.keep_last) {
        auto& oldest = checkpoint_files_.front();
        if (std::filesystem::exists(oldest)) {
            std::filesystem::remove(oldest);
            spdlog::debug("Removed old checkpoint: {}", oldest.string());
        }
        checkpoint_files_.pop_front();
    }
}

} // namespace ac
