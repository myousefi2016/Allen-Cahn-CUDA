#include "core/CheckpointManager.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <filesystem>
#include <vector>

namespace ac {

CheckpointManager::CheckpointManager(const CheckpointParams& params, const Grid& grid)
    : params_(params), grid_(grid)
{
    std::filesystem::create_directories(params_.checkpoint_dir);
    scan_existing_checkpoints();
}

void CheckpointManager::scan_existing_checkpoints()
{
    if (!std::filesystem::exists(params_.checkpoint_dir)) return;

    // Collect existing checkpoint files with their step numbers
    std::vector<std::pair<int, std::filesystem::path>> existing;
    for (const auto& entry : std::filesystem::directory_iterator(params_.checkpoint_dir)) {
        if (entry.path().extension() != ".acbin") continue;
        auto stem = entry.path().stem().string();
        auto pos = stem.rfind('_');
        if (pos == std::string::npos) continue;
        try {
            int step = std::stoi(stem.substr(pos + 1));
            existing.emplace_back(step, entry.path());
        } catch (const std::exception&) {
            // Skip malformed filenames
        }
    }

    // Sort by step number so deque is in chronological order
    std::sort(existing.begin(), existing.end());
    for (auto& [step, path] : existing) {
        checkpoint_files_.push_back(std::move(path));
    }
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
                    try {
                        int step = std::stoi(stem.substr(pos + 1));
                        if (step > latest_step) {
                            latest_step = step;
                            latest = entry.path();
                        }
                    } catch (const std::exception&) {
                        spdlog::warn("Skipping checkpoint with malformed name: {}",
                                     entry.path().string());
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
