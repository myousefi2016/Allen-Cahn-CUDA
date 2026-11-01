#include "core/CheckpointManager.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <string>
#include <system_error>
#include <vector>

namespace ac {

namespace {

/// Step of a file named exactly checkpoint_<digits>.acbin, or -1.
int checkpoint_step(const std::filesystem::path& path) {
    const std::string name = path.filename().string();
    const std::string prefix = "checkpoint_", suffix = ".acbin";
    if (name.size() <= prefix.size() + suffix.size() || name.rfind(prefix, 0) != 0 ||
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + name.size() - suffix.size();
    int step = -1;
    auto [ptr, ec] = std::from_chars(first, last, step);
    if (ec != std::errc() || ptr != last || step < 0)
        return -1;
    return step;
}

} // namespace

CheckpointManager::CheckpointManager(const CheckpointParams& params, const Grid& grid)
    : params_(params), grid_(grid) {
    std::filesystem::create_directories(params_.checkpoint_dir);
}

std::vector<std::pair<int, std::filesystem::path>> CheckpointManager::list_checkpoints() const {
    std::vector<std::pair<int, std::filesystem::path>> files;
    if (!std::filesystem::exists(params_.checkpoint_dir))
        return files;
    for (const auto& entry : std::filesystem::directory_iterator(params_.checkpoint_dir)) {
        if (!entry.is_regular_file())
            continue;
        const int step = checkpoint_step(entry.path());
        if (step >= 0)
            files.emplace_back(step, entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool CheckpointManager::should_checkpoint(int step) const {
    return params_.frequency > 0 && step > 0 && (step % params_.frequency == 0);
}

void CheckpointManager::save(int step, double time, double dt, std::span<const Real> phi,
                             std::span<const Real> u) {
    auto path = params_.checkpoint_dir / ("checkpoint_" + std::to_string(step) + ".acbin");

    CheckpointIO::write(path, step, time, dt, grid_, phi, u);
    enforce_retention(step);
}

CheckpointIO::RestoreData CheckpointManager::restore() const {
    if (params_.restart_file.has_value()) {
        const auto& file = params_.restart_file.value();
        if (!std::filesystem::is_regular_file(file)) {
            throw std::runtime_error("checkpoint.restart_file does not exist: " + file.string() +
                                     " (refusing to cold-start: that would overwrite and rotate "
                                     "away the existing checkpoints)");
        }
        return CheckpointIO::read(file);
    }

    if (auto latest = restore_latest())
        return std::move(*latest);
    throw std::runtime_error("No checkpoint of this run's grid in " +
                             params_.checkpoint_dir.string() + " reads back intact");
}

std::optional<CheckpointIO::RestoreData> CheckpointManager::restore_latest() const {
    // The newest checkpoint of this run's grid that reads back intact. One of another grid
    // (retention leaves those in the directory), a damaged one (header, size or CRC) or one
    // whose header records another step than its name is skipped with a warning, and the
    // next older one is tried. Retention orders and deletes by the step in the name: a
    // copy restored under a higher name would restart at its older step, and the next save
    // would delete the checkpoints named above that step, the genuine newer ones among them.
    const auto files = list_checkpoints();
    for (auto it = files.rbegin(); it != files.rend(); ++it) {
        const auto& [step, path] = *it;
        const auto header = CheckpointIO::read_header(path);
        if (!header) {
            spdlog::warn("Skipping checkpoint {}: no intact checkpoint header", path.string());
            continue;
        }
        if (!of_this_grid(*header)) {
            spdlog::warn("Skipping checkpoint {}: of a {}x{}x{} grid, not this run's {}x{}x{}",
                         path.string(), header->Nx, header->Ny, header->Nz, grid_.Nx(), grid_.Ny(),
                         grid_.Nz());
            continue;
        }
        if (header->step != step) {
            spdlog::warn("Skipping checkpoint {}: its header records step {}, not the step of "
                         "its name",
                         path.string(), header->step);
            continue;
        }
        try {
            return CheckpointIO::read(path);
        } catch (const std::runtime_error& e) {
            spdlog::warn("Skipping checkpoint {}: {}", path.string(), e.what());
        }
    }
    return std::nullopt;
}

bool CheckpointManager::of_this_grid(const CheckpointIO::Header& header) const {
    return header.Nx == grid_.Nx() && header.Ny == grid_.Ny() && header.Nz == grid_.Nz() &&
           header.dx == grid_.dx() && header.dy == grid_.dy() && header.dz == grid_.dz();
}

bool CheckpointManager::restart_requested() const {
    return params_.restart_file.has_value();
}

void CheckpointManager::enforce_retention(int newest_step) {
    // The directory is the source of truth: a cached list goes stale when a
    // resumed run rewrites a step that already exists (duplicates), which
    // made the old deque-based policy delete the checkpoint it had just
    // written.
    const auto files = list_checkpoints();
    int kept_older = 0;
    for (auto it = files.rbegin(); it != files.rend(); ++it) {
        const auto& [step, path] = *it;
        const char* reason = nullptr;
        if (step == newest_step)
            continue;
        const auto header = CheckpointIO::read_header(path);
        if (!header || !of_this_grid(*header)) {
            if (!warned_foreign_) {
                spdlog::warn("{} also holds checkpoints that are not of this run's {}x{}x{} grid "
                             "(first: {}); retention leaves them alone. Give every run its own "
                             "checkpoint.checkpoint_dir.",
                             params_.checkpoint_dir.string(), grid_.Nx(), grid_.Ny(), grid_.Nz(),
                             path.filename().string());
                warned_foreign_ = true;
            }
            continue;
        }
        if (step > newest_step)
            reason = "newer than the checkpoint just written (from an earlier run)";
        else if (kept_older < params_.keep_last - 1)
            ++kept_older;
        else
            reason = "beyond keep_last";
        if (reason == nullptr)
            continue;
        std::error_code ec;
        std::filesystem::remove(path, ec);
        if (ec)
            spdlog::warn("Could not remove checkpoint {}: {}", path.string(), ec.message());
        else
            spdlog::info("Removed checkpoint {} ({})", path.string(), reason);
    }
}

} // namespace ac
