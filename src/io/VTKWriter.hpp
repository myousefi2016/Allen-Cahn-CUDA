#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/LRUCache.hpp"
#include "core/SimulationConfig.hpp"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace ac {

/// Cached field statistics to avoid recomputation.
struct FieldStatistics {
    double min_val = 0.0;
    double max_val = 0.0;
    double mean_val = 0.0;
    double l2_norm = 0.0;
};

/// Asynchronous VTK file writer.
/// Uses a background thread to avoid blocking the simulation.
/// Caches field statistics via LRU cache for efficient metadata output.
class VTKWriter {
public:
    VTKWriter(const Grid& grid, const OutputParams& params);
    ~VTKWriter();

    VTKWriter(const VTKWriter&) = delete;
    VTKWriter& operator=(const VTKWriter&) = delete;

    /// Enqueue a write job (non-blocking: copies data internally).
    void write_async(int step, double time, const FieldData& phi, const FieldData& u);

    /// Wait for all pending writes to complete.
    void flush();

    /// Get number of pending write jobs.
    [[nodiscard]] int pending_jobs() const;

    /// Get cached statistics for a step/field combo.
    [[nodiscard]] std::optional<FieldStatistics>
    get_cached_stats(int step, const std::string& field_name) const;

private:
    struct WriteJob {
        int step;
        double time;
        std::vector<Real> phi_data;
        std::vector<Real> u_data;
    };

    void writer_loop();
    void write_vtk_file(const WriteJob& job);
    void write_raw_file(const WriteJob& job);

    /// Compute and cache field statistics.
    FieldStatistics compute_statistics(const std::vector<Real>& data, int step,
                                       const std::string& name);

    Grid grid_;
    OutputParams params_;
    std::thread writer_thread_;
    std::queue<WriteJob> job_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
    /// Number of jobs currently being written (popped from queue but not finished).
    /// flush() must wait for both queue empty AND active_jobs_ == 0.
    int active_jobs_{0};

    /// LRU cache for field statistics keyed by "step:field_name".
    /// Capacity of 256 covers the last 128 output steps (2 fields each).
    mutable LRUCache<std::string, FieldStatistics> stats_cache_{256};
};

} // namespace ac
