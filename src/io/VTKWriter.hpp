#pragma once

#include "core/Grid.hpp"
#include "core/FieldData.hpp"
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

/// Asynchronous VTK file writer.
/// Uses a background thread to avoid blocking the simulation.
class VTKWriter {
public:
    VTKWriter(const Grid& grid, const OutputParams& params);
    ~VTKWriter();

    VTKWriter(const VTKWriter&) = delete;
    VTKWriter& operator=(const VTKWriter&) = delete;

    /// Enqueue a write job (non-blocking: copies data internally).
    void write_async(int step, double time,
                     const FieldData& phi, const FieldData& u);

    /// Wait for all pending writes to complete.
    void flush();

    /// Get number of pending write jobs.
    [[nodiscard]] int pending_jobs() const;

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

    Grid grid_;
    OutputParams params_;
    std::thread writer_thread_;
    std::queue<WriteJob> job_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_{false};
};

} // namespace ac
