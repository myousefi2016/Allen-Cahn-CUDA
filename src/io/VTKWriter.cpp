#include "io/VTKWriter.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>

#ifdef AC_HAS_VTK
#include <vtkDoubleArray.h>
#include <vtkErrorCode.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#endif

namespace ac {

namespace {

/// Stores a written file's data (fsync), renames it into place and stores the rename
/// (fsync of the directory), as CheckpointIO::write does for a checkpoint: the engine
/// writes a checkpoint only after the frames before it, which a resume never writes
/// again, so they must survive whatever the checkpoint survives.
void publish(const std::filesystem::path& tmp, const std::filesystem::path& path) {
    const int fd = ::open(tmp.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Cannot open " + tmp.string() +
                                 " to sync it: " + std::strerror(errno));
    }
    if (::fsync(fd) != 0) {
        const int error = errno;
        ::close(fd);
        throw std::runtime_error("fsync failed for " + tmp.string() + ": " + std::strerror(error));
    }
    ::close(fd);
    std::filesystem::rename(tmp, path);
    const int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }
}

} // namespace

VTKWriter::VTKWriter(const Grid& grid, const OutputParams& params) : grid_(grid), params_(params) {
    // Ensure output directory exists
    std::filesystem::create_directories(params_.output_dir);

    // Start background writer thread
    writer_thread_ = std::thread(&VTKWriter::writer_loop, this);
    spdlog::debug("VTKWriter initialized, output_dir={}", params_.output_dir.string());
}

VTKWriter::~VTKWriter() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
}

void VTKWriter::write_async(int step, double time, const FieldData& phi, const FieldData& u,
                            bool frame, Checkpoint checkpoint) {
    WriteJob job;
    job.step = step;
    job.time = time;
    job.phi_data.assign(phi.data(), phi.data() + phi.size());
    job.u_data.assign(u.data(), u.data() + u.size());
    job.frame = frame;
    job.checkpoint = std::move(checkpoint);

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        rethrow_error_locked();
        queue_cv_.wait(lock, [this] {
            return static_cast<int>(job_queue_.size()) + active_jobs_ < max_queue_depth_;
        });
        job_queue_.push(std::move(job));
    }
    queue_cv_.notify_all();
    if (!params_.async_io)
        flush();
}

void VTKWriter::flush() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return job_queue_.empty() && active_jobs_ == 0; });
    rethrow_error_locked();
}

void VTKWriter::rethrow_error_locked() const {
    if (first_error_)
        std::rethrow_exception(first_error_);
}

int VTKWriter::pending_jobs() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(job_queue_.size()) + active_jobs_;
}

void VTKWriter::writer_loop() {
    while (true) {
        WriteJob job;
        bool failed_before = false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !job_queue_.empty() || stop_; });

            if (stop_ && job_queue_.empty())
                return;
            if (job_queue_.empty())
                continue;

            job = std::move(job_queue_.front());
            job_queue_.pop();
            ++active_jobs_;
            failed_before = static_cast<bool>(first_error_);
        }

        // After a failure the later snapshots are dropped: a checkpoint after a lost frame
        // would make a resume skip it, and the run stops at the next write_async() or
        // flush() anyway.
        std::exception_ptr error;
        const char* what = "output";
        try {
            if (!failed_before && job.frame) {
                // Compute and cache field statistics
                auto phi_stats = compute_statistics(job.phi_data, job.step, "phi");
                auto u_stats = compute_statistics(job.u_data, job.step, "u");

                spdlog::debug("Step {} stats: phi=[{:.4f}, {:.4f}], u=[{:.4f}, {:.4f}]", job.step,
                              phi_stats.min_val, phi_stats.max_val, u_stats.min_val,
                              u_stats.max_val);

                if (params_.format == "vts") {
                    write_vtk_file(job);
                } else {
                    write_raw_file(job);
                }
            }
            if (!failed_before && job.checkpoint) {
                what = "checkpoint";
                job.checkpoint(job.step, job.time, job.phi_data, job.u_data);
            }
        } catch (const std::exception& e) {
            spdlog::error("{} write failed for step {}: {}", what, job.step, e.what());
            error = std::make_exception_ptr(
                std::runtime_error(std::string(what) + " write failed for step " +
                                   std::to_string(job.step) + ": " + e.what()));
        } catch (...) {
            spdlog::error("{} write failed for step {} with unknown error", what, job.step);
            error = std::make_exception_ptr(
                std::runtime_error(std::string(what) + " write failed for step " +
                                   std::to_string(job.step) + ": unknown error"));
        }

        // Mark job as done (recording the first failure) and notify flush()
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            --active_jobs_;
            if (error && !first_error_)
                first_error_ = error;
        }
        queue_cv_.notify_all();
    }
}

void VTKWriter::write_vtk_file(const WriteJob& job) {
#ifdef AC_HAS_VTK
    const int Nx = grid_.Nx(), Ny = grid_.Ny(), Nz = grid_.Nz();
    const Index n = grid_.total_points();

    // vtkStructuredGrid numbers point (i, j, k) as i + Nx*(j + Ny*k), i.e.
    // x fastest, while FieldData stores (x, y, z) at (x*Ny + y)*Nz + z, i.e.
    // z fastest. Points and both arrays are written in VTK order so that the
    // structured topology matches the geometry (otherwise cubic grids get
    // axis-permuted, inside-out cells and non-cubic grids a scrambled mesh).
    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(n);
    vtkNew<vtkDoubleArray> phi_arr;
    phi_arr->SetNumberOfComponents(1);
    phi_arr->SetNumberOfTuples(n);
    phi_arr->SetName("phi");
    vtkNew<vtkDoubleArray> u_arr;
    u_arr->SetNumberOfComponents(1);
    u_arr->SetNumberOfTuples(n);
    u_arr->SetName("u");

    vtkIdType vtk_id = 0;
    for (int z = 0; z < Nz; ++z) {
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x, ++vtk_id) {
                const std::size_t src =
                    (static_cast<std::size_t>(x) * static_cast<std::size_t>(Ny) +
                     static_cast<std::size_t>(y)) *
                        static_cast<std::size_t>(Nz) +
                    static_cast<std::size_t>(z);
                points->SetPoint(vtk_id, x * grid_.dx(), y * grid_.dy(), z * grid_.dz());
                phi_arr->SetValue(vtk_id, job.phi_data[src]);
                u_arr->SetValue(vtk_id, job.u_data[src]);
            }
        }
    }

    vtkNew<vtkStructuredGrid> sg;
    sg->SetDimensions(Nx, Ny, Nz);
    sg->SetPoints(points);
    sg->GetPointData()->AddArray(phi_arr);
    sg->GetPointData()->AddArray(u_arr);

    // Write to temp file first, then rename atomically to avoid corruption on crash
    std::string filename =
        (params_.output_dir / ("output_" + std::to_string(job.step) + ".vts")).string();
    std::string tmp_filename = filename + ".tmp";

    vtkNew<vtkXMLStructuredGridWriter> writer;
    writer->SetFileName(tmp_filename.c_str());
    writer->SetInputData(sg);
    // Write() returns 1 on success; the error code names the cause (e.g. out
    // of disk space). A failed file must never be renamed into place.
    // A failed or unpublishable file is removed: a partial snapshot only takes space.
    if (writer->Write() != 1 || writer->GetErrorCode() != vtkErrorCode::NoError) {
        const unsigned long code = writer->GetErrorCode();
        std::error_code ec;
        std::filesystem::remove(tmp_filename, ec);
        throw std::runtime_error("VTK write of " + tmp_filename +
                                 " failed: " + vtkErrorCode::GetStringFromErrorCode(code));
    }
    try {
        publish(tmp_filename, filename);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp_filename, ec);
        throw;
    }
    spdlog::info("Wrote VTK file: {} (step={}, time={:.4f})", filename, job.step, job.time);
#else
    spdlog::warn("VTK support not compiled in, falling back to raw output");
    write_raw_file(job);
#endif
}

void VTKWriter::write_raw_file(const WriteJob& job) {
    std::string base = (params_.output_dir / ("output_" + std::to_string(job.step))).string();

    // The file is renamed into place only once close() has flushed it without an error:
    // a write smaller than the stream's buffer (8 KiB in libstdc++) reaches the file only
    // there. A failed or unpublishable file is removed (a partial snapshot only takes
    // space).
    auto write_atomic = [](const std::string& path, const void* data, std::size_t bytes) {
        const std::string tmp = path + ".tmp";
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + tmp);
        }
        try {
            ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            ofs.close();
            if (ofs.fail()) {
                throw std::runtime_error("Failed to write file: " + tmp);
            }
            publish(tmp, path);
        } catch (...) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            throw;
        }
    };

    write_atomic(base + "_phi.raw", job.phi_data.data(), job.phi_data.size() * sizeof(Real));
    write_atomic(base + "_u.raw", job.u_data.data(), job.u_data.size() * sizeof(Real));

    spdlog::info("Wrote raw files: {}_phi.raw, {}_u.raw (step={}, time={:.4f})", base, base,
                 job.step, job.time);
}

FieldStatistics VTKWriter::compute_statistics(const std::vector<Real>& data, int step,
                                              const std::string& name) {
    std::string key = std::to_string(step) + ":" + name;

    // Check cache first
    auto cached = stats_cache_.get(key);
    if (cached)
        return *cached;

    FieldStatistics stats;
    if (data.empty())
        return stats;

    stats.min_val = *std::min_element(data.begin(), data.end());
    stats.max_val = *std::max_element(data.begin(), data.end());

    if (!std::isfinite(stats.min_val) || !std::isfinite(stats.max_val)) {
        spdlog::warn("Field '{}' at step {} contains NaN/Inf values", name, step);
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (auto v : data) {
        sum += v;
        sum_sq += v * v;
    }
    auto n = static_cast<double>(data.size());
    stats.mean_val = sum / n;
    stats.l2_norm = std::sqrt(sum_sq / n);

    stats_cache_.put(key, stats);
    return stats;
}

std::optional<FieldStatistics> VTKWriter::get_cached_stats(int step,
                                                           const std::string& field_name) const {
    std::string key = std::to_string(step) + ":" + field_name;
    return stats_cache_.get(key);
}

} // namespace ac
