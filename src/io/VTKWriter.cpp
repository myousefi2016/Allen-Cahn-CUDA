#include "io/VTKWriter.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <spdlog/spdlog.h>

#ifdef AC_HAS_VTK
#include <vtkDoubleArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#endif

namespace ac {

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

void VTKWriter::write_async(int step, double time, const FieldData& phi, const FieldData& u) {
    WriteJob job;
    job.step = step;
    job.time = time;
    job.phi_data.assign(phi.data(), phi.data() + phi.size());
    job.u_data.assign(u.data(), u.data() + u.size());

    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return static_cast<int>(job_queue_.size()) + active_jobs_ < max_queue_depth_;
        });
        job_queue_.push(std::move(job));
    }
    queue_cv_.notify_all();
}

void VTKWriter::flush() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return job_queue_.empty() && active_jobs_ == 0; });
}

int VTKWriter::pending_jobs() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(job_queue_.size()) + active_jobs_;
}

void VTKWriter::writer_loop() {
    while (true) {
        WriteJob job;
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
        }

        try {
            // Compute and cache field statistics
            auto phi_stats = compute_statistics(job.phi_data, job.step, "phi");
            auto u_stats = compute_statistics(job.u_data, job.step, "u");

            spdlog::debug("Step {} stats: phi=[{:.4f}, {:.4f}], u=[{:.4f}, {:.4f}]", job.step,
                          phi_stats.min_val, phi_stats.max_val, u_stats.min_val, u_stats.max_val);

            if (params_.format == "vts") {
                write_vtk_file(job);
            } else {
                write_raw_file(job);
            }
        } catch (const std::exception& e) {
            spdlog::error("VTK writer failed for step {}: {}", job.step, e.what());
        } catch (...) {
            spdlog::error("VTK writer failed for step {} with unknown error", job.step);
        }

        // Mark job as done and notify flush()
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            --active_jobs_;
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
    writer->Write();

    std::filesystem::rename(tmp_filename, filename);
    spdlog::info("Wrote VTK file: {} (step={}, time={:.4f})", filename, job.step, job.time);
#else
    spdlog::warn("VTK support not compiled in, falling back to raw output");
    write_raw_file(job);
#endif
}

void VTKWriter::write_raw_file(const WriteJob& job) {
    std::string base = (params_.output_dir / ("output_" + std::to_string(job.step))).string();

    auto write_atomic = [](const std::string& path, const void* data, std::size_t bytes) {
        std::string tmp = path + ".tmp";
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + tmp);
        }
        ofs.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!ofs.good()) {
            throw std::runtime_error("Failed to write file: " + tmp);
        }
        ofs.close();
        std::filesystem::rename(tmp, path);
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
