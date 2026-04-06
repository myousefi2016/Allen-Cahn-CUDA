#include "io/VTKWriter.hpp"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <numeric>

#ifdef AC_HAS_VTK
#include <vtkNew.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridWriter.h>
#endif

namespace ac {

VTKWriter::VTKWriter(const Grid& grid, const OutputParams& params)
    : grid_(grid), params_(params)
{
    // Ensure output directory exists
    std::filesystem::create_directories(params_.output_dir);

    // Start background writer thread
    writer_thread_ = std::thread(&VTKWriter::writer_loop, this);
    spdlog::debug("VTKWriter initialized, output_dir={}", params_.output_dir.string());
}

VTKWriter::~VTKWriter()
{
    flush();
    stop_ = true;
    queue_cv_.notify_one();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
}

void VTKWriter::write_async(int step, double time,
                             const FieldData& phi, const FieldData& u)
{
    WriteJob job;
    job.step = step;
    job.time = time;
    job.phi_data.assign(phi.data(), phi.data() + phi.size());
    job.u_data.assign(u.data(), u.data() + u.size());

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        job_queue_.push(std::move(job));
    }
    queue_cv_.notify_one();
}

void VTKWriter::flush()
{
    std::unique_lock<std::mutex> lock(queue_mutex_);
    queue_cv_.wait(lock, [this] { return job_queue_.empty(); });
}

int VTKWriter::pending_jobs() const
{
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return static_cast<int>(job_queue_.size());
}

void VTKWriter::writer_loop()
{
    while (true) {
        WriteJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !job_queue_.empty() || stop_; });

            if (stop_ && job_queue_.empty()) return;
            if (job_queue_.empty()) continue;

            job = std::move(job_queue_.front());
            job_queue_.pop();
        }
        // Notify flush() that queue shrunk
        queue_cv_.notify_all();

        try {
            // Compute and cache field statistics
            auto phi_stats = compute_statistics(job.phi_data, job.step, "phi");
            auto u_stats = compute_statistics(job.u_data, job.step, "u");

            spdlog::debug("Step {} stats: phi=[{:.4f}, {:.4f}], u=[{:.4f}, {:.4f}]",
                          job.step, phi_stats.min_val, phi_stats.max_val,
                          u_stats.min_val, u_stats.max_val);

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
    }
}

void VTKWriter::write_vtk_file(const WriteJob& job)
{
#ifdef AC_HAS_VTK
    int Nx = grid_.Nx(), Ny = grid_.Ny(), Nz = grid_.Nz();

    // Create VTK arrays
    vtkNew<vtkDoubleArray> phi_arr;
    phi_arr->SetNumberOfComponents(1);
    phi_arr->SetNumberOfTuples(grid_.total_points());
    phi_arr->SetName("phi");
    for (Index i = 0; i < grid_.total_points(); ++i) {
        phi_arr->SetValue(i, job.phi_data[static_cast<std::size_t>(i)]);
    }

    vtkNew<vtkDoubleArray> u_arr;
    u_arr->SetNumberOfComponents(1);
    u_arr->SetNumberOfTuples(grid_.total_points());
    u_arr->SetName("u");
    for (Index i = 0; i < grid_.total_points(); ++i) {
        u_arr->SetValue(i, job.u_data[static_cast<std::size_t>(i)]);
    }

    // Create structured grid
    vtkNew<vtkPoints> points;
    for (int x = 0; x < Nx; ++x) {
        for (int y = 0; y < Ny; ++y) {
            for (int z = 0; z < Nz; ++z) {
                points->InsertNextPoint(
                    x * grid_.dx(), y * grid_.dy(), z * grid_.dz());
            }
        }
    }

    vtkNew<vtkStructuredGrid> sg;
    sg->SetDimensions(Nx, Ny, Nz);
    sg->SetPoints(points);
    sg->GetPointData()->AddArray(phi_arr);
    sg->GetPointData()->AddArray(u_arr);

    // Write VTS file
    std::string filename = (params_.output_dir /
        ("output_" + std::to_string(job.step) + ".vts")).string();

    vtkNew<vtkXMLStructuredGridWriter> writer;
    writer->SetFileName(filename.c_str());
    writer->SetInputData(sg);
    writer->Write();

    spdlog::info("Wrote VTK file: {} (step={}, time={:.4f})",
                 filename, job.step, job.time);
#else
    spdlog::warn("VTK support not compiled in, falling back to raw output");
    write_raw_file(job);
#endif
}

void VTKWriter::write_raw_file(const WriteJob& job)
{
    std::string base = (params_.output_dir /
        ("output_" + std::to_string(job.step))).string();

    // Write phi
    {
        std::ofstream ofs(base + "_phi.raw", std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + base + "_phi.raw");
        }
        ofs.write(reinterpret_cast<const char*>(job.phi_data.data()),
                  static_cast<std::streamsize>(job.phi_data.size() * sizeof(Real)));
        if (!ofs.good()) {
            throw std::runtime_error("Failed to write file: " + base + "_phi.raw");
        }
    }

    // Write u
    {
        std::ofstream ofs(base + "_u.raw", std::ios::binary);
        if (!ofs.is_open()) {
            throw std::runtime_error("Failed to open file: " + base + "_u.raw");
        }
        ofs.write(reinterpret_cast<const char*>(job.u_data.data()),
                  static_cast<std::streamsize>(job.u_data.size() * sizeof(Real)));
        if (!ofs.good()) {
            throw std::runtime_error("Failed to write file: " + base + "_u.raw");
        }
    }

    spdlog::info("Wrote raw files: {}_phi.raw, {}_u.raw (step={}, time={:.4f})",
                 base, base, job.step, job.time);
}

FieldStatistics VTKWriter::compute_statistics(const std::vector<Real>& data,
                                               int step, const std::string& name)
{
    std::string key = std::to_string(step) + ":" + name;

    // Check cache first
    auto cached = stats_cache_.get(key);
    if (cached) return *cached;

    FieldStatistics stats;
    if (data.empty()) return stats;

    stats.min_val = *std::min_element(data.begin(), data.end());
    stats.max_val = *std::max_element(data.begin(), data.end());

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

std::optional<FieldStatistics> VTKWriter::get_cached_stats(
    int step, const std::string& field_name) const
{
    std::string key = std::to_string(step) + ":" + field_name;
    return stats_cache_.get(key);
}

} // namespace ac
