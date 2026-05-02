#include "io/CheckpointIO.hpp"

#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unistd.h>

namespace ac {

void CheckpointIO::write(const std::filesystem::path& path, int step, double time, double dt,
                         const Grid& grid, const FieldData& phi, const FieldData& u) {
    std::filesystem::create_directories(path.parent_path());

    // Write to a temporary file first, then atomically rename to avoid corruption
    auto tmp_path = std::filesystem::path(path.string() + ".tmp");

    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open checkpoint file for writing: " + tmp_path.string());
    }

    // Write header
    Header hdr{};
    hdr.Nx = grid.Nx();
    hdr.Ny = grid.Ny();
    hdr.Nz = grid.Nz();
    hdr.dx = grid.dx();
    hdr.dy = grid.dy();
    hdr.dz = grid.dz();
    hdr.dt = dt;
    hdr.time = time;
    hdr.step = step;
    hdr.num_fields = 2;

    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(Header));

    // Write phi
    ofs.write(reinterpret_cast<const char*>(phi.data()),
              static_cast<std::streamsize>(phi.size() * sizeof(Real)));

    // Write u
    ofs.write(reinterpret_cast<const char*>(u.data()),
              static_cast<std::streamsize>(u.size() * sizeof(Real)));

    // Flush and sync to ensure data is on disk before renaming
    ofs.flush();

    ofs.close();

    // Open the file read-only to obtain a descriptor for fsync
    int fd = ::open(tmp_path.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }

    // Atomically rename the temporary file to the final path
    std::filesystem::rename(tmp_path, path);

    spdlog::info(
        "Checkpoint written: {} (step={}, time={:.4f}, size={:.1f} MB)", path.string(), step, time,
        static_cast<double>(sizeof(Header) + 2 * phi.size() * sizeof(Real)) / (1024.0 * 1024.0));
}

CheckpointIO::RestoreData CheckpointIO::read(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open checkpoint file: " + path.string());
    }

    // Read header
    Header hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(Header));

    if (std::strncmp(hdr.magic, "ACCHKPT", 7) != 0) {
        throw std::runtime_error("Invalid checkpoint file magic: " + path.string());
    }
    if (hdr.version != 1) {
        throw std::runtime_error("Unsupported checkpoint version: " + std::to_string(hdr.version));
    }

    // Reconstruct grid
    Grid grid(Dim3{hdr.Nx, hdr.Ny, hdr.Nz}, Spacing{hdr.dx, hdr.dy, hdr.dz});

    // Read fields
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    ifs.read(reinterpret_cast<char*>(phi.data()),
             static_cast<std::streamsize>(phi.size() * sizeof(Real)));
    ifs.read(reinterpret_cast<char*>(u.data()),
             static_cast<std::streamsize>(u.size() * sizeof(Real)));

    if (!ifs.good()) {
        throw std::runtime_error("Checkpoint file truncated: " + path.string());
    }

    spdlog::info("Checkpoint restored: {} (step={}, time={:.4f})", path.string(), hdr.step,
                 hdr.time);

    return RestoreData{.step = hdr.step,
                       .time = hdr.time,
                       .dt = hdr.dt,
                       .grid = grid,
                       .phi = std::move(phi),
                       .u = std::move(u)};
}

bool CheckpointIO::is_valid_checkpoint(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    Header hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(Header));
    return ifs.good() && std::strncmp(hdr.magic, "ACCHKPT", 7) == 0;
}

} // namespace ac
