#include "io/CheckpointIO.hpp"

#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace ac {

uint32_t CheckpointIO::compute_crc32(const void* data, std::size_t len, uint32_t crc) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    // zlib-compatible chaining: the pre/post inversions cancel between calls,
    // so compute_crc32(b, lb, compute_crc32(a, la)) == CRC32(a || b).
    auto* p = static_cast<const uint8_t*>(data);
    crc ^= 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i)
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

namespace {

/// The run state a checkpoint records, as the engine produces it: a step >= 0, a finite
/// time >= 0 (the sum of the positive steps) and a finite dt > 0 (validated, or adapted
/// within [dt_min, dt_max]). Empty if it is, else what is wrong. A restart takes them as
/// they are: a negative step would count the loop through negative steps, a NaN dt passes
/// the adaptive restart's clamp.
std::string invalid_run_state(int step, double time, double dt) {
    if (step < 0)
        return "a negative step " + std::to_string(step);
    if (!std::isfinite(time) || time < 0.0)
        return "time " + std::to_string(time) + " (not a finite value >= 0)";
    if (!std::isfinite(dt) || !(dt > 0.0))
        return "dt " + std::to_string(dt) + " (not a finite value > 0)";
    return {};
}

} // namespace

void CheckpointIO::write(const std::filesystem::path& path, int step, double time, double dt,
                         const Grid& grid, const FieldData& phi, const FieldData& u) {
    if (const std::string why = invalid_run_state(step, time, dt); !why.empty())
        throw std::invalid_argument("Refusing to write a checkpoint with " + why + ": " +
                                    path.string());
    std::filesystem::create_directories(path.parent_path());

    // Write to a temporary file first, then atomically rename to avoid corruption
    auto tmp_path = std::filesystem::path(path.string() + ".tmp");

    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Cannot open checkpoint file for writing: " + tmp_path.string());
    }
    // Every later failure removes the partial file (as large as the checkpoint: 3.46 GB at
    // 600^3), which nothing else would: retention counts only checkpoint_<step>.acbin.
    auto fail = [&](const std::string& message) {
        if (fd >= 0)
            ::close(fd);
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw std::runtime_error(message);
    };

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

    // CRC32 over phi || u, hashed in place (a combined copy would double peak
    // host memory: +3.46 GB per checkpoint at 600^3).
    std::size_t phi_bytes = phi.size() * sizeof(Real);
    std::size_t u_bytes = u.size() * sizeof(Real);
    hdr.data_crc32 = compute_crc32(u.data(), u_bytes, compute_crc32(phi.data(), phi_bytes));

    auto write_all = [&](const void* buf, std::size_t len) {
        auto* p = static_cast<const char*>(buf);
        while (len > 0) {
            auto n = ::write(fd, p, len);
            if (n < 0) {
                if (errno == EINTR)
                    continue;
                fail("Failed writing checkpoint: " + tmp_path.string());
            }
            p += n;
            len -= static_cast<std::size_t>(n);
        }
    };

    write_all(&hdr, sizeof(Header));
    write_all(phi.data(), phi.size() * sizeof(Real));
    write_all(u.data(), u.size() * sizeof(Real));

    if (::fsync(fd) != 0)
        fail("fsync failed for checkpoint: " + tmp_path.string());
    // fsync succeeded, so the data is stored; close() releases the descriptor whatever it
    // returns (Linux).
    ::close(fd);
    fd = -1;

    // Atomically rename the temporary file to the final path
    std::error_code rename_error;
    std::filesystem::rename(tmp_path, path, rename_error);
    if (rename_error) {
        fail("Cannot rename checkpoint " + tmp_path.string() + " to " + path.string() + ": " +
             rename_error.message());
    }

    // fsync the parent directory so the rename is durable
    int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }

    // MB = 10^6 bytes, as the documented checkpoint sizes.
    spdlog::info("Checkpoint written: {} (step={}, time={:.4f}, size={:.1f} MB)", path.string(),
                 step, time,
                 static_cast<double>(sizeof(Header) + 2 * phi.size() * sizeof(Real)) / 1e6);
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

    if (hdr.Nx < 3 || hdr.Ny < 3 || hdr.Nz < 3) {
        throw std::runtime_error("Checkpoint has invalid dimensions: " + std::to_string(hdr.Nx) +
                                 "x" + std::to_string(hdr.Ny) + "x" + std::to_string(hdr.Nz));
    }
    if (!(hdr.dx > 0.0) || !(hdr.dy > 0.0) || !(hdr.dz > 0.0)) {
        throw std::runtime_error("Checkpoint has non-positive grid spacing");
    }
    if (hdr.num_fields != 2) {
        throw std::runtime_error("Checkpoint has unexpected num_fields: " +
                                 std::to_string(hdr.num_fields));
    }
    // The CRC covers the fields only: the header's run state is checked here.
    if (const std::string why = invalid_run_state(hdr.step, hdr.time, hdr.dt); !why.empty()) {
        throw std::runtime_error("Checkpoint has " + why + ": " + path.string());
    }
    // The file must be exactly the header and the two fields of the header's
    // grid. Checked before allocating: a corrupted header cannot request an
    // arbitrary allocation, and a truncated or extended file is rejected.
    std::uint64_t cells = 0, payload = 0, expected = 0;
    const bool overflow =
        __builtin_mul_overflow(static_cast<std::uint64_t>(hdr.Nx),
                               static_cast<std::uint64_t>(hdr.Ny), &cells) ||
        __builtin_mul_overflow(cells, static_cast<std::uint64_t>(hdr.Nz), &cells) ||
        __builtin_mul_overflow(cells, std::uint64_t{2} * sizeof(Real), &payload) ||
        __builtin_add_overflow(payload, std::uint64_t{sizeof(Header)}, &expected);
    std::error_code size_error;
    const std::uintmax_t actual = std::filesystem::file_size(path, size_error);
    if (size_error) {
        throw std::runtime_error("Cannot read the size of checkpoint file " + path.string() + ": " +
                                 size_error.message());
    }
    if (overflow || actual != expected) {
        throw std::runtime_error(
            "Checkpoint file size " + std::to_string(actual) + " does not match its " +
            std::to_string(hdr.Nx) + "x" + std::to_string(hdr.Ny) + "x" + std::to_string(hdr.Nz) +
            " grid (" +
            (overflow ? std::string("overflows") : std::to_string(expected) + " bytes") +
            "): " + path.string());
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

    // Verify CRC32 if present (0 means old checkpoint without CRC)
    if (hdr.data_crc32 != 0) {
        std::size_t phi_sz = phi.size() * sizeof(Real);
        std::size_t u_sz = u.size() * sizeof(Real);
        uint32_t crc = compute_crc32(u.data(), u_sz, compute_crc32(phi.data(), phi_sz));
        if (crc != hdr.data_crc32) {
            throw std::runtime_error("Checkpoint CRC32 mismatch (data corrupted): " +
                                     path.string());
        }
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

std::optional<CheckpointIO::Header> CheckpointIO::read_header(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return std::nullopt;

    Header hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(Header));
    if (!ifs.good() || std::strncmp(hdr.magic, "ACCHKPT", 7) != 0)
        return std::nullopt;
    return hdr;
}

bool CheckpointIO::is_valid_checkpoint(const std::filesystem::path& path) {
    return read_header(path).has_value();
}

} // namespace ac
