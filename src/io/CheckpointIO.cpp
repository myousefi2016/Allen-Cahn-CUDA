#include "io/CheckpointIO.hpp"

#include <array>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <unistd.h>

namespace ac {

uint32_t CheckpointIO::compute_crc32(const void* data, std::size_t len) {
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
    auto* p = static_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i)
        crc = table[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

void CheckpointIO::write(const std::filesystem::path& path, int step, double time, double dt,
                         const Grid& grid, const FieldData& phi, const FieldData& u) {
    std::filesystem::create_directories(path.parent_path());

    // Write to a temporary file first, then atomically rename to avoid corruption
    auto tmp_path = std::filesystem::path(path.string() + ".tmp");

    int fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        throw std::runtime_error("Cannot open checkpoint file for writing: " + tmp_path.string());
    }

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

    // Compute CRC32 over field data
    std::size_t phi_bytes = phi.size() * sizeof(Real);
    std::size_t u_bytes = u.size() * sizeof(Real);
    uint32_t crc = 0xFFFFFFFFu;
    {
        auto update_crc = [](uint32_t c, const void* data, std::size_t len) {
            static const auto tbl = [] {
                std::array<uint32_t, 256> t{};
                for (uint32_t i = 0; i < 256; ++i) {
                    uint32_t v = i;
                    for (int j = 0; j < 8; ++j)
                        v = (v & 1) ? (0xEDB88320u ^ (v >> 1)) : (v >> 1);
                    t[i] = v;
                }
                return t;
            }();
            auto* p = static_cast<const uint8_t*>(data);
            for (std::size_t i = 0; i < len; ++i)
                c = tbl[(c ^ p[i]) & 0xFF] ^ (c >> 8);
            return c;
        };
        crc = update_crc(crc, phi.data(), phi_bytes);
        crc = update_crc(crc, u.data(), u_bytes);
    }
    hdr.data_crc32 = crc ^ 0xFFFFFFFFu;

    auto write_all = [&](const void* buf, std::size_t len) {
        auto* p = static_cast<const char*>(buf);
        while (len > 0) {
            auto n = ::write(fd, p, len);
            if (n <= 0) {
                ::close(fd);
                throw std::runtime_error("Failed writing checkpoint: " + tmp_path.string());
            }
            p += n;
            len -= static_cast<std::size_t>(n);
        }
    };

    write_all(&hdr, sizeof(Header));
    write_all(phi.data(), phi.size() * sizeof(Real));
    write_all(u.data(), u.size() * sizeof(Real));

    if (::fsync(fd) != 0) {
        ::close(fd);
        throw std::runtime_error("fsync failed for checkpoint: " + tmp_path.string());
    }
    ::close(fd);

    // Atomically rename the temporary file to the final path
    std::filesystem::rename(tmp_path, path);

    // fsync the parent directory so the rename is durable
    int dir_fd = ::open(path.parent_path().c_str(), O_RDONLY);
    if (dir_fd >= 0) {
        ::fsync(dir_fd);
        ::close(dir_fd);
    }

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

    if (hdr.Nx < 1 || hdr.Ny < 1 || hdr.Nz < 1 || hdr.Nx > 100000 || hdr.Ny > 100000 ||
        hdr.Nz > 100000) {
        throw std::runtime_error("Checkpoint has invalid dimensions: " + std::to_string(hdr.Nx) +
                                 "x" + std::to_string(hdr.Ny) + "x" + std::to_string(hdr.Nz));
    }
    if (hdr.dx <= 0.0 || hdr.dy <= 0.0 || hdr.dz <= 0.0) {
        throw std::runtime_error("Checkpoint has non-positive grid spacing");
    }
    if (hdr.num_fields != 2) {
        throw std::runtime_error("Checkpoint has unexpected num_fields: " +
                                 std::to_string(hdr.num_fields));
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
        uint32_t crc = 0xFFFFFFFFu;
        auto update_crc = [](uint32_t c, const void* data, std::size_t len) {
            static const auto tbl = [] {
                std::array<uint32_t, 256> t{};
                for (uint32_t i = 0; i < 256; ++i) {
                    uint32_t v = i;
                    for (int j = 0; j < 8; ++j)
                        v = (v & 1) ? (0xEDB88320u ^ (v >> 1)) : (v >> 1);
                    t[i] = v;
                }
                return t;
            }();
            auto* p = static_cast<const uint8_t*>(data);
            for (std::size_t i = 0; i < len; ++i)
                c = tbl[(c ^ p[i]) & 0xFF] ^ (c >> 8);
            return c;
        };
        crc = update_crc(crc, phi.data(), phi.size() * sizeof(Real));
        crc = update_crc(crc, u.data(), u.size() * sizeof(Real));
        crc ^= 0xFFFFFFFFu;
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

bool CheckpointIO::is_valid_checkpoint(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open())
        return false;

    Header hdr{};
    ifs.read(reinterpret_cast<char*>(&hdr), sizeof(Header));
    return ifs.good() && std::strncmp(hdr.magic, "ACCHKPT", 7) == 0;
}

} // namespace ac
