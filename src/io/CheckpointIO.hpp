#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>

namespace ac {

/// Binary checkpoint I/O for saving and restoring simulation state.
/// Uses a simple binary format with a header for portability.
class CheckpointIO {
public:
    /// The format write() produces: version 2, whose header_crc32 covers the header.
    /// read() also takes version 1, which has no header CRC (only the run state is checked).
    static constexpr int kVersion = 2;

#pragma pack(push, 1)
    struct Header {
        char magic[8] = {'A', 'C', 'C', 'H', 'K', 'P', 'T', '\0'};
        int version = kVersion;
        int Nx, Ny, Nz;
        double dx, dy, dz;
        double dt;
        double time;
        int step;
        int num_fields;            // Always 2 (phi, u)
        uint32_t data_crc32 = 0;   // CRC-32 of phi || u (version 1: 0 = not computed)
        uint32_t header_crc32 = 0; // version 2: CRC-32 of these 128 bytes with this field 0
        char reserved[48] = {};
    };
#pragma pack(pop)
    static_assert(sizeof(Header) == 128,
                  "Header must be exactly 128 bytes for binary compatibility");

    /// The CRC-32 a version-2 header stores in header_crc32: of its 128 bytes with
    /// header_crc32 zero.
    static uint32_t header_crc32(const Header& header);

    /// Standard CRC-32 (IEEE 802.3, reflected, poly 0xEDB88320). Pass a previous
    /// result as `crc` to continue a running checksum over consecutive buffers.
    static uint32_t compute_crc32(const void* data, std::size_t len, uint32_t crc = 0);

    /// Write a checkpoint to disk: phi and u hold the grid's cells in FieldData's order
    /// (throws std::invalid_argument if their sizes are not the grid's).
    static void write(const std::filesystem::path& path, int step, double time, double dt,
                      const Grid& grid, std::span<const Real> phi, std::span<const Real> u);
    static void write(const std::filesystem::path& path, int step, double time, double dt,
                      const Grid& grid, const FieldData& phi, const FieldData& u) {
        write(path, step, time, dt, grid, std::span<const Real>(phi.data(), phi.size()),
              std::span<const Real>(u.data(), u.size()));
    }

    /// Read a checkpoint from disk. Returns the restored fields.
    struct RestoreData {
        int step;
        double time;
        double dt;
        Grid grid;
        FieldData phi;
        FieldData u;
    };

    static RestoreData read(const std::filesystem::path& path);

    /// The header of a checkpoint file, if the file holds a whole header with
    /// the ACCHKPT magic, a version read() takes (1 or 2) and, for version 2, a
    /// matching header CRC.
    static std::optional<Header> read_header(const std::filesystem::path& path);

    /// Check if a file is a valid checkpoint (read_header succeeds).
    static bool is_valid_checkpoint(const std::filesystem::path& path);
};

} // namespace ac
