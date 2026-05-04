#pragma once

#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

namespace ac {

/// Binary checkpoint I/O for saving and restoring simulation state.
/// Uses a simple binary format with a header for portability.
/// When HDF5 is available, uses HDF5 format instead.
class CheckpointIO {
public:
#pragma pack(push, 1)
    struct Header {
        char magic[8] = {'A', 'C', 'C', 'H', 'K', 'P', 'T', '\0'};
        int version = 1;
        int Nx, Ny, Nz;
        double dx, dy, dz;
        double dt;
        double time;
        int step;
        int num_fields;          // Always 2 (phi, u)
        uint32_t data_crc32 = 0; // CRC32 of field data (0 = not computed, backward compat)
        char reserved[52] = {};
    };
#pragma pack(pop)
    static_assert(sizeof(Header) == 128,
                  "Header must be exactly 128 bytes for binary compatibility");

    static uint32_t compute_crc32(const void* data, std::size_t len);

    /// Write a checkpoint to disk.
    static void write(const std::filesystem::path& path, int step, double time, double dt,
                      const Grid& grid, const FieldData& phi, const FieldData& u);

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

    /// Check if a file is a valid checkpoint.
    static bool is_valid_checkpoint(const std::filesystem::path& path);
};

} // namespace ac
