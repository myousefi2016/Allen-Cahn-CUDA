#pragma once

#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "core/SimulationConfig.hpp"

#include <filesystem>
#include <string>

namespace ac {

/// Binary checkpoint I/O for saving and restoring simulation state.
/// Uses a simple binary format with a header for portability.
/// When HDF5 is available, uses HDF5 format instead.
class CheckpointIO {
public:
    struct Header {
        char magic[8] = {'A','C','C','H','K','P','T','\0'};
        int version = 1;
        int Nx, Ny, Nz;
        double dx, dy, dz;
        double dt;
        double time;
        int step;
        int num_fields;  // Always 2 (phi, u)
    };

    /// Write a checkpoint to disk.
    static void write(const std::filesystem::path& path,
                      int step, double time, double dt,
                      const Grid& grid,
                      const FieldData& phi, const FieldData& u);

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
