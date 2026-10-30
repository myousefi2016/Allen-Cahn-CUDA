#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ac {

/// Floating-point type of host-side data such as FieldData, grid spacing, the configuration,
/// and checkpoint and VTK I/O. No kernel in src/cuda/ uses it. It must stay double:
/// CudaSolver copies FieldData::data() into and out of DeviceField<double>
/// (src/cuda/CudaSolver.cu), and scripts/resume_from_latest_checkpoint.py accepts only
/// checkpoints sized for 8-byte reals (the checkpoint header records no element size).
using Real = double;

/// Index type for grid addressing.
using Index = std::int64_t;

/// 3D integer dimensions.
struct Dim3 {
    int nx = 0;
    int ny = 0;
    int nz = 0;

    [[nodiscard]] constexpr Index total() const noexcept {
        return static_cast<Index>(nx) * ny * nz;
    }
};

/// Grid spacing in each direction.
struct Spacing {
    Real dx = 1.0;
    Real dy = 1.0;
    Real dz = 1.0;
};

/// Encapsulates a uniform structured 3D grid.
class Grid {
public:
    Grid() = default;
    Grid(Dim3 dims, Spacing spacing);

    [[nodiscard]] Dim3 dims() const noexcept { return dims_; }
    [[nodiscard]] Spacing spacing() const noexcept { return spacing_; }
    [[nodiscard]] int Nx() const noexcept { return dims_.nx; }
    [[nodiscard]] int Ny() const noexcept { return dims_.ny; }
    [[nodiscard]] int Nz() const noexcept { return dims_.nz; }
    [[nodiscard]] Real dx() const noexcept { return spacing_.dx; }
    [[nodiscard]] Real dy() const noexcept { return spacing_.dy; }
    [[nodiscard]] Real dz() const noexcept { return spacing_.dz; }
    [[nodiscard]] Index total_points() const noexcept { return dims_.total(); }
    [[nodiscard]] std::size_t total_bytes() const noexcept {
        return static_cast<std::size_t>(dims_.total()) * sizeof(Real);
    }

    [[nodiscard]] bool is_interior(int x, int y, int z) const noexcept {
        return x > 0 && x < dims_.nx - 1 && y > 0 && y < dims_.ny - 1 && z > 0 && z < dims_.nz - 1;
    }

    [[nodiscard]] bool is_boundary(int x, int y, int z) const noexcept {
        return !is_interior(x, y, z) && x >= 0 && x < dims_.nx && y >= 0 && y < dims_.ny &&
               z >= 0 && z < dims_.nz;
    }

    /// Validate grid parameters.
    void validate() const;

private:
    Dim3 dims_{};
    Spacing spacing_{};
};

} // namespace ac
