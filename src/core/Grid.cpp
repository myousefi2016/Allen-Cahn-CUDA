#include "core/Grid.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>

namespace ac {

Grid::Grid(Dim3 dims, Spacing spacing) : dims_(dims), spacing_(spacing) {
    validate();
}

void Grid::validate() const {
    if (dims_.nx < 3 || dims_.ny < 3 || dims_.nz < 3) {
        throw std::invalid_argument(
            "Grid dimensions must be >= 3 in each direction. Got: " + std::to_string(dims_.nx) +
            "x" + std::to_string(dims_.ny) + "x" + std::to_string(dims_.nz));
    }
    if (spacing_.dx <= 0.0 || spacing_.dy <= 0.0 || spacing_.dz <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }
    spdlog::debug("Grid validated: {}x{}x{}, dx={}, dy={}, dz={}", dims_.nx, dims_.ny, dims_.nz,
                  spacing_.dx, spacing_.dy, spacing_.dz);
}

} // namespace ac
