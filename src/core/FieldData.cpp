#include "core/FieldData.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ac {

FieldData::FieldData(const Grid& grid, std::string name)
    : data_(static_cast<std::size_t>(grid.total_points()), 0.0), Nx_(grid.Nx()), Ny_(grid.Ny()),
      Nz_(grid.Nz()), name_(std::move(name)) {}

void FieldData::fill(Real value) {
    std::fill(data_.begin(), data_.end(), value);
}

void FieldData::copy_from(const Real* src, std::size_t count) {
    if (count > data_.size()) {
        throw std::out_of_range("FieldData::copy_from: count (" + std::to_string(count) +
                                ") exceeds field size (" + std::to_string(data_.size()) + ")");
    }
    std::memcpy(data_.data(), src, count * sizeof(Real));
}

} // namespace ac
