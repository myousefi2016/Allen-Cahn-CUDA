#include "core/FieldData.hpp"

#include <algorithm>
#include <cstring>

namespace ac {

FieldData::FieldData(const Grid& grid, std::string name)
    : data_(static_cast<std::size_t>(grid.total_points()), 0.0)
    , Nx_(grid.Nx())
    , Ny_(grid.Ny())
    , Nz_(grid.Nz())
    , name_(std::move(name))
{
}

void FieldData::fill(Real value)
{
    std::fill(data_.begin(), data_.end(), value);
}

void FieldData::copy_from(const Real* src, std::size_t count)
{
    assert(count <= data_.size());
    std::memcpy(data_.data(), src, count * sizeof(Real));
}

} // namespace ac
