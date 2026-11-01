#pragma once

#include "core/Grid.hpp"

#include <cassert>
#include <cstring>
#include <string>
#include <vector>

namespace ac {

/// Host-side 3D field storage backed by std::vector<Real>.
/// Row-major ordering: index = x * Ny * Nz + y * Nz + z.
class FieldData {
public:
    FieldData() = default;
    explicit FieldData(const Grid& grid, std::string name = "unnamed");

    [[nodiscard]] Real& operator()(int x, int y, int z) {
        assert(x >= 0 && x < Nx_ && y >= 0 && y < Ny_ && z >= 0 && z < Nz_);
        return data_[linear_index(x, y, z)];
    }

    [[nodiscard]] const Real& operator()(int x, int y, int z) const {
        assert(x >= 0 && x < Nx_ && y >= 0 && y < Ny_ && z >= 0 && z < Nz_);
        return data_[linear_index(x, y, z)];
    }

    [[nodiscard]] Real* data() noexcept { return data_.data(); }
    [[nodiscard]] const Real* data() const noexcept { return data_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

    [[nodiscard]] int Nx() const noexcept { return Nx_; }
    [[nodiscard]] int Ny() const noexcept { return Ny_; }
    [[nodiscard]] int Nz() const noexcept { return Nz_; }

    /// Fill entire field with a constant value.
    void fill(Real value);

    /// Deep copy from raw pointer.
    void copy_from(const Real* src, std::size_t count);

private:
    [[nodiscard]] std::size_t linear_index(int x, int y, int z) const noexcept {
        const auto xs = static_cast<std::size_t>(x);
        const auto ys = static_cast<std::size_t>(y);
        const auto zs = static_cast<std::size_t>(z);
        const auto ny = static_cast<std::size_t>(Ny_);
        const auto nz = static_cast<std::size_t>(Nz_);
        return (xs * ny + ys) * nz + zs;
    }

    std::vector<Real> data_;
    int Nx_ = 0;
    int Ny_ = 0;
    int Nz_ = 0;
    std::string name_;
};

} // namespace ac
