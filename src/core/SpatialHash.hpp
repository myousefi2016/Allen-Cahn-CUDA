#pragma once

#include "core/Grid.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ac {

/// 3D integer cell coordinate.
struct CellCoord {
    int cx, cy, cz;

    bool operator==(const CellCoord& other) const {
        return cx == other.cx && cy == other.cy && cz == other.cz;
    }
};

/// Hash function for CellCoord using FNV-1a mixing.
struct CellCoordHash {
    std::size_t operator()(const CellCoord& c) const noexcept {
        // FNV-1a-inspired hash combining
        std::size_t h = 14695981039346656037ULL;
        h ^= static_cast<std::size_t>(c.cx);
        h *= 1099511628211ULL;
        h ^= static_cast<std::size_t>(c.cy);
        h *= 1099511628211ULL;
        h ^= static_cast<std::size_t>(c.cz);
        h *= 1099511628211ULL;
        return h;
    }
};

/// Spatial hash map for efficient region queries on 3D grid data.
/// Partitions the simulation domain into coarse cells, each containing
/// a list of grid point indices. This enables O(1) lookup of all points
/// in a given spatial region, useful for:
///   - Interface tracking (find all points near the phase boundary)
///   - Adaptive refinement regions
///   - Localized statistics computation
///
/// Cell size should be chosen based on the expected query radius.
/// A cell_size of ~10*dx gives good balance between memory and query speed.
template <typename Value = int> class SpatialHash {
public:
    /// Construct with given cell size in physical units.
    explicit SpatialHash(Real cell_size) : cell_size_(cell_size), inv_cell_size_(1.0 / cell_size) {
        if (cell_size <= 0.0) {
            throw std::invalid_argument("SpatialHash cell_size must be positive");
        }
    }

    /// Insert a value at a physical position.
    void insert(Real x, Real y, Real z, const Value& value) {
        auto cell = to_cell(x, y, z);
        buckets_[cell].push_back(value);
        ++total_entries_;
    }

    /// Insert a grid point by its (ix, iy, iz) indices and grid spacing.
    void insert_grid_point(int ix, int iy, int iz, Real dx, Real dy, Real dz, const Value& value) {
        insert(ix * dx, iy * dy, iz * dz, value);
    }

    /// Query all values in a single cell containing the given position.
    [[nodiscard]] const std::vector<Value>* query_cell(Real x, Real y, Real z) const {
        auto cell = to_cell(x, y, z);
        auto it = buckets_.find(cell);
        return (it != buckets_.end()) ? &it->second : nullptr;
    }

    /// Query all values within a radius (in cell coordinates).
    /// Returns values from all cells within 'radius_cells' of the query point.
    [[nodiscard]] std::vector<Value> query_radius(Real x, Real y, Real z,
                                                  int radius_cells = 1) const {
        std::vector<Value> result;
        auto center = to_cell(x, y, z);

        for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
            for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                for (int dz = -radius_cells; dz <= radius_cells; ++dz) {
                    CellCoord c{center.cx + dx, center.cy + dy, center.cz + dz};
                    auto it = buckets_.find(c);
                    if (it != buckets_.end()) {
                        result.insert(result.end(), it->second.begin(), it->second.end());
                    }
                }
            }
        }
        return result;
    }

    /// Build spatial hash from a field, inserting all grid points where
    /// the predicate returns true. Useful for extracting interface points.
    template <typename FieldAccessor, typename Predicate>
    void build_from_field(int Nx, int Ny, int Nz, Real dx, Real dy, Real dz,
                          FieldAccessor&& accessor, Predicate&& pred) {
        clear();
        for (int ix = 0; ix < Nx; ++ix) {
            for (int iy = 0; iy < Ny; ++iy) {
                for (int iz = 0; iz < Nz; ++iz) {
                    if (pred(accessor(ix, iy, iz))) {
                        int linear = ix * Ny * Nz + iy * Nz + iz;
                        insert_grid_point(ix, iy, iz, dx, dy, dz, static_cast<Value>(linear));
                    }
                }
            }
        }
    }

    /// Clear all entries.
    void clear() {
        buckets_.clear();
        total_entries_ = 0;
    }

    /// Number of non-empty cells.
    [[nodiscard]] std::size_t num_cells() const { return buckets_.size(); }

    /// Total number of entries across all cells.
    [[nodiscard]] std::size_t total_entries() const { return total_entries_; }

    /// Cell size.
    [[nodiscard]] Real cell_size() const { return cell_size_; }

private:
    [[nodiscard]] CellCoord to_cell(Real x, Real y, Real z) const {
        return CellCoord{static_cast<int>(std::floor(x * inv_cell_size_)),
                         static_cast<int>(std::floor(y * inv_cell_size_)),
                         static_cast<int>(std::floor(z * inv_cell_size_))};
    }

    Real cell_size_;
    Real inv_cell_size_;
    std::unordered_map<CellCoord, std::vector<Value>, CellCoordHash> buckets_;
    std::size_t total_entries_ = 0;
};

} // namespace ac
