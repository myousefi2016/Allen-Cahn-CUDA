#include "core/SpatialHash.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace ac;

TEST(SpatialHashTest, InsertAndQuery)
{
    SpatialHash<int> hash(1.0);
    hash.insert(0.5, 0.5, 0.5, 42);
    hash.insert(0.7, 0.3, 0.8, 43);

    auto* cell = hash.query_cell(0.5, 0.5, 0.5);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->size(), 2u);
    EXPECT_EQ(hash.total_entries(), 2u);
}

TEST(SpatialHashTest, DifferentCells)
{
    SpatialHash<int> hash(1.0);
    hash.insert(0.5, 0.5, 0.5, 1);   // cell (0,0,0)
    hash.insert(1.5, 0.5, 0.5, 2);   // cell (1,0,0)

    auto* cell0 = hash.query_cell(0.5, 0.5, 0.5);
    auto* cell1 = hash.query_cell(1.5, 0.5, 0.5);

    ASSERT_NE(cell0, nullptr);
    ASSERT_NE(cell1, nullptr);
    EXPECT_EQ(cell0->size(), 1u);
    EXPECT_EQ(cell1->size(), 1u);
    EXPECT_EQ(hash.num_cells(), 2u);
}

TEST(SpatialHashTest, RadiusQuery)
{
    SpatialHash<int> hash(1.0);
    hash.insert(0.5, 0.5, 0.5, 1);   // cell (0,0,0)
    hash.insert(1.5, 0.5, 0.5, 2);   // cell (1,0,0)
    hash.insert(5.5, 5.5, 5.5, 99);  // far away

    auto result = hash.query_radius(0.5, 0.5, 0.5, 1);
    EXPECT_EQ(result.size(), 2u);

    auto far_result = hash.query_radius(0.5, 0.5, 0.5, 0);
    EXPECT_EQ(far_result.size(), 1u);
}

TEST(SpatialHashTest, EmptyQuery)
{
    SpatialHash<int> hash(1.0);
    auto* cell = hash.query_cell(10.0, 10.0, 10.0);
    EXPECT_EQ(cell, nullptr);

    auto result = hash.query_radius(10.0, 10.0, 10.0, 1);
    EXPECT_TRUE(result.empty());
}

TEST(SpatialHashTest, NegativeCoordinates)
{
    SpatialHash<int> hash(2.0);
    hash.insert(-1.5, -3.5, -0.5, 10);

    auto* cell = hash.query_cell(-1.0, -3.0, -0.1);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ(cell->size(), 1u);
}

TEST(SpatialHashTest, Clear)
{
    SpatialHash<int> hash(1.0);
    hash.insert(0.5, 0.5, 0.5, 1);
    hash.insert(1.5, 1.5, 1.5, 2);
    hash.clear();

    EXPECT_EQ(hash.num_cells(), 0u);
    EXPECT_EQ(hash.total_entries(), 0u);
}

TEST(SpatialHashTest, BuildFromField)
{
    SpatialHash<int> hash(2.0);
    int Nx = 10, Ny = 10, Nz = 10;
    Real dx = 0.5, dy = 0.5, dz = 0.5;

    // Create a simple field: phi > 0 near center
    auto accessor = [&](int x, int y, int z) -> Real {
        Real cx = 5.0, cy = 5.0, cz = 5.0;
        Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz));
        return (r < 3.0) ? 1.0 : -1.0;
    };

    // Insert only interface points (|phi| < 0.5 won't match here, use phi > 0)
    auto is_interface = [](Real phi) { return phi > 0.0; };

    hash.build_from_field(Nx, Ny, Nz, dx, dy, dz, accessor, is_interface);

    EXPECT_GT(hash.total_entries(), 0u);
    EXPECT_GT(hash.num_cells(), 0u);
}

TEST(SpatialHashTest, GridPointInsert)
{
    SpatialHash<int> hash(1.0);
    hash.insert_grid_point(5, 10, 15, 0.1, 0.1, 0.1, 42);

    // Physical position is (0.5, 1.0, 1.5), cell (0, 1, 1)
    auto* cell = hash.query_cell(0.5, 1.0, 1.5);
    ASSERT_NE(cell, nullptr);
    EXPECT_EQ((*cell)[0], 42);
}

TEST(SpatialHashTest, InvalidCellSizeThrows)
{
    EXPECT_THROW(SpatialHash<int>(0.0), std::invalid_argument);
    EXPECT_THROW(SpatialHash<int>(-1.0), std::invalid_argument);
}
