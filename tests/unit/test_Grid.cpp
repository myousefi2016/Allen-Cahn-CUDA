#include "core/Grid.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace ac;

TEST(GridTest, Construction) {
    Grid grid(Dim3{10, 20, 30}, Spacing{0.5, 0.5, 0.5});
    EXPECT_EQ(grid.Nx(), 10);
    EXPECT_EQ(grid.Ny(), 20);
    EXPECT_EQ(grid.Nz(), 30);
    EXPECT_DOUBLE_EQ(grid.dx(), 0.5);
    EXPECT_DOUBLE_EQ(grid.dy(), 0.5);
    EXPECT_DOUBLE_EQ(grid.dz(), 0.5);
}

TEST(GridTest, TotalPoints) {
    Grid grid(Dim3{10, 20, 30}, Spacing{1.0, 1.0, 1.0});
    EXPECT_EQ(grid.total_points(), 6000);
}

TEST(GridTest, TotalBytes) {
    Grid grid(Dim3{10, 10, 10}, Spacing{1.0, 1.0, 1.0});
    EXPECT_EQ(grid.total_bytes(), 1000 * sizeof(double));
}

TEST(GridTest, Interior) {
    Grid grid(Dim3{10, 10, 10}, Spacing{1.0, 1.0, 1.0});
    EXPECT_TRUE(grid.is_interior(5, 5, 5));
    EXPECT_TRUE(grid.is_interior(1, 1, 1));
    EXPECT_TRUE(grid.is_interior(8, 8, 8));
    EXPECT_FALSE(grid.is_interior(0, 5, 5));
    EXPECT_FALSE(grid.is_interior(9, 5, 5));
    EXPECT_FALSE(grid.is_interior(5, 0, 5));
    EXPECT_FALSE(grid.is_interior(5, 9, 5));
    EXPECT_FALSE(grid.is_interior(5, 5, 0));
    EXPECT_FALSE(grid.is_interior(5, 5, 9));
}

TEST(GridTest, Boundary) {
    Grid grid(Dim3{10, 10, 10}, Spacing{1.0, 1.0, 1.0});
    EXPECT_TRUE(grid.is_boundary(0, 5, 5));
    EXPECT_TRUE(grid.is_boundary(9, 5, 5));
    EXPECT_TRUE(grid.is_boundary(5, 0, 5));
    EXPECT_TRUE(grid.is_boundary(0, 0, 0));
    EXPECT_FALSE(grid.is_boundary(5, 5, 5));
}

TEST(GridTest, InvalidDimensionsTooSmall) {
    EXPECT_THROW(Grid(Dim3{2, 10, 10}, Spacing{1.0, 1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(Grid(Dim3{10, 2, 10}, Spacing{1.0, 1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(Grid(Dim3{10, 10, 2}, Spacing{1.0, 1.0, 1.0}), std::invalid_argument);
}

TEST(GridTest, InvalidSpacingNonPositive) {
    EXPECT_THROW(Grid(Dim3{10, 10, 10}, Spacing{0.0, 1.0, 1.0}), std::invalid_argument);
    EXPECT_THROW(Grid(Dim3{10, 10, 10}, Spacing{1.0, -1.0, 1.0}), std::invalid_argument);
}

TEST(GridTest, Dims) {
    Grid grid(Dim3{5, 7, 11}, Spacing{0.1, 0.2, 0.3});
    auto d = grid.dims();
    EXPECT_EQ(d.nx, 5);
    EXPECT_EQ(d.ny, 7);
    EXPECT_EQ(d.nz, 11);
    auto s = grid.spacing();
    EXPECT_DOUBLE_EQ(s.dx, 0.1);
    EXPECT_DOUBLE_EQ(s.dy, 0.2);
    EXPECT_DOUBLE_EQ(s.dz, 0.3);
}

TEST(GridTest, DefaultConstruction) {
    Grid grid;
    EXPECT_EQ(grid.Nx(), 0);
    EXPECT_EQ(grid.total_points(), 0);
}
