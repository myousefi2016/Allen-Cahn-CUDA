#include "core/FieldData.hpp"
#include "core/Grid.hpp"

#include <gtest/gtest.h>

using namespace ac;

TEST(FieldDataTest, Construction) {
    Grid grid(Dim3{10, 10, 10}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid, "test_field");

    EXPECT_EQ(field.size(), 1000u);
    EXPECT_EQ(field.Nx(), 10);
    EXPECT_EQ(field.Ny(), 10);
    EXPECT_EQ(field.Nz(), 10);
    EXPECT_EQ(field.name(), "test_field");
}

TEST(FieldDataTest, InitializedToZero) {
    Grid grid(Dim3{5, 5, 5}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);

    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 5; ++y)
            for (int z = 0; z < 5; ++z)
                EXPECT_DOUBLE_EQ(field(x, y, z), 0.0);
}

TEST(FieldDataTest, ReadWrite) {
    Grid grid(Dim3{5, 5, 5}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);

    field(2, 3, 4) = 42.0;
    EXPECT_DOUBLE_EQ(field(2, 3, 4), 42.0);

    field(0, 0, 0) = -1.5;
    EXPECT_DOUBLE_EQ(field(0, 0, 0), -1.5);
}

TEST(FieldDataTest, Fill) {
    Grid grid(Dim3{4, 4, 4}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);

    field.fill(3.14);
    for (std::size_t i = 0; i < field.size(); ++i) {
        EXPECT_DOUBLE_EQ(field.data()[i], 3.14);
    }
}

TEST(FieldDataTest, CopyFrom) {
    Grid grid(Dim3{3, 3, 3}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);

    std::vector<Real> src(27, 7.7);
    field.copy_from(src.data(), src.size());

    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            for (int z = 0; z < 3; ++z)
                EXPECT_DOUBLE_EQ(field(x, y, z), 7.7);
}

TEST(FieldDataTest, IndexOrdering) {
    // Verify row-major layout: index = x * Ny * Nz + y * Nz + z
    Grid grid(Dim3{3, 4, 5}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);

    int counter = 0;
    for (int x = 0; x < 3; ++x) {
        for (int y = 0; y < 4; ++y) {
            for (int z = 0; z < 5; ++z) {
                field(x, y, z) = static_cast<Real>(counter);
                ++counter;
            }
        }
    }

    // Check that data is laid out contiguously in x,y,z order
    for (int i = 0; i < 60; ++i) {
        EXPECT_DOUBLE_EQ(field.data()[i], static_cast<Real>(i));
    }
}

TEST(FieldDataTest, DataPointerNonNull) {
    Grid grid(Dim3{3, 3, 3}, Spacing{1.0, 1.0, 1.0});
    FieldData field(grid);
    EXPECT_NE(field.data(), nullptr);

    const FieldData& cref = field;
    EXPECT_NE(cref.data(), nullptr);
}
