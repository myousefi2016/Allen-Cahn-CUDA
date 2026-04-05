#include "io/CheckpointIO.hpp"
#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>

using namespace ac;

class CheckpointIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = std::filesystem::temp_directory_path() / "ac_checkpoint_test";
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    std::filesystem::path test_dir_;
};

TEST_F(CheckpointIOTest, WriteAndRead)
{
    Grid grid(Dim3{8, 8, 8}, Spacing{0.5, 0.5, 0.5});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    // Fill with known data
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z) {
                phi(x, y, z) = std::sin(x * 0.1) * std::cos(y * 0.2) * std::sin(z * 0.3);
                u(x, y, z) = x + y * 0.1 + z * 0.01;
            }

    auto path = test_dir_ / "test_checkpoint.acbin";
    CheckpointIO::write(path, 42, 1.234, 0.005, grid, phi, u);

    ASSERT_TRUE(std::filesystem::exists(path));
    EXPECT_TRUE(CheckpointIO::is_valid_checkpoint(path));

    auto data = CheckpointIO::read(path);
    EXPECT_EQ(data.step, 42);
    EXPECT_DOUBLE_EQ(data.time, 1.234);
    EXPECT_DOUBLE_EQ(data.dt, 0.005);
    EXPECT_EQ(data.grid.Nx(), 8);
    EXPECT_EQ(data.grid.Ny(), 8);
    EXPECT_EQ(data.grid.Nz(), 8);
    EXPECT_DOUBLE_EQ(data.grid.dx(), 0.5);

    // Verify field data integrity
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            for (int z = 0; z < 8; ++z) {
                EXPECT_DOUBLE_EQ(data.phi(x, y, z), phi(x, y, z));
                EXPECT_DOUBLE_EQ(data.u(x, y, z), u(x, y, z));
            }
}

TEST_F(CheckpointIOTest, InvalidFile)
{
    auto path = test_dir_ / "not_a_checkpoint.bin";
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << "garbage data";
    }
    EXPECT_FALSE(CheckpointIO::is_valid_checkpoint(path));
    EXPECT_THROW(CheckpointIO::read(path), std::runtime_error);
}

TEST_F(CheckpointIOTest, MissingFile)
{
    EXPECT_FALSE(CheckpointIO::is_valid_checkpoint("/nonexistent/file.acbin"));
}

TEST_F(CheckpointIOTest, LargerGrid)
{
    Grid grid(Dim3{16, 16, 16}, Spacing{0.3, 0.3, 0.3});
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    phi.fill(1.0);
    u.fill(-0.5);

    auto path = test_dir_ / "large.acbin";
    CheckpointIO::write(path, 100, 5.0, 0.01, grid, phi, u);

    auto data = CheckpointIO::read(path);
    EXPECT_EQ(data.step, 100);
    EXPECT_EQ(data.phi.size(), 16u * 16 * 16);

    for (std::size_t i = 0; i < data.phi.size(); ++i) {
        EXPECT_DOUBLE_EQ(data.phi.data()[i], 1.0);
        EXPECT_DOUBLE_EQ(data.u.data()[i], -0.5);
    }
}
