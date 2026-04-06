#include "core/CheckpointManager.hpp"
#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "io/CheckpointIO.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>
#include <random>

using namespace ac;

class CheckpointManagerTest : public ::testing::Test {
protected:
    static constexpr int N = 4;

    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = std::filesystem::temp_directory_path() / "test_ckpt_mgr";
        std::filesystem::create_directories(test_dir_);

        grid_ = Grid(Dim3{N, N, N}, Spacing{1.0, 1.0, 1.0});
    }

    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }

    CheckpointParams make_params(int frequency = 10, int keep_last = 3) {
        CheckpointParams p;
        p.frequency = frequency;
        p.checkpoint_dir = test_dir_;
        p.keep_last = keep_last;
        p.restart_file = std::nullopt;
        return p;
    }

    void fill_field(FieldData& field, double base) {
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z)
                    field(x, y, z) = base + x * 0.1 + y * 0.01 + z * 0.001;
    }

    std::filesystem::path test_dir_;
    Grid grid_;
};

TEST_F(CheckpointManagerTest, ShouldCheckpoint)
{
    auto params = make_params(10);
    CheckpointManager mgr(params, grid_);

    // step=0 should never trigger
    EXPECT_FALSE(mgr.should_checkpoint(0));

    // Non-multiples of 10
    EXPECT_FALSE(mgr.should_checkpoint(1));
    EXPECT_FALSE(mgr.should_checkpoint(5));
    EXPECT_FALSE(mgr.should_checkpoint(9));
    EXPECT_FALSE(mgr.should_checkpoint(11));
    EXPECT_FALSE(mgr.should_checkpoint(15));

    // Exact multiples of 10
    EXPECT_TRUE(mgr.should_checkpoint(10));
    EXPECT_TRUE(mgr.should_checkpoint(20));
    EXPECT_TRUE(mgr.should_checkpoint(30));
    EXPECT_TRUE(mgr.should_checkpoint(100));

    // Frequency = 0 disables checkpointing
    auto params_disabled = make_params(0);
    CheckpointManager mgr_disabled(params_disabled, grid_);
    EXPECT_FALSE(mgr_disabled.should_checkpoint(10));
    EXPECT_FALSE(mgr_disabled.should_checkpoint(100));
}

TEST_F(CheckpointManagerTest, SaveCreatesFile)
{
    auto params = make_params(10);
    CheckpointManager mgr(params, grid_);

    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);

    mgr.save(10, 0.1, 0.01, phi, u);

    auto expected_path = test_dir_ / "checkpoint_10.acbin";
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_TRUE(CheckpointIO::is_valid_checkpoint(expected_path));
}

TEST_F(CheckpointManagerTest, RollingRetention)
{
    auto params = make_params(10, 2);  // keep_last = 2
    CheckpointManager mgr(params, grid_);

    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);

    // Save 5 checkpoints
    for (int step = 10; step <= 50; step += 10) {
        mgr.save(step, step * 0.01, 0.01, phi, u);
    }

    // Only the last 2 should remain (step 40 and 50)
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_10.acbin"));
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_20.acbin"));
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_30.acbin"));
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "checkpoint_40.acbin"));
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "checkpoint_50.acbin"));
}

TEST_F(CheckpointManagerTest, RestoreFromExplicitFile)
{
    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 7.0);
    fill_field(u, -3.0);

    auto ckpt_path = test_dir_ / "explicit_checkpoint.acbin";
    CheckpointIO::write(ckpt_path, 42, 1.234, 0.005, grid_, phi, u);

    auto params = make_params(10);
    params.restart_file = ckpt_path;
    CheckpointManager mgr(params, grid_);

    auto data = mgr.restore();
    EXPECT_EQ(data.step, 42);
    EXPECT_DOUBLE_EQ(data.time, 1.234);
    EXPECT_DOUBLE_EQ(data.dt, 0.005);

    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_DOUBLE_EQ(data.phi(x, y, z), phi(x, y, z));
                EXPECT_DOUBLE_EQ(data.u(x, y, z), u(x, y, z));
            }
}

TEST_F(CheckpointManagerTest, RestoreLatestFromDir)
{
    auto params = make_params(10);
    CheckpointManager mgr(params, grid_);

    // Save 3 checkpoints with different data
    for (int step = 10; step <= 30; step += 10) {
        FieldData phi(grid_, "phi"), u(grid_, "u");
        fill_field(phi, static_cast<double>(step));
        fill_field(u, static_cast<double>(-step));
        mgr.save(step, step * 0.01, 0.01, phi, u);
    }

    // Restore without restart_file should get the latest (step 30)
    auto data = mgr.restore();
    EXPECT_EQ(data.step, 30);
    EXPECT_DOUBLE_EQ(data.time, 0.3);

    // Verify the field data matches step=30
    FieldData expected_phi(grid_, "phi");
    fill_field(expected_phi, 30.0);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_DOUBLE_EQ(data.phi(x, y, z), expected_phi(x, y, z));
            }
}

TEST_F(CheckpointManagerTest, RestoreEmptyDirThrows)
{
    // Create a fresh empty directory
    auto empty_dir = std::filesystem::temp_directory_path() / "test_ckpt_empty";
    std::filesystem::create_directories(empty_dir);

    CheckpointParams params;
    params.frequency = 10;
    params.checkpoint_dir = empty_dir;
    params.keep_last = 3;
    params.restart_file = std::nullopt;

    CheckpointManager mgr(params, grid_);

    EXPECT_THROW(mgr.restore(), std::runtime_error);

    std::filesystem::remove_all(empty_dir);
}
