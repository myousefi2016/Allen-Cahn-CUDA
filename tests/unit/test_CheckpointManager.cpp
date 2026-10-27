#include "common/TempDir.hpp"
#include "core/CheckpointManager.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "io/CheckpointIO.hpp"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <vector>

using namespace ac;

class CheckpointManagerTest : public ::testing::Test {
protected:
    static constexpr int N = 4;

    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("test_ckpt_mgr");
        std::filesystem::create_directories(test_dir_);

        grid_ = Grid(Dim3{N, N, N}, Spacing{1.0, 1.0, 1.0});
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

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

TEST_F(CheckpointManagerTest, ShouldCheckpoint) {
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

TEST_F(CheckpointManagerTest, SaveCreatesFile) {
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

TEST_F(CheckpointManagerTest, RollingRetention) {
    auto params = make_params(10, 2); // keep_last = 2
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

TEST_F(CheckpointManagerTest, RestoreFromExplicitFile) {
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

TEST_F(CheckpointManagerTest, RestoreLatestFromDir) {
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

TEST_F(CheckpointManagerTest, RestoreEmptyDirThrows) {
    // Create a fresh empty directory
    auto empty_dir = ac::test::unique_temp_dir("test_ckpt_empty");
    std::filesystem::create_directories(empty_dir);

    CheckpointParams params;
    params.frequency = 10;
    params.checkpoint_dir = empty_dir;
    params.keep_last = 3;
    params.restart_file = std::nullopt;

    CheckpointManager mgr(params, grid_);

    EXPECT_THROW((void)mgr.restore(), std::runtime_error);

    std::filesystem::remove_all(empty_dir);
}

// A resumed run rewrites steps that already exist in checkpoint_dir. The old
// retention kept a list seeded from the directory and appended every save,
// so the rewritten path appeared twice and, with keep_last = 1, the
// checkpoint that had just been written was deleted.
TEST_F(CheckpointManagerTest, RetentionKeepsTheCheckpointJustWritten) {
    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);
    CheckpointIO::write(test_dir_ / "checkpoint_20.acbin", 20, 0.2, 0.01, grid_, phi, u);

    CheckpointManager mgr(make_params(10, 1), grid_);
    mgr.save(20, 0.2, 0.01, phi, u);

    EXPECT_TRUE(CheckpointIO::is_valid_checkpoint(test_dir_ / "checkpoint_20.acbin"));
}

// After a restart from step 20 of a directory holding steps 10..50, the
// directory must hold one consistent trajectory: the checkpoint just written,
// keep_last - 1 older ones, and nothing above it (steps 30..50 came from the
// earlier run, and "resume from latest" would otherwise jump back to them).
TEST_F(CheckpointManagerTest, RetentionAfterRestartKeepsOneTrajectory) {
    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);
    for (int step = 10; step <= 50; step += 10)
        CheckpointIO::write(test_dir_ / ("checkpoint_" + std::to_string(step) + ".acbin"), step,
                            step * 0.01, 0.01, grid_, phi, u);

    CheckpointManager mgr(make_params(10, 2), grid_);
    auto present = [&] {
        std::vector<std::string> names;
        for (const auto& e : std::filesystem::directory_iterator(test_dir_))
            names.push_back(e.path().filename().string());
        std::sort(names.begin(), names.end());
        return names;
    };

    mgr.save(20, 0.2, 0.01, phi, u);
    EXPECT_EQ(present(), (std::vector<std::string>{"checkpoint_10.acbin", "checkpoint_20.acbin"}));
    mgr.save(30, 0.3, 0.01, phi, u);
    EXPECT_EQ(present(), (std::vector<std::string>{"checkpoint_20.acbin", "checkpoint_30.acbin"}));
}

// Only checkpoint_<digits>.acbin files are managed; anything else is left alone.
TEST_F(CheckpointManagerTest, RetentionIgnoresUnrelatedFiles) {
    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);
    for (const char* name : {"notes.txt", "checkpoint_x.acbin", "checkpoint_5.acbin.tmp"})
        std::ofstream(test_dir_ / name) << "keep";

    CheckpointManager mgr(make_params(10, 1), grid_);
    mgr.save(10, 0.1, 0.01, phi, u);
    mgr.save(20, 0.2, 0.01, phi, u);

    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_10.acbin"));
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "checkpoint_20.acbin"));
    for (const char* name : {"notes.txt", "checkpoint_x.acbin", "checkpoint_5.acbin.tmp"})
        EXPECT_TRUE(std::filesystem::exists(test_dir_ / name)) << name;
}

// Checkpoints of another grid in the same directory (another configuration's
// run) are not this trajectory: retention leaves them alone, even those with
// higher steps, and they do not count toward keep_last. Before, a 160^3 demo
// writing step 12000 deleted the 384^3 run's checkpoints 16000..30000.
TEST_F(CheckpointManagerTest, RetentionLeavesCheckpointsOfOtherGridsAlone) {
    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);
    const Grid other(Dim3{N + 2, N, N}, Spacing{1.0, 1.0, 1.0});
    const Grid spacing(Dim3{N, N, N}, Spacing{0.5, 1.0, 1.0});
    FieldData phi_o(other, "phi"), u_o(other, "u"), phi_s(spacing, "phi"), u_s(spacing, "u");
    for (int step : {16000, 24000, 30000})
        CheckpointIO::write(test_dir_ / ("checkpoint_" + std::to_string(step) + ".acbin"), step,
                            step * 0.01, 0.01, other, phi_o, u_o);
    CheckpointIO::write(test_dir_ / "checkpoint_14000.acbin", 14000, 140.0, 0.01, spacing, phi_s,
                        u_s);

    CheckpointManager mgr(make_params(4000, 1), grid_);
    mgr.save(4000, 40.0, 0.01, phi, u);
    mgr.save(8000, 80.0, 0.01, phi, u);
    mgr.save(12000, 120.0, 0.01, phi, u);

    for (int step : {14000, 16000, 24000, 30000})
        EXPECT_TRUE(
            std::filesystem::exists(test_dir_ / ("checkpoint_" + std::to_string(step) + ".acbin")))
            << step;
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_4000.acbin"));
    EXPECT_FALSE(std::filesystem::exists(test_dir_ / "checkpoint_8000.acbin"));
    EXPECT_TRUE(std::filesystem::exists(test_dir_ / "checkpoint_12000.acbin"));
}

// Removals are logged at info level, the console level of the binary: the
// user sees which checkpoint was deleted and why.
TEST_F(CheckpointManagerTest, RemovalsAreLoggedAtInfoLevel) {
    auto stream = std::make_shared<std::ostringstream>();
    auto logger = std::make_shared<spdlog::logger>(
        "capture", std::make_shared<spdlog::sinks::ostream_sink_mt>(*stream));
    logger->set_level(spdlog::level::info);
    const auto previous = spdlog::default_logger();
    spdlog::set_default_logger(logger);

    FieldData phi(grid_, "phi"), u(grid_, "u");
    fill_field(phi, 1.0);
    fill_field(u, -0.5);
    CheckpointManager mgr(make_params(10, 1), grid_);
    mgr.save(10, 0.1, 0.01, phi, u);
    mgr.save(20, 0.2, 0.01, phi, u);
    spdlog::set_default_logger(previous);

    EXPECT_NE(stream->str().find("Removed checkpoint"), std::string::npos) << stream->str();
    EXPECT_NE(stream->str().find("checkpoint_10.acbin (beyond keep_last)"), std::string::npos)
        << stream->str();
}

// A configured restart_file that does not exist must stop the run: the old
// has_restart_file() returned false, the engine cold-started from step 0 and
// its retention then rotated the real checkpoints away.
TEST_F(CheckpointManagerTest, MissingRestartFileIsAnError) {
    auto params = make_params(10);
    params.restart_file = test_dir_ / "checkpoint_999.acbin";
    CheckpointManager mgr(params, grid_);

    EXPECT_TRUE(mgr.restart_requested());
    EXPECT_THROW((void)mgr.restore(), std::runtime_error);
}
