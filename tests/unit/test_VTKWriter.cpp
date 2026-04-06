#include "io/VTKWriter.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

using namespace ac;
namespace fs = std::filesystem;

class VTKWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = fs::temp_directory_path() / "vtk_test";
        fs::create_directories(test_dir_);
    }

    void TearDown() override { fs::remove_all(test_dir_); }

    fs::path test_dir_;
};

TEST_F(VTKWriterTest, WritesRawFiles) {
    Grid grid(Dim3{4, 4, 4}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "raw";
    params.frequency = 1;

    VTKWriter writer(grid, params);

    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z) {
                phi(x, y, z) = 1.0;
                u(x, y, z) = -0.5;
            }

    writer.write_async(0, 0.0, phi, u);
    writer.flush();

    EXPECT_TRUE(fs::exists(test_dir_ / "output_0_phi.raw"));
    EXPECT_TRUE(fs::exists(test_dir_ / "output_0_u.raw"));

    auto phi_size = fs::file_size(test_dir_ / "output_0_phi.raw");
    EXPECT_EQ(phi_size, 64 * sizeof(double));
}

TEST_F(VTKWriterTest, MultipleAsyncWrites) {
    Grid grid(Dim3{4, 4, 4}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "raw";
    params.frequency = 1;

    VTKWriter writer(grid, params);
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    for (int step = 0; step < 5; ++step) {
        writer.write_async(step, step * 0.1, phi, u);
    }
    writer.flush();

    for (int step = 0; step < 5; ++step) {
        EXPECT_TRUE(fs::exists(test_dir_ / ("output_" + std::to_string(step) + "_phi.raw")));
    }
}

TEST_F(VTKWriterTest, PendingJobsCount) {
    Grid grid(Dim3{4, 4, 4}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "raw";
    params.frequency = 1;

    VTKWriter writer(grid, params);
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");

    // After flush, should have 0 pending
    writer.flush();
    EXPECT_EQ(writer.pending_jobs(), 0);
}

TEST_F(VTKWriterTest, StatisticsCaching) {
    Grid grid(Dim3{4, 4, 4}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "raw";
    params.frequency = 1;

    VTKWriter writer(grid, params);
    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    for (int x = 0; x < 4; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 4; ++z) {
                phi(x, y, z) = static_cast<double>(x) / 3.0;
                u(x, y, z) = -1.0 + static_cast<double>(y) / 3.0;
            }

    writer.write_async(10, 1.0, phi, u);
    writer.flush();

    // Statistics should be cached after write
    auto phi_stats = writer.get_cached_stats(10, "phi");
    ASSERT_TRUE(phi_stats.has_value());
    EXPECT_NEAR(phi_stats->min_val, 0.0, 1e-12);
    EXPECT_NEAR(phi_stats->max_val, 1.0, 1e-12);
}
