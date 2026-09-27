#include "common/TempDir.hpp"
#include "io/VTKWriter.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#ifdef AC_HAS_VTK
#include <vtkDataArray.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkStructuredGrid.h>
#include <vtkXMLStructuredGridReader.h>
#endif

using namespace ac;
namespace fs = std::filesystem;

class VTKWriterTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("vtk_test");
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

#ifdef AC_HAS_VTK
// A .vts round trip through VTK's own reader on a non-cubic grid with
// anisotropic spacing: VTK's structured index (i, j, k) of every point, derived
// from its id as i + Nx*(j + Ny*k), must match its coordinates (i*dx, j*dy,
// k*dz) and carry FieldData value (x=i, y=j, z=k). With z-fastest output this
// fails on the first point off the origin.
TEST_F(VTKWriterTest, VtsTopologyMatchesGeometryAndData) {
    const int Nx = 3, Ny = 4, Nz = 5;
    const double dx = 0.5, dy = 0.25, dz = 2.0;
    Grid grid(Dim3{Nx, Ny, Nz}, Spacing{dx, dy, dz});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "vts";
    params.frequency = 1;

    FieldData phi(grid, "phi");
    FieldData u(grid, "u");
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                phi(x, y, z) = 100.0 * x + 10.0 * y + z;
                u(x, y, z) = -(100.0 * x + 10.0 * y + z);
            }
    {
        VTKWriter writer(grid, params);
        writer.write_async(7, 0.5, phi, u);
        writer.flush();
    }

    const auto path = test_dir_ / "output_7.vts";
    ASSERT_TRUE(fs::exists(path));
    vtkNew<vtkXMLStructuredGridReader> reader;
    reader->SetFileName(path.string().c_str());
    reader->Update();
    vtkStructuredGrid* sg = reader->GetOutput();
    ASSERT_NE(sg, nullptr);

    int dims[3] = {0, 0, 0};
    sg->GetDimensions(dims);
    ASSERT_EQ(dims[0], Nx);
    ASSERT_EQ(dims[1], Ny);
    ASSERT_EQ(dims[2], Nz);
    ASSERT_EQ(sg->GetNumberOfPoints(), static_cast<vtkIdType>(Nx) * Ny * Nz);

    vtkDataArray* phi_arr = sg->GetPointData()->GetArray("phi");
    vtkDataArray* u_arr = sg->GetPointData()->GetArray("u");
    ASSERT_NE(phi_arr, nullptr);
    ASSERT_NE(u_arr, nullptr);

    for (vtkIdType id = 0; id < sg->GetNumberOfPoints(); ++id) {
        const int i = static_cast<int>(id % Nx);
        const int j = static_cast<int>((id / Nx) % Ny);
        const int k = static_cast<int>(id / (static_cast<vtkIdType>(Nx) * Ny));
        double pt[3];
        sg->GetPoint(id, pt);
        ASSERT_DOUBLE_EQ(pt[0], i * dx) << "id=" << id;
        ASSERT_DOUBLE_EQ(pt[1], j * dy) << "id=" << id;
        ASSERT_DOUBLE_EQ(pt[2], k * dz) << "id=" << id;
        ASSERT_DOUBLE_EQ(phi_arr->GetTuple1(id), phi(i, j, k)) << "id=" << id;
        ASSERT_DOUBLE_EQ(u_arr->GetTuple1(id), u(i, j, k)) << "id=" << id;
    }
}
#endif

namespace {

void fill(FieldData& phi, FieldData& u) {
    for (std::size_t i = 0; i < phi.size(); ++i) {
        phi.data()[i] = std::sin(0.01 * static_cast<double>(i));
        u.data()[i] = -0.5;
    }
}

} // namespace

// output.async_io = false: write_async() returns only once the snapshot is on
// disk (nothing pending, file complete). Before, the flag was parsed but never
// read and every write was asynchronous.
TEST_F(VTKWriterTest, SynchronousModeReturnsOnlyWhenWritten) {
    const int N = 48;
    Grid grid(Dim3{N, N, N}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "raw";
    params.async_io = false;
    VTKWriter writer(grid, params);
    FieldData phi(grid, "phi"), u(grid, "u");
    fill(phi, u);

    for (int step = 0; step < 5; ++step) {
        writer.write_async(step, 0.1 * step, phi, u);
        EXPECT_EQ(writer.pending_jobs(), 0) << "step " << step;
        const auto f = test_dir_ / ("output_" + std::to_string(step) + "_u.raw");
        ASSERT_TRUE(fs::exists(f)) << f;
        EXPECT_EQ(fs::file_size(f), static_cast<std::uintmax_t>(N) * N * N * sizeof(double));
    }
}

// A failed write must surface: flush() and every later write_async() throw.
// Before, the writer thread logged the error and the run finished "fine".
TEST_F(VTKWriterTest, WriteFailureIsReportedByFlushAndNextWrite) {
    for (const char* format : {"raw", "vts"}) {
        SCOPED_TRACE(format);
        const auto dir = test_dir_ / format;
        Grid grid(Dim3{8, 8, 8}, Spacing{1.0, 1.0, 1.0});
        OutputParams params;
        params.output_dir = dir;
        params.format = format;
        VTKWriter writer(grid, params);
        FieldData phi(grid, "phi"), u(grid, "u");
        fill(phi, u);

        // Replace the output directory by a regular file: every write fails.
        fs::remove_all(dir);
        std::ofstream(dir) << "not a directory";

        writer.write_async(7, 0.0, phi, u);
        try {
            writer.flush();
            ADD_FAILURE() << "flush() did not report the failed write";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find("step 7"), std::string::npos) << e.what();
        }
        EXPECT_THROW(writer.write_async(8, 0.0, phi, u), std::runtime_error);
    }
}

#ifdef AC_HAS_VTK
// A VTK write that fails part-way (here: the temporary file is /dev/full, so
// every write returns ENOSPC) must not be renamed into place as if complete.
// Before, Write()'s result was ignored and the broken file was published.
TEST_F(VTKWriterTest, FailedVtkWriteIsNotPublished) {
    ASSERT_TRUE(fs::exists("/dev/full")) << "the test needs /dev/full";
    Grid grid(Dim3{8, 8, 8}, Spacing{1.0, 1.0, 1.0});
    OutputParams params;
    params.output_dir = test_dir_;
    params.format = "vts";
    VTKWriter writer(grid, params);
    FieldData phi(grid, "phi"), u(grid, "u");
    fill(phi, u);

    fs::create_symlink("/dev/full", test_dir_ / "output_0.vts.tmp");
    writer.write_async(0, 0.0, phi, u);
    EXPECT_THROW(writer.flush(), std::runtime_error);
    EXPECT_FALSE(fs::exists(fs::symlink_status(test_dir_ / "output_0.vts")))
        << "a failed write was published as output_0.vts";
}
#endif
