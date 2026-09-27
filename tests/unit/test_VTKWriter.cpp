#include "common/TempDir.hpp"
#include "io/VTKWriter.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>

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
