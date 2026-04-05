#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class BoundaryConditionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);
    }
};

TEST_F(BoundaryConditionsTest, DirichletBC)
{
    const int N = 16;
    const double h = 1.0;
    const double bc_value = -1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize field with 1.0 everywhere
    std::vector<double> field_host(total, 1.0);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply Dirichlet BC
    launch_boundary_conditions(d_field.data(), p,
                                BCType::Dirichlet, bc_value, 0.0,
                                0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: boundaries should be -1.0, interior should be 1.0
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                double val = field_host[x*N*N + y*N + z];
                if (x == 0 || x == N-1 || y == 0 || y == N-1 || z == 0 || z == N-1) {
                    EXPECT_DOUBLE_EQ(val, bc_value)
                        << "Boundary point (" << x << "," << y << "," << z << ")";
                } else {
                    EXPECT_DOUBLE_EQ(val, 1.0)
                        << "Interior point (" << x << "," << y << "," << z << ")";
                }
            }
}

TEST_F(BoundaryConditionsTest, NeumannZeroFlux)
{
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize with linear field in x: f(x,y,z) = x
    std::vector<double> field_host(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x*N*N + y*N + z] = static_cast<double>(x);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply Neumann BC (zero flux)
    launch_boundary_conditions(d_field.data(), p,
                                BCType::Neumann, 0.0, 0.0,
                                0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // For zero-flux Neumann on X-faces:
    // x=0 face: field[0,y,z] = field[1,y,z]
    // x=N-1 face: field[N-1,y,z] = field[N-2,y,z]
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            EXPECT_DOUBLE_EQ(field_host[0*N*N + y*N + z],
                             field_host[1*N*N + y*N + z])
                << "X-lo face at y=" << y << " z=" << z;
            EXPECT_DOUBLE_EQ(field_host[(N-1)*N*N + y*N + z],
                             field_host[(N-2)*N*N + y*N + z])
                << "X-hi face at y=" << y << " z=" << z;
        }
}

TEST_F(BoundaryConditionsTest, PeriodicBC)
{
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize interior with recognizable pattern
    std::vector<double> field_host(total, 0.0);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x*N*N + y*N + z] = x * 100.0 + y * 10.0 + z;

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p,
                                BCType::Periodic, 0.0, 0.0,
                                0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // For periodic BC on X:
    // x=0 should equal x=N-2 (the second-to-last plane)
    // x=N-1 should equal x=1 (the second plane)
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            EXPECT_DOUBLE_EQ(field_host[0*N*N + y*N + z],
                             field_host[(N-2)*N*N + y*N + z])
                << "Periodic X-lo at y=" << y << " z=" << z;
            EXPECT_DOUBLE_EQ(field_host[(N-1)*N*N + y*N + z],
                             field_host[1*N*N + y*N + z])
                << "Periodic X-hi at y=" << y << " z=" << z;
        }
}

TEST_F(BoundaryConditionsTest, InteriorUnchanged)
{
    // Verify that boundary condition application doesn't touch interior points.
    const int N = 10;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> field_host(total, 42.0);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p,
                                BCType::Dirichlet, -999.0, 0.0,
                                0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z) {
                EXPECT_DOUBLE_EQ(field_host[x*N*N + y*N + z], 42.0)
                    << "Interior modified at (" << x << "," << y << "," << z << ")";
            }
}
