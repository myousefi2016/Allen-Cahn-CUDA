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
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) GTEST_SKIP() << "No CUDA devices available";
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

TEST_F(BoundaryConditionsTest, RobinBC)
{
    // Robin: alpha*u + beta*du/dn = gamma
    // Test with alpha=1, beta=1, gamma=0 on a uniform field u=2.0
    // At X-lo face (side=0): du/dn ~ (u_bnd - u_inner)/(-1*dx) => sign=-1
    // u_bnd*(alpha + beta/(sign*dx)) = gamma + beta*u_inner/(sign*dx)
    // u_bnd*(1 + 1/(-1)) = 0 + 1*2/(-1) = -2
    // denom = 1 - 1 = 0 => falls back to u_inner = 2.0 (singular case)
    //
    // Instead, test with alpha=1, beta=0.5, gamma=1.0, dx=1.0:
    // X-lo (sign=-1): u_bnd*(1 + 0.5/(-1)) = 1.0 + 0.5*u_inner/(-1)
    // u_bnd*(0.5) = 1.0 - 0.5*u_inner
    // u_bnd = (1.0 - 0.5*u_inner) / 0.5 = 2.0 - u_inner
    // With u_inner=2.0: u_bnd = 2.0 - 2.0 = 0.0

    const int N = 8;
    const double h = 1.0;
    const double alpha = 1.0, beta = 0.5, gamma_val = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize field with uniform value 2.0
    std::vector<double> field_host(total, 2.0);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p,
                                BCType::Robin, 0.0, 0.0,
                                alpha, beta, gamma_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check interior is untouched
    for (int x = 1; x < N-1; ++x)
        for (int y = 1; y < N-1; ++y)
            for (int z = 1; z < N-1; ++z)
                EXPECT_DOUBLE_EQ(field_host[x*N*N + y*N + z], 2.0);

    // Check X-lo face: sign=-1, u_inner = field[1,y,z] = 2.0
    // denom = alpha + beta/(sign*ds) = 1.0 + 0.5/(-1.0) = 0.5
    // u_bnd = (gamma + beta*u_inner/(sign*ds)) / denom
    //       = (1.0 + 0.5*2.0/(-1.0)) / 0.5 = (1.0 - 1.0) / 0.5 = 0.0
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            double u_bnd = field_host[0*N*N + y*N + z];
            EXPECT_NEAR(u_bnd, 0.0, 1e-12)
                << "Robin X-lo at y=" << y << " z=" << z;
        }

    // Check X-hi face: sign=+1, u_inner = field[N-2,y,z] = 2.0
    // denom = 1.0 + 0.5/(1.0) = 1.5
    // u_bnd = (1.0 + 0.5*2.0/(1.0)) / 1.5 = (1.0 + 1.0) / 1.5 = 4/3
    double expected_hi = (gamma_val + beta * 2.0 / (1.0 * h)) / (alpha + beta / (1.0 * h));
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            double u_bnd = field_host[(N-1)*N*N + y*N + z];
            EXPECT_NEAR(u_bnd, expected_hi, 1e-12)
                << "Robin X-hi at y=" << y << " z=" << z;
        }
}

TEST_F(BoundaryConditionsTest, PerFaceMixedBC)
{
    // Test per-face BCs: Dirichlet on X-lo, Neumann on X-hi
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N; p.Ny = N; p.Nz = N;
    p.dx = h; p.dy = h; p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> field_host(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x*N*N + y*N + z] = static_cast<double>(x);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create per-face BC: Dirichlet(-5.0) on X-lo, Neumann(0) on X-hi,
    // Periodic on Y faces, Dirichlet(-1.0) on Z faces
    PerFaceBoundary face_bcs;
    face_bcs.faces[0] = {BCType::Dirichlet, -5.0, 0.0, 1.0, 0.0, 0.0};  // X-lo
    face_bcs.faces[1] = {BCType::Neumann, 0.0, 0.0, 1.0, 0.0, 0.0};     // X-hi
    face_bcs.faces[2] = {BCType::Periodic, 0.0, 0.0, 1.0, 0.0, 0.0};    // Y-lo
    face_bcs.faces[3] = {BCType::Periodic, 0.0, 0.0, 1.0, 0.0, 0.0};    // Y-hi
    face_bcs.faces[4] = {BCType::Dirichlet, -1.0, 0.0, 1.0, 0.0, 0.0};  // Z-lo
    face_bcs.faces[5] = {BCType::Dirichlet, -1.0, 0.0, 1.0, 0.0, 0.0};  // Z-hi

    launch_boundary_conditions_per_face(d_field.data(), p, face_bcs);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check X-lo: Dirichlet = -5.0
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z)
            EXPECT_DOUBLE_EQ(field_host[0*N*N + y*N + z], -5.0);

    // Check X-hi: Neumann zero flux => field[N-1] = field[N-2]
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z)
            EXPECT_DOUBLE_EQ(field_host[(N-1)*N*N + y*N + z],
                             field_host[(N-2)*N*N + y*N + z]);

    // Check Z-lo: Dirichlet = -1.0
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            EXPECT_DOUBLE_EQ(field_host[x*N*N + y*N + 0], -1.0);
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
