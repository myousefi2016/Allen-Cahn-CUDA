#include "common/Gpu.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaUtils.cuh"
#include "cuda/DeviceField.cuh"
#include "cuda/Kernels.cuh"
#include "cuda/WillmoreSolve.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace ac;
using namespace ac::cuda;

class BoundaryConditionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

TEST_F(BoundaryConditionsTest, DirichletBC) {
    const int N = 16;
    const double h = 1.0;
    const double bc_value = -1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize field with 1.0 everywhere
    std::vector<double> field_host(total, 1.0);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply Dirichlet BC
    launch_boundary_conditions(d_field.data(), p, BCType::Dirichlet, bc_value, 0.0, 0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download
    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify: boundaries should be -1.0, interior should be 1.0
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                double val = field_host[x * N * N + y * N + z];
                if (x == 0 || x == N - 1 || y == 0 || y == N - 1 || z == 0 || z == N - 1) {
                    EXPECT_DOUBLE_EQ(val, bc_value)
                        << "Boundary point (" << x << "," << y << "," << z << ")";
                } else {
                    EXPECT_DOUBLE_EQ(val, 1.0)
                        << "Interior point (" << x << "," << y << "," << z << ")";
                }
            }
}

TEST_F(BoundaryConditionsTest, NeumannZeroFlux) {
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize with linear field in x: f(x,y,z) = x
    std::vector<double> field_host(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x * N * N + y * N + z] = static_cast<double>(x);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply Neumann BC (zero flux)
    launch_boundary_conditions(d_field.data(), p, BCType::Neumann, 0.0, 0.0, 0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // For zero-flux Neumann on X-faces:
    // x=0 face: field[0,y,z] = field[1,y,z]
    // x=N-1 face: field[N-1,y,z] = field[N-2,y,z]
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            EXPECT_DOUBLE_EQ(field_host[0 * N * N + y * N + z], field_host[1 * N * N + y * N + z])
                << "X-lo face at y=" << y << " z=" << z;
            EXPECT_DOUBLE_EQ(field_host[(N - 1) * N * N + y * N + z],
                             field_host[(N - 2) * N * N + y * N + z])
                << "X-hi face at y=" << y << " z=" << z;
        }
}

TEST_F(BoundaryConditionsTest, PeriodicBC) {
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize interior with recognizable pattern
    std::vector<double> field_host(total, 0.0);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x * N * N + y * N + z] = x * 100.0 + y * 10.0 + z;

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p, BCType::Periodic, 0.0, 0.0, 0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // For periodic BC on X:
    // x=0 should equal x=N-2 (the second-to-last plane)
    // x=N-1 should equal x=1 (the second plane)
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            EXPECT_DOUBLE_EQ(field_host[0 * N * N + y * N + z],
                             field_host[(N - 2) * N * N + y * N + z])
                << "Periodic X-lo at y=" << y << " z=" << z;
            EXPECT_DOUBLE_EQ(field_host[(N - 1) * N * N + y * N + z],
                             field_host[1 * N * N + y * N + z])
                << "Periodic X-hi at y=" << y << " z=" << z;
        }
}

TEST_F(BoundaryConditionsTest, RobinBC) {
    // Robin: alpha*u + beta*du/dn = gamma with n the OUTWARD normal, as for
    // Neumann. On every face du/dn ~ (u_bnd - u_inner)/ds (at the lo face the
    // outward normal is -x, so du/dn = -(u[1]-u[0])/ds = (u[0]-u[1])/ds), hence
    //   u_bnd = (gamma + beta*u_inner/ds) / (alpha + beta/ds).
    // alpha=1, beta=0.5, gamma=1, ds=1, u_inner=2 gives 4/3 on all six faces;
    // lo and hi faces must agree by mirror symmetry of the uniform problem.

    const int N = 8;
    const double h = 1.0;
    const double alpha = 1.0, beta = 0.5, gamma_val = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    // Initialize field with uniform value 2.0
    std::vector<double> field_host(total, 2.0);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p, BCType::Robin, 0.0, 0.0, alpha, beta, gamma_val);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check interior is untouched
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z)
                EXPECT_DOUBLE_EQ(field_host[x * N * N + y * N + z], 2.0);

    // Face interiors only: edge/corner cells are shared between faces and
    // follow the Z, Y, X ownership order, not a single face's formula.
    const double expected = (gamma_val + beta * 2.0 / h) / (alpha + beta / h); // 4/3
    auto at = [&](int x, int y, int z) { return field_host[(x * N + y) * N + z]; };
    for (int a = 1; a < N - 1; ++a)
        for (int b = 1; b < N - 1; ++b) {
            EXPECT_NEAR(at(0, a, b), expected, 1e-12) << "X-lo " << a << "," << b;
            EXPECT_NEAR(at(N - 1, a, b), expected, 1e-12) << "X-hi " << a << "," << b;
            EXPECT_NEAR(at(a, 0, b), expected, 1e-12) << "Y-lo " << a << "," << b;
            EXPECT_NEAR(at(a, N - 1, b), expected, 1e-12) << "Y-hi " << a << "," << b;
            EXPECT_NEAR(at(a, b, 0), expected, 1e-12) << "Z-lo " << a << "," << b;
            EXPECT_NEAR(at(a, b, N - 1), expected, 1e-12) << "Z-hi " << a << "," << b;
        }
}

// Each Robin face must use its own adjacent interior cell. With u = x the
// X-lo neighbour is u[1] = 1 and the X-hi neighbour is u[N-2] = N-2.
TEST_F(BoundaryConditionsTest, RobinBCUsesAdjacentInteriorCell) {
    const int N = 8;
    const double h = 0.5;
    const double alpha = 2.0, beta = 0.25, gamma_val = -1.0;

    KernelParams p{};
    p.Nx = p.Ny = p.Nz = N;
    p.dx = p.dy = p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> field_host(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[(x * N + y) * N + z] = static_cast<double>(x);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p, BCType::Robin, 0.0, 0.0, alpha, beta, gamma_val);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto robin = [&](double u_inner) {
        return (gamma_val + beta * u_inner / h) / (alpha + beta / h);
    };
    for (int y = 1; y < N - 1; ++y)
        for (int z = 1; z < N - 1; ++z) {
            EXPECT_NEAR(field_host[(0 * N + y) * N + z], robin(1.0), 1e-12);
            EXPECT_NEAR(field_host[((N - 1) * N + y) * N + z], robin(N - 2.0), 1e-12);
        }
}

// A Robin face with a small denominator is still that Robin face: alpha =
// 1e-31, beta = 0 is the Dirichlet value gamma/alpha = 2 (2e-31 is exactly
// twice the double nearest 1e-31). The fill scales the coefficients by a
// power of two (scaled_robin) and compares them with no size threshold.
TEST_F(BoundaryConditionsTest, RobinWithSmallDenominatorKeepsItsValue) {
    const int N = 6;
    KernelParams p{};
    p.Nx = p.Ny = p.Nz = N;
    p.dx = p.dy = p.dz = 0.4;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> field_host(total, -0.25);
    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p, BCType::Robin, 0.0, 0.0, 1e-31, 0.0, 2e-31);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int a = 1; a < N - 1; ++a)
        for (int b = 1; b < N - 1; ++b)
            for (int side : {0, N - 1}) {
                EXPECT_EQ(field_host[(side * N + a) * N + b], 2.0) << "X face " << a << "," << b;
                EXPECT_EQ(field_host[(a * N + side) * N + b], 2.0) << "Y face " << a << "," << b;
                EXPECT_EQ(field_host[(a * N + b) * N + side], 2.0) << "Z face " << a << "," << b;
            }
}

// Robin faces at the edges of the double range fill their ghosts like the
// boundary conditions they are: zero flux with beta = 1e308 (unscaled, beta/h
// overflows to inf at h = 0.4, and the ghost is 0, or NaN where beta*u/h
// overflows too, |u| > 0.72) and with the smallest subnormal beta at h = 4
// (unscaled, beta/h underflows to 0 and the ghost is NaN), the Dirichlet value
// 1 from 1e-300 u = 1e-300, and the Dirichlet value 2^1024/1.5 = 1.2e308 from
// 1.5 * 2^-10 u = 2^1014 (whose gamma a scaling with a denominator above 1
// would overflow).
TEST_F(BoundaryConditionsTest, RobinAtTheEdgesOfTheDoubleRange) {
    const int N = 6;
    // Zero flux (alpha = gamma = 0): the ghost is u (1+d1)(1+d2)(1+d4)/(1+d3),
    // the roundings of beta*u, of its division by h, of beta/h and of the
    // quotient, so |ghost - u| <= gamma_4 |u| (Higham, Lemma 3.1). Dirichlet
    // (beta = 0): beta*u/h is zero, so the ghost is gamma/alpha in one
    // rounding, the bits of the host's gamma/alpha.
    const double unit = std::numeric_limits<double>::epsilon() / 2;
    const double gamma_4 = 4 * unit / (1 - 4 * unit);
    struct Case {
        double h, alpha, beta, gamma;
        bool zero_flux;
    };
    for (const Case c :
         {Case{0.4, 0.0, 1e308, 0.0, true}, Case{4.0, 0.0, 5e-324, 0.0, true},
          Case{0.4, 1e-300, 0.0, 1e-300, false}, Case{0.4, 0x1.8p-10, 0.0, 0x1p1014, false}}) {
        std::ostringstream trace;
        trace << std::setprecision(17) << "h " << c.h << " alpha " << c.alpha << " beta " << c.beta
              << " gamma " << c.gamma;
        SCOPED_TRACE(trace.str());
        KernelParams p{};
        p.Nx = p.Ny = p.Nz = N;
        p.dx = p.dy = p.dz = c.h;
        const std::size_t total = static_cast<std::size_t>(N) * N * N;
        std::vector<double> field(total);
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z)
                    field[(x * N + y) * N + z] = -0.8 + 0.05 * x;
        DeviceField<double> d_field(total);
        d_field.copy_from_host(field.data());
        launch_boundary_conditions(d_field.data(), p, BCType::Robin, 0.0, 0.0, c.alpha, c.beta,
                                   c.gamma);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_field.copy_to_host(field.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        auto at = [&](int x, int y, int z) { return field[(x * N + y) * N + z]; };
        for (int a = 1; a < N - 1; ++a)
            for (int b = 1; b < N - 1; ++b) {
                if (c.zero_flux) {
                    const double lo = at(1, a, b), hi = at(N - 2, a, b);
                    EXPECT_NEAR(at(0, a, b), lo, gamma_4 * std::fabs(lo)) << a << "," << b;
                    EXPECT_NEAR(at(N - 1, a, b), hi, gamma_4 * std::fabs(hi)) << a << "," << b;
                } else {
                    EXPECT_EQ(at(0, a, b), c.gamma / c.alpha) << a << "," << b;
                    EXPECT_EQ(at(N - 1, a, b), c.gamma / c.alpha) << a << "," << b;
                }
            }
    }
}

// robin_ghost_coefficient, the factor c of ghost = c * inner with which the
// Willmore solve's operator and spectral bounds treat a Robin face, is the
// factor the fill applies with gamma = 0, bit for bit on all six faces: the
// fill of a field of ones computes (beta' * 1/h)/(alpha' + beta'/h) with the
// same operations. Ordinary and extreme coefficients alike (unscaled, c is NaN
// for beta = 1e308 at h = 0.4 and for beta = 5e-324 at h = 4); c lies in
// [0, 1] since alpha*beta >= 0.
TEST_F(BoundaryConditionsTest, RobinGhostCoefficientIsTheFillsFactor) {
    const int N = 4;
    struct Case {
        double h, alpha, beta;
    };
    for (const Case c : {Case{0.4, 1.0, 0.5}, Case{0.4, 0.05, 1.0}, Case{0.4, 20.0, 0.1},
                         Case{0.4, -1.0, -0.5}, Case{0.4, 0.0, 1e308}, Case{4.0, 0.0, 5e-324},
                         Case{0.4, 1e-300, 0.0}, Case{0.4, 0x1.8p-10, 1e300}}) {
        std::ostringstream trace;
        trace << std::setprecision(17) << "h " << c.h << " alpha " << c.alpha << " beta " << c.beta;
        SCOPED_TRACE(trace.str());
        KernelParams p{};
        p.Nx = p.Ny = p.Nz = N;
        p.dx = p.dy = p.dz = c.h;
        const std::size_t total = static_cast<std::size_t>(N) * N * N;
        std::vector<double> field(total, 1.0);
        DeviceField<double> d_field(total);
        d_field.copy_from_host(field.data());
        launch_boundary_conditions(d_field.data(), p, BCType::Robin, 0.0, 0.0, c.alpha, c.beta,
                                   0.0);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_field.copy_to_host(field.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        const double coef = robin_ghost_coefficient(c.alpha, c.beta, c.h);
        EXPECT_GE(coef, 0.0);
        EXPECT_LE(coef, 1.0);
        auto at = [&](int x, int y, int z) { return field[(x * N + y) * N + z]; };
        for (int a = 1; a < N - 1; ++a)
            for (int b = 1; b < N - 1; ++b)
                for (int side : {0, N - 1}) {
                    EXPECT_EQ(at(side, a, b), coef) << "x face " << side;
                    EXPECT_EQ(at(a, side, b), coef) << "y face " << side;
                    EXPECT_EQ(at(a, b, side), coef) << "z face " << side;
                }
    }
}

TEST_F(BoundaryConditionsTest, PerFaceMixedBC) {
    // Test per-face BCs: Dirichlet on X-lo, Neumann on X-hi
    const int N = 8;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;

    std::vector<double> field_host(total);
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z)
                field_host[x * N * N + y * N + z] = static_cast<double>(x);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create per-face BC: Dirichlet(-5.0) on X-lo, Neumann(0) on X-hi,
    // Periodic on Y faces, Dirichlet(-1.0) on Z faces
    PerFaceBoundary face_bcs;
    face_bcs.faces[0] = {BCType::Dirichlet, -5.0, 0.0, 1.0, 0.0, 0.0}; // X-lo
    face_bcs.faces[1] = {BCType::Neumann, 0.0, 0.0, 1.0, 0.0, 0.0};    // X-hi
    face_bcs.faces[2] = {BCType::Periodic, 0.0, 0.0, 1.0, 0.0, 0.0};   // Y-lo
    face_bcs.faces[3] = {BCType::Periodic, 0.0, 0.0, 1.0, 0.0, 0.0};   // Y-hi
    face_bcs.faces[4] = {BCType::Dirichlet, -1.0, 0.0, 1.0, 0.0, 0.0}; // Z-lo
    face_bcs.faces[5] = {BCType::Dirichlet, -1.0, 0.0, 1.0, 0.0, 0.0}; // Z-hi

    launch_boundary_conditions_per_face(d_field.data(), p, face_bcs);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Check X-lo: Dirichlet = -5.0
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z)
            EXPECT_DOUBLE_EQ(field_host[0 * N * N + y * N + z], -5.0);

    // Check X-hi: Neumann zero flux => field[N-1] = field[N-2]
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z)
            EXPECT_DOUBLE_EQ(field_host[(N - 1) * N * N + y * N + z],
                             field_host[(N - 2) * N * N + y * N + z]);

    // Check Z-lo: Dirichlet = -1.0.
    // Skip x=0 (owned by X-lo Dirichlet=-5) — see ownership convention in
    // BoundaryKernels.cu (X faces are applied last and own all of their plane).
    for (int x = 1; x < N; ++x)
        for (int y = 0; y < N; ++y)
            EXPECT_DOUBLE_EQ(field_host[x * N * N + y * N + 0], -1.0)
                << "Z-lo at x=" << x << " y=" << y;
}

// Faces with more rows than a 16 x 16 launch could reach (gridDim.y <= 65535 blocks of 16
// rows: 1048560): Nz = 1048561 (the X and Y faces) and Ny = 1048561 (the Z faces). Every
// boundary cell takes its owning face's Dirichlet value (X faces own their planes, Y faces
// the rest of theirs, Z faces what is left; BoundaryKernels.cu), the interior keeps its
// value.
TEST_F(BoundaryConditionsTest, FacesWithMoreRowsThanALaunchRowLimit) {
    for (const auto& [nx, ny, nz] : {std::tuple{3, 3, 1048561}, std::tuple{3, 1048561, 3}}) {
        SCOPED_TRACE(std::to_string(nx) + " x " + std::to_string(ny) + " x " + std::to_string(nz));
        KernelParams p{};
        p.Nx = nx;
        p.Ny = ny;
        p.Nz = nz;
        p.dx = p.dy = p.dz = 1.0;
        const std::size_t total = static_cast<std::size_t>(nx) * ny * nz;
        std::vector<double> host(total, 1.0);
        DeviceField<double> d_field(total);
        d_field.copy_from_host(host.data());
        PerFaceBoundary faces;
        for (int f = 0; f < 6; ++f)
            faces.faces[f] = {BCType::Dirichlet, -1.0 - f, 0.0, 1.0, 0.0, 0.0};
        launch_boundary_conditions_per_face(d_field.data(), p, faces);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_field.copy_to_host(host.data());
        CUDA_CHECK(cudaDeviceSynchronize());
        long wrong = 0;
        for (int x = 0; x < nx; ++x)
            for (int y = 0; y < ny; ++y)
                for (int z = 0; z < nz; ++z) {
                    const double expected = x == 0        ? -1.0
                                            : x == nx - 1 ? -2.0
                                            : y == 0      ? -3.0
                                            : y == ny - 1 ? -4.0
                                            : z == 0      ? -5.0
                                            : z == nz - 1 ? -6.0
                                                          : 1.0;
                    const double got = host[(static_cast<std::size_t>(x) * ny + y) * nz + z];
                    if (got != expected && ++wrong <= 5)
                        ADD_FAILURE() << "(" << x << ", " << y << ", " << z << "): " << got
                                      << ", expected " << expected;
                }
        EXPECT_EQ(wrong, 0);
    }
}

TEST_F(BoundaryConditionsTest, InteriorUnchanged) {
    // Verify that boundary condition application doesn't touch interior points.
    const int N = 10;
    const double h = 1.0;

    KernelParams p{};
    p.Nx = N;
    p.Ny = N;
    p.Nz = N;
    p.dx = h;
    p.dy = h;
    p.dz = h;

    std::size_t total = static_cast<std::size_t>(N) * N * N;
    std::vector<double> field_host(total, 42.0);

    DeviceField<double> d_field(total);
    d_field.copy_from_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    launch_boundary_conditions(d_field.data(), p, BCType::Dirichlet, -999.0, 0.0, 0.0, 0.0, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());

    d_field.copy_to_host(field_host.data());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                EXPECT_DOUBLE_EQ(field_host[x * N * N + y * N + z], 42.0)
                    << "Interior modified at (" << x << "," << y << "," << z << ")";
            }
}
