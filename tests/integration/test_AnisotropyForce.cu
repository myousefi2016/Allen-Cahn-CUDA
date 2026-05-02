#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace ac;
using namespace ac::cuda;

class AnisotropyForceTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0)
            GTEST_SKIP() << "No CUDA devices available";
        Logger::init(spdlog::level::off);
    }

    /// Build a SimulationConfig for the anisotropy force tests.
    SimulationConfig make_config(double epsilon) {
        SimulationConfig cfg;
        cfg.grid.Nx = 16;
        cfg.grid.Ny = 16;
        cfg.grid.Nz = 16;
        cfg.grid.dx = 0.4;
        cfg.grid.dy = 0.4;
        cfg.grid.dz = 0.4;
        cfg.time.dt = 0.001;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = epsilon;
        cfg.physics.W0 = 1.0;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
        cfg.boundary.u_bc = {BCType::Dirichlet, -0.8, 0.0, 0.0, 0.0, 0.0};
        cfg.validate();
        return cfg;
    }

    /// Initialize phi with a tanh-profile sphere (r0=4, W0=1) and u = -delta.
    void init_tanh_sphere(FieldData& phi, FieldData& u, const SimulationConfig& cfg) {
        int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
        Real cx = 0.5 * Nx, cy = 0.5 * Ny, cz = 0.5 * Nz;
        Real r0 = 4.0;
        Real W0 = cfg.physics.W0;
        Real inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * W0);

        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z) {
                    Real rx = x - cx, ry = y - cy, rz = z - cz;
                    Real r = std::sqrt(rx * rx + ry * ry + rz * rz);
                    phi(x, y, z) = -std::tanh((r - r0) * inv_sqrt2_W0);
                    u(x, y, z) = -cfg.physics.delta;
                }
    }
};

/// Verify that the anisotropy force is active: ε=0.10 produces different
/// evolution than ε=0.
TEST_F(AnisotropyForceTest, AnisotropyIsActive) {
    // Run one Euler step with epsilon = 0.10
    auto cfg_aniso = make_config(0.10);
    Grid grid_aniso = cfg_aniso.make_grid();
    FieldData phi_aniso(grid_aniso, "phi"), u_aniso(grid_aniso, "u");
    init_tanh_sphere(phi_aniso, u_aniso, cfg_aniso);

    CudaSolver solver_aniso(cfg_aniso);
    solver_aniso.initialize(phi_aniso, u_aniso);
    solver_aniso.step(cfg_aniso.time.dt);
    solver_aniso.copy_phi_to_host(phi_aniso);

    // Run one Euler step with epsilon = 0
    auto cfg_iso = make_config(0.0);
    Grid grid_iso = cfg_iso.make_grid();
    FieldData phi_iso(grid_iso, "phi"), u_iso(grid_iso, "u");
    init_tanh_sphere(phi_iso, u_iso, cfg_iso);

    CudaSolver solver_iso(cfg_iso);
    solver_iso.initialize(phi_iso, u_iso);
    solver_iso.step(cfg_iso.time.dt);
    solver_iso.copy_phi_to_host(phi_iso);

    // The two fields must differ — anisotropy should have an effect.
    double max_diff = 0.0;
    int Nx = cfg_aniso.grid.Nx, Ny = cfg_aniso.grid.Ny, Nz = cfg_aniso.grid.Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                double diff = std::abs(phi_aniso(x, y, z) - phi_iso(x, y, z));
                if (diff > max_diff)
                    max_diff = diff;
            }

    EXPECT_GT(max_diff, 1e-10) << "Anisotropy (epsilon=0.10) had no effect on phi evolution";
}

/// Verify that growth rate is anisotropic: the <100> direction should grow
/// faster than the <111> direction because A(100) = 1+ε > A(111) = 1 - 5ε/3.
TEST_F(AnisotropyForceTest, DirectionalGrowthRate) {
    auto cfg = make_config(0.10);
    Grid grid = cfg.make_grid();

    // Save initial phi
    FieldData phi_init(grid, "phi_init"), u_init(grid, "u_init");
    init_tanh_sphere(phi_init, u_init, cfg);

    // Evolve
    FieldData phi(grid, "phi"), u(grid, "u");
    init_tanh_sphere(phi, u, cfg);

    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    // Run a few Euler steps to accumulate a measurable anisotropy signal
    for (int step = 0; step < 5; ++step) {
        solver.step(cfg.time.dt);
    }
    solver.copy_phi_to_host(phi);

    int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
    int cx = Nx / 2, cy = Ny / 2, cz = Nz / 2;

    // Measure |Δφ| along the (1,0,0) direction: sample a point at the
    // interface along x-axis (roughly at x=cx+4, y=cy, z=cz).
    // We scan a few points near the expected interface radius to find the
    // maximum |Δφ|.
    auto max_delta_phi_along = [&](int dx, int dy, int dz) -> double {
        double max_dphi = 0.0;
        // Normalize direction and walk from radius 2 to 6
        double len = std::sqrt(double(dx * dx + dy * dy + dz * dz));
        for (int t = 2; t <= 6; ++t) {
            int px = cx + static_cast<int>(std::round(t * dx / len));
            int py = cy + static_cast<int>(std::round(t * dy / len));
            int pz = cz + static_cast<int>(std::round(t * dz / len));
            if (px >= 0 && px < Nx && py >= 0 && py < Ny && pz >= 0 && pz < Nz) {
                double dphi = std::abs(phi(px, py, pz) - phi_init(px, py, pz));
                if (dphi > max_dphi)
                    max_dphi = dphi;
            }
        }
        return max_dphi;
    };

    // <100> direction: along x-axis
    double dphi_100 = max_delta_phi_along(1, 0, 0);

    // <111> direction: along (1,1,1) diagonal
    double dphi_111 = max_delta_phi_along(1, 1, 1);

    // Both should have some growth
    EXPECT_GT(dphi_100, 1e-12) << "No measurable growth along <100>";
    EXPECT_GT(dphi_111, 1e-12) << "No measurable growth along <111>";

    // The <100> direction should grow faster: A(100)=1+ε > A(111)=1-5ε/3
    EXPECT_GT(dphi_100, dphi_111) << "Expected faster growth along <100> than <111>; "
                                  << "dphi_100=" << dphi_100 << " dphi_111=" << dphi_111;
}

/// Verify that all fields remain finite after a step with anisotropy.
TEST_F(AnisotropyForceTest, FieldsRemainFinite) {
    auto cfg = make_config(0.10);
    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    init_tanh_sphere(phi, u, cfg);

    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    solver.step(cfg.time.dt);
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z) {
                EXPECT_FALSE(std::isnan(phi(x, y, z)))
                    << "NaN in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(phi(x, y, z)))
                    << "Inf in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isnan(u(x, y, z)))
                    << "NaN in u at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(u(x, y, z)))
                    << "Inf in u at (" << x << "," << y << "," << z << ")";
            }
}
