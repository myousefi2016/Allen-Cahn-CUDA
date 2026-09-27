#include "common/Gpu.hpp"
#include "core/FieldData.hpp"
#include "core/Grid.hpp"
#include "core/SimulationConfig.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>

using namespace ac;
using namespace ac::cuda;

class AnisotropyForceTest : public ::testing::Test {
public:
    /// Config of the seed-growth runs: N^3 points at spacing h, undercooling
    /// 0.55, zero-flux phi and u = -delta on the walls.
    SimulationConfig config_for_growth(double eps, int N, double h, double dt) {
        auto cfg = make_config(eps);
        cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = h;
        cfg.physics.delta = 0.55;
        cfg.time.dt = dt;
        cfg.boundary.phi_bc = {BCType::Neumann, 0.0, 0.0, 0.0, 0.0, 0.0};
        cfg.boundary.u_bc = {BCType::Dirichlet, -0.55, 0.0, 0.0, 0.0, 0.0};
        cfg.validate();
        return cfg;
    }

    void init_sphere(FieldData& phi, FieldData& u, const SimulationConfig& cfg, double r0) {
        init_tanh_sphere(phi, u, cfg, r0);
    }

protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
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

    /// Equilibrium tanh-profile sphere of physical radius r0 centred at the
    /// domain midpoint 0.5*(N-1) (exact mirror symmetry), with u = -delta.
    void init_tanh_sphere(FieldData& phi, FieldData& u, const SimulationConfig& cfg,
                          double r0 = 1.6) {
        int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
        Real cx = 0.5 * (Nx - 1), cy = 0.5 * (Ny - 1), cz = 0.5 * (Nz - 1);
        Real inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * cfg.physics.W0);

        for (int x = 0; x < Nx; ++x)
            for (int y = 0; y < Ny; ++y)
                for (int z = 0; z < Nz; ++z) {
                    Real rx = (x - cx) * cfg.grid.dx, ry = (y - cy) * cfg.grid.dy,
                         rz = (z - cz) * cfg.grid.dz;
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

namespace {

double trilinear(const FieldData& f, double x, double y, double z) {
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y)),
              z0 = static_cast<int>(std::floor(z));
    const double tx = x - x0, ty = y - y0, tz = z - z0;
    double v = 0.0;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c)
                v += (a ? tx : 1 - tx) * (b ? ty : 1 - ty) * (c ? tz : 1 - tz) *
                     f(x0 + a, y0 + b, z0 + c);
    return v;
}

/// Distance (cells) from the centre to the first phi = 0 crossing along unit
/// direction d, linearly interpolated between samples; -1 if none found.
double interface_radius(const FieldData& phi, double c, const double d[3], int N) {
    double prev = trilinear(phi, c, c, c), s_prev = 0.0;
    for (double s = 0.05; s < 0.5 * N - 2; s += 0.05) {
        const double v = trilinear(phi, c + s * d[0], c + s * d[1], c + s * d[2]);
        if (prev > 0.0 && v <= 0.0)
            return s_prev + (s - s_prev) * prev / (prev - v);
        prev = v;
        s_prev = s;
    }
    return -1.0;
}

struct GrowthExtent {
    double r100 = 0, r111 = 0, spread100 = 0, spread111 = 0;
};

} // namespace

namespace {

/// Mean interface extent along the 6 <100> axes and the 8 <111> diagonals of
/// a seed of radius 4 grown for t = 15 at undercooling 0.55 in a box of side
/// 50.4 (N points at spacing h), plus the spread over each family (cubic
/// symmetry check).
GrowthExtent grow_seed(AnisotropyForceTest& t, double eps, int N, double h, double dt, int steps) {
    auto cfg = t.config_for_growth(eps, N, h, dt);
    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    t.init_sphere(phi, u, cfg, 4.0);
    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    for (int s = 0; s < steps; ++s)
        solver.step(dt);
    solver.copy_phi_to_host(phi);

    const double c = 0.5 * (N - 1);
    const double s3 = 1.0 / std::sqrt(3.0);
    GrowthExtent ext;
    const double axes[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    double lo = 1e30, hi = -1e30, sum = 0;
    for (const auto& d : axes) {
        const double r = interface_radius(phi, c, d, N);
        EXPECT_GT(r, 0.0) << "no interface along an axis (wall reached?)";
        lo = std::min(lo, r);
        hi = std::max(hi, r);
        sum += r;
    }
    ext.r100 = sum / 6 * h; // physical length
    ext.spread100 = (hi - lo) * h;
    lo = 1e30, hi = -1e30, sum = 0;
    for (int sx : {-1, 1})
        for (int sy : {-1, 1})
            for (int sz : {-1, 1}) {
                const double d[3] = {sx * s3, sy * s3, sz * s3};
                const double r = interface_radius(phi, c, d, N);
                EXPECT_GT(r, 0.0) << "no interface along a diagonal";
                lo = std::min(lo, r);
                hi = std::max(hi, r);
                sum += r;
            }
    ext.r111 = sum / 8 * h;
    ext.spread111 = (hi - lo) * h;
    return ext;
}

} // namespace

/// Cubic anisotropy (eps > 0) must make a growing seed extend further along
/// the <100> axes than along the <111> diagonals. The early-time local rate
/// cannot show this (in the Karma-Rappel model tau(n) = tau0 A(n)^2 makes the
/// instantaneous relaxation *faster* where A is smaller), so the test measures
/// the interface extent after the Mullins-Sekerka growth regime has set in
/// (t = 15 tau units), before any wall contact.
///
/// The eps = 0 run is the control for the grid's own anisotropy: with the
/// compact face-flux operator it is within 2% of isotropic at h = 0.8 W0
/// (GridAnisotropyVanishesUnderRefinement shows it is discretization error).
/// The <100>/<111> ratio must increase strictly with eps and cubic symmetry
/// must hold across all 6 axes and 8 diagonals. Measured on RTX 4090 (CUDA
/// 13.2): ratios 0.9875, 1.0467, 1.1478 for eps = 0, 0.02, 0.05.
TEST_F(AnisotropyForceTest, GrowthExtentFavoursAxesUnderAnisotropy) {
    const double eps_values[] = {0.0, 0.02, 0.05};
    double ratio[3];
    for (int k = 0; k < 3; ++k) {
        SCOPED_TRACE(eps_values[k]);
        const GrowthExtent ext = grow_seed(*this, eps_values[k], 64, 0.8, 0.03, 500);
        EXPECT_LT(ext.spread100, 1e-9) << "cubic symmetry broken along <100>";
        EXPECT_LT(ext.spread111, 1e-9) << "cubic symmetry broken along <111>";
        ratio[k] = ext.r100 / ext.r111;
        std::printf("eps=%.2f  R100=%.4f  R111=%.4f  R100/R111=%.4f\n", eps_values[k], ext.r100,
                    ext.r111, ratio[k]);
    }
    EXPECT_LT(std::fabs(ratio[0] - 1.0), 0.02) << "grid anisotropy at eps = 0";
    EXPECT_GT(ratio[1], ratio[0] + 0.035) << ratio[0] << " -> " << ratio[1];
    EXPECT_GT(ratio[2], ratio[1] + 0.035) << ratio[1] << " -> " << ratio[2];
    EXPECT_GT(ratio[2], 1.1) << "eps = 0.05 must clearly favour <100>";
}

/// At eps = 0 the model is isotropic, so any <100>/<111> difference in the
/// grown seed is grid anisotropy. For a consistent second-order scheme it must
/// shrink under refinement: the same physical problem at h = 0.4 (N = 127)
/// must be at least 3x closer to isotropic than at h = 0.8 (N = 64 spans the
/// same box to within half a cell), with dt scaled by h^2.
TEST_F(AnisotropyForceTest, GridAnisotropyVanishesUnderRefinement) {
    const GrowthExtent coarse = grow_seed(*this, 0.0, 64, 0.8, 0.03, 500);
    const GrowthExtent fine = grow_seed(*this, 0.0, 127, 0.4, 0.0075, 2000);
    const double dev_coarse = std::fabs(coarse.r100 / coarse.r111 - 1.0);
    const double dev_fine = std::fabs(fine.r100 / fine.r111 - 1.0);
    std::printf("eps=0  |R100/R111 - 1|: h=0.8 %.5f  h=0.4 %.5f  (R100 %.4f -> %.4f, R111 %.4f -> "
                "%.4f)\n",
                dev_coarse, dev_fine, coarse.r100, fine.r100, coarse.r111, fine.r111);
    EXPECT_LT(dev_fine, dev_coarse / 3.0);
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
