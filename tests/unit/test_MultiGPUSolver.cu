#include "common/Gpu.hpp"
#include "common/TempDir.hpp"
#include "core/SimulationEngine.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/MultiGPUSolver.cuh"
#include "logging/Logger.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ac;
using namespace ac::cuda;

// MultiGPUSolver must reproduce CudaSolver bit for bit. Every owned cell is
// computed by the same kernel from the same operands once the halos, the
// periodic ghost planes and the intermediate stage buffers are exchanged
// correctly, and max reductions do not depend on evaluation order. All domains
// share device 0, which runs the whole decomposition and exchange path on a
// single GPU.

namespace {

enum class BcLayout {
    Uniform,   ///< uniform phi Neumann / u Dirichlet (non-per-face config)
    PerFace,   ///< distinct BC on every face incl. Robin, fluxes, Z periodic; 27-point stencil
    PeriodicX, ///< periodic X for both fields, walls on Y/Z
};

struct Case {
    TimeScheme scheme;
    int domains;
    BcLayout bc;
    int Nx;
};

const char* scheme_name(TimeScheme s) {
    switch (s) {
    case TimeScheme::Euler:
        return "Euler";
    case TimeScheme::Heun:
        return "Heun";
    case TimeScheme::RK4:
        return "RK4";
    case TimeScheme::IMEX:
        return "IMEX";
    }
    return "Unknown";
}

const char* bc_name(BcLayout b) {
    switch (b) {
    case BcLayout::Uniform:
        return "Uniform";
    case BcLayout::PerFace:
        return "PerFace";
    case BcLayout::PeriodicX:
        return "PeriodicX";
    }
    return "Unknown";
}

constexpr int kNy = 10;
constexpr int kNz = 9;
constexpr double kH = 0.4;

BoundaryConfig bc_of(BCType type, double value = 0.0, double flux = 0.0) {
    BoundaryConfig b;
    b.type = type;
    b.value = value;
    b.flux = flux;
    return b;
}

BoundaryConfig robin(double alpha, double beta, double gamma) {
    BoundaryConfig b;
    b.type = BCType::Robin;
    b.alpha = alpha;
    b.beta = beta;
    b.gamma = gamma;
    return b;
}

SimulationConfig make_config(const Case& c) {
    SimulationConfig cfg;
    cfg.grid.Nx = c.Nx;
    cfg.grid.Ny = kNy;
    cfg.grid.Nz = kNz;
    cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = kH;
    cfg.physics.delta = 0.55;
    cfg.physics.epsilon = 0.05;
    cfg.physics.W0 = 1.0;
    cfg.physics.D = 2.0;
    cfg.physics.d0 = 0.5;
    cfg.time.scheme = c.scheme;
    cfg.time.dt = (c.scheme == TimeScheme::IMEX) ? 0.01 : 0.004;
    cfg.stencil = StencilType::Standard7Point;
    cfg.boundary.phi_bc = bc_of(BCType::Neumann);
    cfg.boundary.u_bc = bc_of(BCType::Dirichlet, -0.55);

    switch (c.bc) {
    case BcLayout::Uniform:
        break;
    case BcLayout::PerFace: {
        cfg.stencil = StencilType::Isotropic27Point; // corner/edge reach across the cut
        cfg.boundary.per_face = true;
        auto& pf = cfg.boundary.phi_faces;
        pf[Face::XLo] = robin(1.0, 0.5, -0.9);
        pf[Face::XHi] = bc_of(BCType::Dirichlet, -0.95);
        pf[Face::YLo] = bc_of(BCType::Neumann, 0.0, 0.05);
        pf[Face::YHi] = bc_of(BCType::Neumann);
        pf[Face::ZLo] = bc_of(BCType::Periodic);
        pf[Face::ZHi] = bc_of(BCType::Periodic);
        auto& uf = cfg.boundary.u_faces;
        uf[Face::XLo] = bc_of(BCType::Dirichlet, -0.4);
        uf[Face::XHi] = robin(2.0, 1.0, -1.1);
        uf[Face::YLo] = bc_of(BCType::Neumann, 0.0, -0.02);
        uf[Face::YHi] = bc_of(BCType::Dirichlet, -0.6);
        uf[Face::ZLo] = bc_of(BCType::Periodic);
        uf[Face::ZHi] = bc_of(BCType::Periodic);
        // The uniform fields are never used in per-face mode; make any
        // accidental use visible.
        cfg.boundary.phi_bc = bc_of(BCType::Dirichlet, 0.77);
        cfg.boundary.u_bc = bc_of(BCType::Dirichlet, 0.77);
        break;
    }
    case BcLayout::PeriodicX:
        cfg.boundary.per_face = true;
        cfg.boundary.phi_faces = PerFaceBoundary::uniform(bc_of(BCType::Neumann));
        cfg.boundary.u_faces = PerFaceBoundary::uniform(bc_of(BCType::Dirichlet, -0.55));
        cfg.boundary.phi_faces[Face::XLo] = bc_of(BCType::Periodic);
        cfg.boundary.phi_faces[Face::XHi] = bc_of(BCType::Periodic);
        cfg.boundary.u_faces[Face::XLo] = bc_of(BCType::Periodic);
        cfg.boundary.u_faces[Face::XHi] = bc_of(BCType::Periodic);
        break;
    }
    return cfg;
}

/// Off-centre tanh seed next to the x_lo wall (so it also crosses the periodic
/// seam) plus a smooth non-uniform u, so every domain cut sees gradients in
/// both fields.
void make_ic(FieldData& phi, FieldData& u) {
    const int Nx = phi.Nx();
    const double cx = 0.3 * Nx, cy = 0.45 * kNy, cz = 0.55 * kNz;
    const double r0 = 2.0;
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < kNy; ++y)
            for (int z = 0; z < kNz; ++z) {
                const double rx = (x - cx) * kH, ry = (y - cy) * kH, rz = (z - cz) * kH;
                const double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                phi(x, y, z) = -std::tanh((r - r0) / std::sqrt(2.0));
                u(x, y, z) = -0.55 + 0.1 * std::sin(0.7 * x + 1.3 * y) * std::cos(0.9 * z);
            }
}

std::uint64_t bits(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

/// Number of cells whose bit patterns differ; describes the first one.
std::size_t count_mismatches(const FieldData& a, const FieldData& b, std::string& first) {
    std::size_t n = 0;
    const int Nx = a.Nx(), Ny = a.Ny(), Nz = a.Nz();
    for (int x = 0; x < Nx; ++x)
        for (int y = 0; y < Ny; ++y)
            for (int z = 0; z < Nz; ++z)
                if (bits(a(x, y, z)) != bits(b(x, y, z))) {
                    if (n++ == 0) {
                        std::ostringstream os;
                        os.precision(17);
                        os << "(" << x << "," << y << "," << z << "): single=" << a(x, y, z)
                           << " multi=" << b(x, y, z);
                        first = os.str();
                    }
                }
    return n;
}

/// First owned global X plane of every domain after the first (the cuts),
/// using the solver's partition: Nx / n planes each, remainder to the first.
std::vector<int> cut_planes(int Nx, int domains) {
    std::vector<int> cuts;
    int x = 0;
    for (int g = 0; g < domains - 1; ++g) {
        x += Nx / domains + (g < Nx % domains ? 1 : 0);
        cuts.push_back(x);
    }
    return cuts;
}

std::vector<int> shared_device(int domains) {
    return std::vector<int>(domains, 0);
}

class MultiGPUEquivalence : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

} // namespace

TEST_P(MultiGPUEquivalence, MatchesSingleGpuBitwise) {
    const Case c = GetParam();
    SimulationConfig single_cfg = make_config(c);
    SimulationConfig multi_cfg = single_cfg;
    multi_cfg.gpu.multi_gpu = true;
    multi_cfg.gpu.device_ids = shared_device(c.domains);

    CudaSolver single(single_cfg);
    MultiGPUSolver multi(multi_cfg);

    Grid grid(Dim3{c.Nx, kNy, kNz}, Spacing{kH, kH, kH});
    FieldData phi0(grid, "phi0"), u0(grid, "u0");
    make_ic(phi0, u0);
    single.initialize(phi0, u0);
    multi.initialize(phi0, u0);

    FieldData phi_s(grid, "phi_s"), u_s(grid, "u_s"), phi_m(grid, "phi_m"), u_m(grid, "u_m");
    std::string where;

    // Initial state after BCs (and, for the multi solver, the wrap exchange).
    single.copy_phi_to_host(phi_s);
    multi.copy_phi_to_host(phi_m);
    ASSERT_EQ(count_mismatches(phi_s, phi_m, where), 0u) << "initial phi " << where;
    single.copy_u_to_host(u_s);
    multi.copy_u_to_host(u_m);
    ASSERT_EQ(count_mismatches(u_s, u_m, where), 0u) << "initial u " << where;

    const int steps = 8;
    double last_dphi = 0.0;
    for (int s = 1; s <= steps; ++s) {
        SCOPED_TRACE("step " + std::to_string(s));
        single.step(single_cfg.time.dt);
        multi.step(single_cfg.time.dt);

        single.copy_phi_to_host(phi_s);
        multi.copy_phi_to_host(phi_m);
        ASSERT_EQ(count_mismatches(phi_s, phi_m, where), 0u) << "phi " << where;
        single.copy_u_to_host(u_s);
        multi.copy_u_to_host(u_m);
        ASSERT_EQ(count_mismatches(u_s, u_m, where), 0u) << "u " << where;

        const double dphi_s = single.compute_max_dphi();
        const double dphi_m = multi.compute_max_dphi();
        ASSERT_EQ(bits(dphi_s), bits(dphi_m)) << dphi_s << " vs " << dphi_m;
        const double bmax_s = single.compute_boundary_max_phi();
        const double bmax_m = multi.compute_boundary_max_phi();
        ASSERT_EQ(bits(bmax_s), bits(bmax_m)) << bmax_s << " vs " << bmax_m;
        last_dphi = dphi_s;
    }

    // Non-vacuity: the fields are still evolving and every cut carries an X
    // gradient in both fields, so a wrong or stale halo changes the result.
    EXPECT_GT(last_dphi, 1e-6);
    for (int cut : cut_planes(c.Nx, c.domains)) {
        double gphi = 0.0, gu = 0.0;
        for (int y = 0; y < kNy; ++y)
            for (int z = 0; z < kNz; ++z) {
                gphi = std::max(gphi, std::abs(phi_s(cut, y, z) - phi_s(cut - 1, y, z)));
                gu = std::max(gu, std::abs(u_s(cut, y, z) - u_s(cut - 1, y, z)));
            }
        EXPECT_GT(gphi, 1e-3) << "cut at x=" << cut;
        EXPECT_GT(gu, 1e-4) << "cut at x=" << cut;
    }
}

namespace {

std::vector<Case> all_cases() {
    std::vector<Case> cases;
    const TimeScheme schemes[] = {TimeScheme::Euler, TimeScheme::Heun, TimeScheme::RK4,
                                  TimeScheme::IMEX};
    for (TimeScheme s : schemes) {
        // Nx = 14: two domains of 7, three of 5/5/4 (uneven split).
        for (int domains : {2, 3})
            for (BcLayout bc : {BcLayout::Uniform, BcLayout::PerFace, BcLayout::PeriodicX})
                cases.push_back({s, domains, bc, 14});
        // Nx = 6 over three domains: every domain owns the minimum two
        // planes, so each end domain's single interior-facing owned plane is
        // the one next to its wall (and a periodic wrap source).
        cases.push_back({s, 3, BcLayout::PerFace, 6});
        cases.push_back({s, 3, BcLayout::PeriodicX, 6});
    }
    return cases;
}

std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    const Case& c = info.param;
    return std::string(scheme_name(c.scheme)) + "_" + std::to_string(c.domains) + "dom_" +
           bc_name(c.bc) + "_Nx" + std::to_string(c.Nx);
}

} // namespace

INSTANTIATE_TEST_SUITE_P(Schemes, MultiGPUEquivalence, ::testing::ValuesIn(all_cases()), case_name);

// ── Construction limits ────────────────────────────────────────────────────

class MultiGPUSolverTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
    }
};

TEST_F(MultiGPUSolverTest, RejectsFewerThanTwoDomains) {
    auto cfg = make_config({TimeScheme::Euler, 1, BcLayout::Uniform, 14});
    cfg.gpu.device_ids = {0};
    EXPECT_THROW(MultiGPUSolver solver(cfg), std::runtime_error);
}

TEST_F(MultiGPUSolverTest, RejectsDomainsThinnerThanTwoPlanes) {
    // 5 planes over 3 domains leaves domains with a single owned plane: the
    // first domain would not own the plane next to its X wall, which the
    // wall's BC reads.
    auto cfg = make_config({TimeScheme::Euler, 3, BcLayout::Uniform, 5});
    cfg.gpu.device_ids = shared_device(3);
    EXPECT_THROW(MultiGPUSolver solver(cfg), std::invalid_argument);
}

// ── Engine level: solver selection, adaptive dt and the step timer ────────

class MultiGPUEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("test_multi_gpu_engine");
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

    std::filesystem::path test_dir_;
};

TEST_F(MultiGPUEngineTest, AdaptiveRunMatchesSingleGpuBitwise) {
    const int N = 16;
    const int steps = 30;
    auto base = make_config({TimeScheme::Heun, 3, BcLayout::Uniform, N});
    base.grid.Ny = base.grid.Nz = N;
    base.time.max_steps = steps;
    base.time.adaptive = true;
    base.time.adaptive_tolerance = 0.002;
    base.time.dt_min = 1e-6;
    base.time.dt_max = 0.02;
    // Seed well inside the box: the saturation guard (on by default) runs
    // compute_boundary_max_phi every step and must not end the run early.
    base.initial.seed_radius = 1.2;
    base.output.frequency = steps; // the final step is copied to the host
    base.output.async_io = false;
    base.checkpoint.frequency = 0;

    auto single_cfg = base;
    single_cfg.output.output_dir = test_dir_ / "single";
    single_cfg.checkpoint.checkpoint_dir = test_dir_ / "single_ckpt";
    auto multi_cfg = base;
    multi_cfg.output.output_dir = test_dir_ / "multi";
    multi_cfg.checkpoint.checkpoint_dir = test_dir_ / "multi_ckpt";
    multi_cfg.gpu.multi_gpu = true;
    multi_cfg.gpu.device_ids = shared_device(3);

    SimulationEngine single(single_cfg);
    single.run();
    SimulationEngine multi(multi_cfg);
    multi.run();

    std::string where;
    EXPECT_EQ(count_mismatches(single.phi(), multi.phi(), where), 0u) << "phi " << where;
    EXPECT_EQ(count_mismatches(single.u(), multi.u(), where), 0u) << "u " << where;

    // Non-vacuity: the run reached the last step (the final output wrote
    // output_<steps>) and the seed evolved away from the initial profile.
    for (const auto& dir : {single_cfg.output.output_dir, multi_cfg.output.output_dir}) {
        const auto last = dir / ("output_" + std::to_string(steps) + ".vts");
        EXPECT_TRUE(std::filesystem::exists(last)) << last;
    }

    SimulationConfig ic_cfg = base;
    ic_cfg.time.max_steps = 0;
    ic_cfg.output.output_dir = test_dir_ / "ic";
    ic_cfg.checkpoint.checkpoint_dir = test_dir_ / "ic_ckpt";
    SimulationEngine ic(ic_cfg);
    ic.run();
    double moved = 0.0;
    for (std::size_t i = 0; i < ic.phi().size(); ++i)
        moved = std::max(moved, std::abs(ic.phi().data()[i] - single.phi().data()[i]));
    EXPECT_GT(moved, 1e-3);
}
