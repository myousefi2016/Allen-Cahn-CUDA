#include "common/Gpu.hpp"
#include "common/TempDir.hpp"
#include "core/SimulationEngine.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/MultiGPUSolver.cuh"
#include "logging/Logger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <spdlog/sinks/ostream_sink.h>
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
    bool willmore = false; ///< eps = 0.12 regularised with beta = 1 (one CG solve per step)
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
    cfg.physics.epsilon = c.willmore ? 0.12 : 0.05;
    cfg.physics.willmore_beta = c.willmore ? 1.0 : 0.0;
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
    // difference in both fields far above rounding (1e-16), so a wrong or
    // stale halo value changes the computed cells and the bitwise comparison
    // above would catch it.
    EXPECT_GT(last_dphi, 1e-6);
    for (int cut : cut_planes(c.Nx, c.domains)) {
        double gphi = 0.0, gu = 0.0;
        for (int y = 0; y < kNy; ++y)
            for (int z = 0; z < kNz; ++z) {
                gphi = std::max(gphi, std::abs(phi_s(cut, y, z) - phi_s(cut - 1, y, z)));
                gu = std::max(gu, std::abs(u_s(cut, y, z) - u_s(cut - 1, y, z)));
            }
        EXPECT_GT(gphi, 1e-8) << "cut at x=" << cut;
        EXPECT_GT(gu, 1e-8) << "cut at x=" << cut;
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
    // Willmore-regularised strong anisotropy: the CG solve spans the domains
    // (halo exchange of omega, p and lap p; per-plane dot products).
    for (TimeScheme s : {TimeScheme::Euler, TimeScheme::IMEX}) {
        for (int domains : {2, 3})
            for (BcLayout bc : {BcLayout::Uniform, BcLayout::PerFace, BcLayout::PeriodicX})
                cases.push_back({s, domains, bc, 14, true});
        cases.push_back({s, 3, BcLayout::PeriodicX, 6, true});
    }
    return cases;
}

std::string case_label(const Case& c) {
    return std::string(scheme_name(c.scheme)) + "_" + std::to_string(c.domains) + "dom_" +
           bc_name(c.bc) + "_Nx" + std::to_string(c.Nx) + (c.willmore ? "_Willmore" : "");
}

std::string case_name(const ::testing::TestParamInfo<Case>& info) {
    return case_label(info.param);
}

/// gtest prints a parameter with PrintTo; ctest names the case after that text (without
/// it, after a dump of the struct's bytes).
void PrintTo(const Case& c, std::ostream* os) {
    *os << case_label(c);
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
    // 5 planes over 3 domains split 2/2/1 (the remainder goes to the first
    // domains): the last domain would own only plane 4, its X wall's plane, and
    // not plane 3 next to it, which the wall's BC reads.
    auto cfg = make_config({TimeScheme::Euler, 3, BcLayout::Uniform, 5});
    cfg.gpu.device_ids = shared_device(3);
    EXPECT_THROW(MultiGPUSolver solver(cfg), std::invalid_argument);
}

// validate() rejects exactly the splits the constructor rejects, so a
// configuration that --validate-config accepts does not fail at startup
// for its split, and none that the solver can run is rejected.
TEST_F(MultiGPUSolverTest, ValidateRejectsExactlyTheSplitsTheSolverRejects) {
    for (int domains = 2; domains <= 4; ++domains) {
        for (int Nx = 3; Nx <= 3 * domains; ++Nx) {
            auto cfg = make_config({TimeScheme::Euler, domains, BcLayout::Uniform, Nx});
            cfg.gpu.multi_gpu = true;
            cfg.gpu.device_ids = shared_device(domains);
            bool solver_rejects = false;
            try {
                MultiGPUSolver solver(cfg);
            } catch (const std::invalid_argument&) {
                solver_rejects = true;
            }
            bool validate_rejects = false;
            try {
                cfg.validate();
            } catch (const std::invalid_argument&) {
                validate_rejects = true;
            }
            EXPECT_EQ(validate_rejects, solver_rejects) << Nx << " planes over " << domains;
            EXPECT_EQ(solver_rejects, Nx / domains < 2) << Nx << " planes over " << domains;
        }
    }
}

// The saturation guard's reduction reads the interior cells next to the
// walls (layers 1 and N-2 of each axis), not the boundary cells: those hold
// the boundary condition. A Dirichlet face at phi = +1, above every interior
// value (they are drawn from [-1, 1)), must not show; the result is the exact maximum over the
// layers next to the walls, on one GPU and split into two and three domains
// (only the physical X walls count). With zero-flux faces every boundary cell
// is a copy of a cell next to a wall, so there the maximum over the boundary
// cells is the same number.
TEST_F(MultiGPUSolverTest, BoundaryMaxReadsTheCellsNextToTheWalls) {
    auto bc = [](BCType type, double value, double alpha, double beta) {
        BoundaryConfig b;
        b.type = type;
        b.value = value;
        b.alpha = alpha;
        b.beta = beta;
        return b;
    };
    const BoundaryConfig zero_flux = bc(BCType::Neumann, 0.0, 1.0, 0.0);
    for (bool all_zero_flux : {false, true}) {
        SCOPED_TRACE(all_zero_flux ? "zero-flux faces" : "Dirichlet +1, Robin, zero-flux faces");
        SimulationConfig cfg;
        cfg.grid.Nx = 12;
        cfg.grid.Ny = 7;
        cfg.grid.Nz = 9;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.epsilon = 0.05;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.boundary.per_face = true;
        if (all_zero_flux)
            cfg.boundary.phi_faces = PerFaceBoundary::uniform(zero_flux);
        else
            cfg.boundary.phi_faces.faces = {
                bc(BCType::Dirichlet, 1.0, 1.0, 0.0), zero_flux,
                bc(BCType::Robin, 0.0, 1.0, 0.5),     zero_flux,
                bc(BCType::Dirichlet, 1.0, 1.0, 0.0), bc(BCType::Dirichlet, 1.0, 1.0, 0.0)};
        cfg.boundary.u_faces = PerFaceBoundary::uniform(zero_flux);
        cfg.validate();
        const Grid grid = cfg.make_grid();
        FieldData phi(grid, "phi"), u(grid, "u");
        std::uint64_t state = 12345;
        for (std::size_t i = 0; i < phi.size(); ++i) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            phi.data()[i] = -1.0 + 2.0 * static_cast<double>(state >> 11) / 9007199254740992.0;
            u.data()[i] = -0.5;
        }
        const int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
        double expected = -1e30;
        for (int x = 1; x < Nx - 1; ++x)
            for (int y = 1; y < Ny - 1; ++y)
                for (int z = 1; z < Nz - 1; ++z)
                    if (x == 1 || x == Nx - 2 || y == 1 || y == Ny - 2 || z == 1 || z == Nz - 2)
                        expected = std::max(expected, phi(x, y, z));

        CudaSolver single(cfg);
        single.initialize(phi, u);
        EXPECT_EQ(single.compute_boundary_max_phi(), expected);
        for (int domains : {2, 3}) {
            auto multi_cfg = cfg;
            multi_cfg.gpu.multi_gpu = true;
            multi_cfg.gpu.device_ids = shared_device(domains);
            MultiGPUSolver multi(multi_cfg);
            multi.initialize(phi, u);
            EXPECT_EQ(multi.compute_boundary_max_phi(), expected) << domains << " domains";
        }
        if (all_zero_flux) {
            // The boundary cells after the BCs hold copies of the cells next
            // to the walls: their maximum is the same number.
            FieldData applied(grid, "phi");
            single.copy_phi_to_host(applied);
            double boundary_max = -1e30;
            for (int x = 0; x < Nx; ++x)
                for (int y = 0; y < Ny; ++y)
                    for (int z = 0; z < Nz; ++z)
                        if (x == 0 || x == Nx - 1 || y == 0 || y == Ny - 1 || z == 0 || z == Nz - 1)
                            boundary_max = std::max(boundary_max, applied(x, y, z));
            EXPECT_EQ(boundary_max, expected);
        }
    }
}

// Every wall layer counts and nothing else does: phi = -0.9 everywhere, 0.9
// at a cell that is not next to any wall (must never show), and 0.5 at one
// cell of a single wall layer, in turn for each of the six layers and, for
// the Y layer, on the planes where two or three domains split the grid
// (x = 4, 6, 8). The reduction must return 0.5 on one GPU and on two and
// three domains. Two face layouts. With zero-flux faces (the uniform phi_bc
// path) every boundary cell is a copy of a cell next to a wall, so that layout
// cannot tell a wall layer from the boundary plane beyond it. With a
// Dirichlet face on every side, each at its own value in (0.5, 1], every
// boundary cell lies strictly above every cell of the wall layers (-0.9 and
// the 0.5 spike): a reduction that reads boundary cells in place of, or
// besides, the layer next to a wall returns more than 0.5. Off its edges each
// face holds its own value, so a reduction that reads a face's plane there
// returns that face's value, which names the face.
TEST_F(MultiGPUSolverTest, BoundaryMaxSeesEveryWallLayerAndNothingElse) {
    BoundaryConfig zero_flux;
    zero_flux.type = BCType::Neumann;
    // Dirichlet phi of the faces in Face order (XLo, XHi, YLo, YHi, ZLo, ZHi):
    // distinct dyadic values above the spike and at most 1 (validate() keeps a
    // phi Dirichlet value in [-1, 1]), none equal to -0.9, 0.5 or 0.9.
    const std::array<double, 6> wall = {1.0, 0.9375, 0.875, 0.8125, 0.75, 0.6875};
    struct Spike {
        int x, y, z;
    };
    for (bool dirichlet_walls : {false, true}) {
        SCOPED_TRACE(dirichlet_walls
                         ? "Dirichlet faces XLo..ZHi = 1, 0.9375, 0.875, 0.8125, 0.75, 0.6875"
                         : "zero-flux faces");
        SimulationConfig cfg;
        cfg.grid.Nx = 12;
        cfg.grid.Ny = 7;
        cfg.grid.Nz = 9;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;
        cfg.physics.epsilon = 0.05;
        cfg.time.scheme = TimeScheme::Euler;
        if (dirichlet_walls) {
            cfg.boundary.per_face = true;
            for (std::size_t f = 0; f < wall.size(); ++f)
                cfg.boundary.phi_faces.faces[f] = bc_of(BCType::Dirichlet, wall[f]);
            cfg.boundary.u_faces = PerFaceBoundary::uniform(zero_flux);
        } else {
            cfg.boundary.phi_bc = zero_flux;
            cfg.boundary.u_bc = zero_flux;
        }
        cfg.validate();
        const Grid grid = cfg.make_grid();
        const int Nx = cfg.grid.Nx, Ny = cfg.grid.Ny, Nz = cfg.grid.Nz;
        for (const Spike sp : {Spike{1, 3, 4}, Spike{10, 3, 4}, Spike{4, 1, 4}, Spike{6, 1, 4},
                               Spike{8, 1, 4}, Spike{5, 5, 4}, Spike{8, 3, 1}, Spike{4, 3, 7}}) {
            SCOPED_TRACE("spike at " + std::to_string(sp.x) + "," + std::to_string(sp.y) + "," +
                         std::to_string(sp.z));
            FieldData phi(grid, "phi"), u(grid, "u");
            phi.fill(-0.9);
            u.fill(-0.5);
            phi(6, 3, 4) = 0.9;
            phi(sp.x, sp.y, sp.z) = 0.5;
            CudaSolver single(cfg);
            single.initialize(phi, u);
            FieldData applied(grid, "phi");
            single.copy_phi_to_host(applied);
            if (dirichlet_walls) {
                // What this layout relies on: the BC fill leaves every cell
                // inside the walls as it was, and every boundary cell of a
                // face off its edges (the cells a reduction reading that
                // face's plane in place of its wall layer would see) holds
                // that face's value.
                std::size_t changed_inside = 0, not_the_face_value = 0;
                for (int x = 0; x < Nx; ++x)
                    for (int y = 0; y < Ny; ++y)
                        for (int z = 0; z < Nz; ++z) {
                            const bool bx = x == 0 || x == Nx - 1, by = y == 0 || y == Ny - 1,
                                       bz = z == 0 || z == Nz - 1;
                            const int faces_here = (bx ? 1 : 0) + (by ? 1 : 0) + (bz ? 1 : 0);
                            if (faces_here == 0 && applied(x, y, z) != phi(x, y, z))
                                ++changed_inside;
                            if (faces_here != 1)
                                continue;
                            std::size_t f = 0;
                            if (bx)
                                f = x == 0 ? 0u : 1u;
                            else if (by)
                                f = y == 0 ? 2u : 3u;
                            else
                                f = z == 0 ? 4u : 5u;
                            if (applied(x, y, z) != wall[f])
                                ++not_the_face_value;
                        }
                ASSERT_EQ(changed_inside, 0u);
                ASSERT_EQ(not_the_face_value, 0u);
            }
            EXPECT_EQ(single.compute_boundary_max_phi(), 0.5);
            for (int domains : {2, 3}) {
                auto multi_cfg = cfg;
                multi_cfg.gpu.multi_gpu = true;
                multi_cfg.gpu.device_ids = shared_device(domains);
                MultiGPUSolver multi(multi_cfg);
                multi.initialize(phi, u);
                // The domains hold the same cells as one GPU, boundary cells
                // included.
                FieldData gathered(grid, "phi");
                multi.copy_phi_to_host(gathered);
                std::string where;
                ASSERT_EQ(count_mismatches(applied, gathered, where), 0u)
                    << domains << " domains: " << where;
                EXPECT_EQ(multi.compute_boundary_max_phi(), 0.5) << domains << " domains";
            }
        }
    }
}

// ── Engine level: solver selection, adaptive dt and the saturation guard ──

namespace {

/// What the program logs at info level while `body` runs: the default logger is replaced
/// by one that writes into a string.
template <class Body> std::string logged(Body&& body) {
    auto stream = std::make_shared<std::ostringstream>();
    const auto previous = spdlog::default_logger();
    auto logger = std::make_shared<spdlog::logger>(
        "capture", std::make_shared<spdlog::sinks::ostream_sink_mt>(*stream));
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    try {
        body();
    } catch (...) {
        spdlog::set_default_logger(previous);
        throw;
    }
    spdlog::set_default_logger(previous);
    return stream->str();
}

std::size_t occurrences(const std::string& text, const std::string& needle) {
    std::size_t n = 0;
    for (auto at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1))
        ++n;
    return n;
}

/// The steps of the checkpoints in `dir` (checkpoint_<step>.acbin), in increasing order.
std::vector<int> checkpoint_steps(const std::filesystem::path& dir) {
    std::vector<int> steps;
    if (!std::filesystem::is_directory(dir))
        return steps;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        const std::string prefix = "checkpoint_", suffix = ".acbin";
        if (name.size() > prefix.size() + suffix.size() && name.starts_with(prefix) &&
            name.ends_with(suffix))
            steps.push_back(std::stoi(name.substr(prefix.size())));
    }
    std::sort(steps.begin(), steps.end());
    return steps;
}

} // namespace

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

// The engine builds a MultiGPUSolver for a multi_gpu configuration with several devices
// and a CudaSolver otherwise. Both give the same bits here, so which one ran shows only
// in what the engine logs as it creates the solver.
constexpr const char* kMultiGpuSolverLog = "Creating MultiGPUSolver with 3 GPUs";

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
    // Seed well inside the box. Started from the initial condition (no
    // checkpoint restore), the saturation guard (on by default) checks only on
    // steps that are multiples of time.saturation_check_freq (100 by default),
    // so it does not run in these 30 steps.
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

    std::unique_ptr<SimulationEngine> single_engine, multi_engine;
    const std::string single_log = logged([&] {
        single_engine = std::make_unique<SimulationEngine>(single_cfg);
        single_engine->run();
    });
    const std::string multi_log = logged([&] {
        multi_engine = std::make_unique<SimulationEngine>(multi_cfg);
        multi_engine->run();
    });
    const SimulationEngine& single = *single_engine;
    const SimulationEngine& multi = *multi_engine;
    EXPECT_EQ(occurrences(multi_log, kMultiGpuSolverLog), 1u) << multi_log;
    EXPECT_EQ(occurrences(single_log, "MultiGPUSolver"), 0u) << single_log;

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

namespace {

/// The wall whose phi has zero flux (the other five hold phi = -1), and the domain of the
/// three (cut_planes) whose cells next to it rise above the saturation threshold first.
struct SaturatingWall {
    Face face;
    int domain;
    const char* name;
};

/// The ctest name of the case (a dump of the struct's bytes would show the address in
/// `name`).
void PrintTo(const SaturatingWall& w, std::ostream* os) {
    *os << w.name;
}

} // namespace

class MultiGPUEngineSaturation : public MultiGPUEngineTest,
                                 public ::testing::WithParamInterface<SaturatingWall> {};

// The saturation guard, every step, through the multi-GPU solver. The seed (radius 1.5,
// centred in 16^3 cells) puts phi = -0.65 on the cells next to every wall; next to the
// zero-flux wall phi rises above the threshold -0.5 first (at step 28 of 200, the other
// walls' cells at -0.73; exp/round26/mgpu_engine/probe.txt), and only one domain reads
// those cells: x_lo's the first, x_hi's the last, y_hi's (where the seed's arm meets it,
// x = 7..8) the middle one. The single- and the multi-GPU run stop at the same step, with
// the same fields.
TEST_P(MultiGPUEngineSaturation, StopsAtTheStepOfTheSingleGpuRun) {
    const SaturatingWall wall = GetParam();
    const int N = 16;
    const int max_steps = 200;
    auto base = make_config({TimeScheme::Euler, 3, BcLayout::Uniform, N});
    base.grid.Ny = base.grid.Nz = N;
    base.time.max_steps = max_steps;
    base.time.saturation_check_freq = 1;
    base.boundary.per_face = true;
    base.boundary.phi_faces = PerFaceBoundary::uniform(bc_of(BCType::Dirichlet, -1.0));
    base.boundary.phi_faces[wall.face] = bc_of(BCType::Neumann);
    base.boundary.u_faces = PerFaceBoundary::uniform(bc_of(BCType::Dirichlet, -0.55));
    base.initial.seed_radius = 1.5;
    base.output.frequency = max_steps;
    base.output.async_io = false;
    base.checkpoint.frequency = 0; // a checkpoint only where the guard stops the run
    ASSERT_TRUE(base.time.exit_on_saturation);
    ASSERT_NO_THROW(base.validate());

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
    std::unique_ptr<SimulationEngine> multi_engine;
    const std::string multi_log = logged([&] {
        multi_engine = std::make_unique<SimulationEngine>(multi_cfg);
        multi_engine->run();
    });
    EXPECT_EQ(occurrences(multi_log, kMultiGpuSolverLog), 1u) << multi_log;

    const std::vector<int> stops = checkpoint_steps(single_cfg.checkpoint.checkpoint_dir);
    ASSERT_EQ(stops.size(), 1u) << "the single-GPU run did not stop at the guard";
    EXPECT_GT(stops[0], 1);
    EXPECT_LT(stops[0], max_steps);
    EXPECT_EQ(checkpoint_steps(multi_cfg.checkpoint.checkpoint_dir), stops);
    std::string where;
    EXPECT_EQ(count_mismatches(single.phi(), multi_engine->phi(), where), 0u) << "phi " << where;
    EXPECT_EQ(count_mismatches(single.u(), multi_engine->u(), where), 0u) << "u " << where;

    // At the stop step only the cells next to the zero-flux wall are above the threshold,
    // and only those of the domain that reads them.
    const std::vector<int> cuts = cut_planes(N, 3);
    const int x_begin = wall.domain == 0 ? 0 : cuts[wall.domain - 1];
    const int x_end = wall.domain == 2 ? N : cuts[wall.domain];
    const double threshold = base.time.saturation_threshold;
    const FieldData& phi = single.phi();
    for (int f = 0; f < 6; ++f) {
        const int axis = f / 2;
        const int layer = f % 2 == 0 ? 1 : N - 2;
        double max_phi = -2.0;
        int above = 0, outside = 0;
        for (int a = 1; a < N - 1; ++a)
            for (int b = 1; b < N - 1; ++b) {
                const int x = axis == 0 ? layer : a;
                const int y = axis == 1 ? layer : (axis == 0 ? a : b);
                const int z = axis == 2 ? layer : b;
                max_phi = std::max(max_phi, phi(x, y, z));
                if (phi(x, y, z) > threshold) {
                    ++above;
                    outside += x < x_begin || x >= x_end;
                }
            }
        if (static_cast<Face>(f) == wall.face) {
            EXPECT_GT(above, 0) << "face " << f << ": max " << max_phi;
            EXPECT_EQ(outside, 0) << "face " << f << ": cells above the threshold outside ["
                                  << x_begin << ", " << x_end << ")";
        } else {
            EXPECT_EQ(above, 0) << "face " << f << ": max " << max_phi;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Walls, MultiGPUEngineSaturation,
                         ::testing::Values(SaturatingWall{Face::XLo, 0, "XLo"},
                                           SaturatingWall{Face::XHi, 2, "XHi"},
                                           SaturatingWall{Face::YHi, 1, "YHi"}),
                         [](const ::testing::TestParamInfo<SaturatingWall>& info) {
                             return std::string(info.param.name);
                         });
