#include "core/KernelRanges.hpp"
#include "core/SimulationConfig.hpp"
#include "core/StabilityLimits.hpp"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ac;

class SimulationConfigTest : public ::testing::Test {
protected:
    void SetUp() override { Logger::init(spdlog::level::off); }
};

TEST_F(SimulationConfigTest, DefaultValues) {
    SimulationConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.physics.delta, 0.8);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.05);
    EXPECT_DOUBLE_EQ(cfg.physics.willmore_beta, 0.0);
    EXPECT_DOUBLE_EQ(cfg.physics.W0, 1.0);
    EXPECT_DOUBLE_EQ(cfg.physics.D, 2.0);
    EXPECT_EQ(cfg.grid.Nx, 600);
    EXPECT_EQ(cfg.grid.Ny, 600);
    EXPECT_EQ(cfg.grid.Nz, 600);
    EXPECT_DOUBLE_EQ(cfg.time.dt, 0.01);
    EXPECT_EQ(cfg.time.max_steps, 6000);
}

TEST_F(SimulationConfigTest, DerivedQuantities) {
    PhysicsParams p;
    double expected_lambda = p.W0 * p.a1 / p.d0;
    double expected_tau0 =
        (p.W0 * p.W0 * p.W0 * p.a1 * p.a2) / (p.d0 * p.D) + (p.W0 * p.W0 * p.beta0) / p.d0;

    EXPECT_NEAR(p.lambda(), expected_lambda, 1e-12);
    EXPECT_NEAR(p.tau0(), expected_tau0, 1e-12);
}

TEST_F(SimulationConfigTest, ParseFromJsonString) {
    std::string json = R"({
        "physics": {
            "delta": 0.5,
            "epsilon": 0.05,
            "W0": 2.0,
            "D": 3.0
        },
        "grid": {
            "Nx": 64, "Ny": 64, "Nz": 64,
            "dx": 0.2, "dy": 0.2, "dz": 0.2
        },
        "time": {
            "dt": 0.004,
            "max_steps": 100,
            "scheme": "rk4",
            "adaptive": true
        },
        "stencil": "27pt",
        "output": {
            "frequency": 10,
            "output_dir": "/tmp/ac_test",
            "format": "raw"
        },
        "initial": {
            "seed_radius": 3.0
        },
        "boundary": {
            "phi": { "type": "dirichlet", "value": -1.0 },
            "u":   { "type": "neumann", "flux": 0.0 }
        }
    })";

    auto cfg = SimulationConfig::from_json_string(json);

    EXPECT_DOUBLE_EQ(cfg.physics.delta, 0.5);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.05);
    EXPECT_DOUBLE_EQ(cfg.physics.W0, 2.0);
    EXPECT_DOUBLE_EQ(cfg.physics.D, 3.0);
    EXPECT_EQ(cfg.grid.Nx, 64);
    EXPECT_DOUBLE_EQ(cfg.grid.dx, 0.2);
    EXPECT_DOUBLE_EQ(cfg.time.dt, 0.004);
    EXPECT_EQ(cfg.time.max_steps, 100);
    EXPECT_EQ(cfg.time.scheme, TimeScheme::RK4);
    EXPECT_TRUE(cfg.time.adaptive);
    EXPECT_EQ(cfg.stencil, StencilType::Isotropic27Point);
    EXPECT_EQ(cfg.output.frequency, 10);
    EXPECT_EQ(cfg.output.format, "raw");
    EXPECT_DOUBLE_EQ(cfg.initial.seed_radius, 3.0);
    EXPECT_EQ(cfg.boundary.phi_bc.type, BCType::Dirichlet);
    EXPECT_DOUBLE_EQ(cfg.boundary.phi_bc.value, -1.0);
    EXPECT_EQ(cfg.boundary.u_bc.type, BCType::Neumann);
}

TEST_F(SimulationConfigTest, ValidationPasses) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    EXPECT_NO_THROW(cfg.validate());
}

TEST_F(SimulationConfigTest, ValidationFailsSmallGrid) {
    SimulationConfig cfg;
    cfg.grid.Nx = 2;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, ValidationFailsNegativeSpacing) {
    SimulationConfig cfg;
    cfg.grid.dx = -1.0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST_F(SimulationConfigTest, ValidationFailsNegativeDt) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    cfg.time.dt = -0.01;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// Kernels index cells with 32-bit int: 3*3*238609294 = 2^31 - 2 points is the
// largest grid of this shape that fits, one more Nz plane does not.
TEST_F(SimulationConfigTest, ValidationBoundsTotalPointsTo32BitIndexing) {
    SimulationConfig cfg;
    cfg.grid.Nx = 3;
    cfg.grid.Ny = 3;
    cfg.grid.Nz = 238609294;
    EXPECT_NO_THROW(cfg.validate());
    cfg.grid.Nz = 238609295;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// The bound is computed without overflow: 2^21 points per axis are 2^63 points, which a
// long long product Nx * Ny * Nz wraps to -2^63, and 2^22 x 2^22 x 2^20 are 2^64, which it
// wraps to 0, both below the bound; (2^31 - 1)^3 and 2^31 - 1 points along one axis of a
// 3 x 3 cross-section exceed it too. Every one is rejected by the point-count check, and
// grid_points_fit_int agrees at the exact bound.
TEST_F(SimulationConfigTest, PointCountBoundDoesNotOverflow) {
    const int big = std::numeric_limits<int>::max();
    struct Dims {
        int nx, ny, nz;
    };
    for (const Dims d :
         {Dims{1 << 21, 1 << 21, 1 << 21}, Dims{1 << 22, 1 << 22, 1 << 20}, Dims{big, big, big},
          Dims{3, 3, big}, Dims{big, 3, 3}, Dims{1 << 16, 1 << 16, 1 << 16}}) {
        SimulationConfig cfg;
        cfg.grid.Nx = d.nx;
        cfg.grid.Ny = d.ny;
        cfg.grid.Nz = d.nz;
        try {
            cfg.validate();
            ADD_FAILURE() << "accepted " << d.nx << " x " << d.ny << " x " << d.nz;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("32-bit cell indices"), std::string::npos)
                << e.what();
        }
        EXPECT_FALSE(grid_points_fit_int(cfg.grid));
    }
    EXPECT_TRUE(grid_points_fit_int(GridParams{3, 3, 238609294, 0.4, 0.4, 0.4}));
    EXPECT_FALSE(grid_points_fit_int(GridParams{3, 3, 238609295, 0.4, 0.4, 0.4}));
    EXPECT_FALSE(grid_points_fit_int(GridParams{0, 3, 3, 0.4, 0.4, 0.4}));
    EXPECT_FALSE(grid_points_fit_int(GridParams{-3, 3, 3, 0.4, 0.4, 0.4}));
}

TEST_F(SimulationConfigTest, ValidationRejectsEmptyRestartFile) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 10;
    cfg.checkpoint.restart_file = std::filesystem::path{};
    EXPECT_THROW(cfg.validate(), std::invalid_argument);
    cfg.checkpoint.restart_file = "checkpoints/checkpoint_10.acbin";
    EXPECT_NO_THROW(cfg.validate());
}

TEST_F(SimulationConfigTest, MakeGrid) {
    SimulationConfig cfg;
    cfg.grid.Nx = 50;
    cfg.grid.Ny = 60;
    cfg.grid.Nz = 70;
    cfg.grid.dx = 0.3;
    cfg.grid.dy = 0.4;
    cfg.grid.dz = 0.5;
    auto grid = cfg.make_grid();
    EXPECT_EQ(grid.Nx(), 50);
    EXPECT_EQ(grid.Ny(), 60);
    EXPECT_EQ(grid.Nz(), 70);
    EXPECT_DOUBLE_EQ(grid.dx(), 0.3);
}

TEST_F(SimulationConfigTest, ParseTimeSchemes) {
    auto test_scheme = [](const std::string& scheme_str, TimeScheme expected) {
        std::string json = R"({"time": {"scheme": ")" + scheme_str + R"("}})";
        auto cfg = SimulationConfig::from_json_string(json);
        EXPECT_EQ(cfg.time.scheme, expected) << "Failed for scheme: " << scheme_str;
    };

    test_scheme("euler", TimeScheme::Euler);
    test_scheme("heun", TimeScheme::Heun);
    test_scheme("rk4", TimeScheme::RK4);
    test_scheme("imex", TimeScheme::IMEX);
}

TEST_F(SimulationConfigTest, ParseBCTypes) {
    auto test_bc = [](const std::string& bc_str, BCType expected) {
        std::string json = R"({"boundary": {"phi": {"type": ")" + bc_str + R"("}}})";
        auto cfg = SimulationConfig::from_json_string(json);
        EXPECT_EQ(cfg.boundary.phi_bc.type, expected) << "Failed for BC: " << bc_str;
    };

    test_bc("dirichlet", BCType::Dirichlet);
    test_bc("neumann", BCType::Neumann);
    test_bc("periodic", BCType::Periodic);
    test_bc("robin", BCType::Robin);
}

TEST_F(SimulationConfigTest, ParseGPUConfig) {
    std::string json = R"({
        "gpu": {
            "device_ids": [0, 1],
            "multi_gpu": true
        }
    })";
    auto cfg = SimulationConfig::from_json_string(json);
    EXPECT_EQ(cfg.gpu.device_ids.size(), 2u);
    EXPECT_EQ(cfg.gpu.device_ids[0], 0);
    EXPECT_EQ(cfg.gpu.device_ids[1], 1);
    EXPECT_TRUE(cfg.gpu.multi_gpu);
}

// ── Strict parsing: nothing in a config file is silently ignored ────────────

namespace {

/// from_json_string must throw invalid_argument whose message contains every
/// string in `needles`.
void expect_rejected(const std::string& json, std::initializer_list<std::string> needles) {
    try {
        (void)SimulationConfig::from_json_string(json);
        ADD_FAILURE() << "accepted: " << json;
    } catch (const std::invalid_argument& e) {
        for (const auto& n : needles)
            EXPECT_NE(std::string(e.what()).find(n), std::string::npos)
                << "message \"" << e.what() << "\" lacks \"" << n << "\"";
    }
}

/// v to 17 significant digits, enough to tell any two doubles apart.
std::string digits(double v) {
    std::ostringstream s;
    s << std::setprecision(17) << v;
    return s.str();
}

std::string all_faces(const std::string& extra = "") {
    std::string s = "{";
    for (const char* f : {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"})
        s += std::string(s.size() > 1 ? "," : "") + "\"" + f + "\": {\"type\": \"neumann\"}";
    return s + extra + "}";
}

} // namespace

// Every integer field takes only a JSON integer in the range of int. The
// parser used get<int>(), which read 3.9 as 3, 4294967300 as 4, -4294967288
// as 8 and true as 1 without a word, and failed on a string with a JSON type
// error instead of a config error.
TEST_F(SimulationConfigTest, IntegerFieldsTakeOnlyIntegersInTheRangeOfInt) {
    const std::string range = " must be an integer from -2147483648 to 2147483647, got ";
    struct Field {
        const char* section;
        const char* key;
    };
    for (const Field f :
         {Field{"grid", "Nx"}, Field{"grid", "Ny"}, Field{"grid", "Nz"}, Field{"time", "max_steps"},
          Field{"time", "saturation_check_freq"}, Field{"output", "frequency"},
          Field{"checkpoint", "frequency"}, Field{"checkpoint", "keep_last"}}) {
        const std::string name = std::string(f.section) + "." + f.key;
        for (const char* bad : {"3.9", "8.0", "1e2", "4294967300", "-4294967288", "2147483648",
                                "-2147483649", "true", "\"8\"", "null", "[8]"})
            expect_rejected(R"({")" + std::string(f.section) + R"(": {")" + f.key + R"(": )" + bad +
                                "}}",
                            {"config: " + name + range});
    }
    // The ends of the range parse (validate() then has its own rules).
    EXPECT_EQ(
        SimulationConfig::from_json_string(R"({"time": {"max_steps": 2147483647}})").time.max_steps,
        2147483647);
    expect_rejected(R"({"checkpoint": {"frequency": -2147483648}})",
                    {"checkpoint frequency must be >= 0"});

    for (const char* bad : {"3.9", "4294967296", "-4294967296", "true", "\"0\"", "null"})
        expect_rejected(std::string(R"({"gpu": {"device_ids": [0, )") + bad + "]}}",
                        {"config: gpu.device_ids[1]" + range});
    for (const char* bad : {"0", "\"0\"", "{}", "null"})
        expect_rejected(std::string(R"({"gpu": {"device_ids": )") + bad + "}}",
                        {"config: gpu.device_ids must be an array of integers"});
}

// saturation_check_freq and the GPU ids have their ranges checked: the engine
// silently used max(1, saturation_check_freq), and a negative id failed only
// when the solver selected the device.
TEST_F(SimulationConfigTest, SaturationIntervalAndDeviceIdsMustBeInRange) {
    for (const char* freq : {"0", "-5"})
        expect_rejected(std::string(R"({"time": {"saturation_check_freq": )") + freq + "}}",
                        {"saturation_check_freq must be >= 1"});
    EXPECT_EQ(SimulationConfig::from_json_string(R"({"time": {"saturation_check_freq": 1}})")
                  .time.saturation_check_freq,
              1);
    expect_rejected(R"({"gpu": {"device_ids": [0, -1]}})", {"gpu.device_ids must be >= 0, got -1"});
}

// The engine creates output_dir and checkpoint_dir (create_directories), which
// fails for an empty path: --validate-config printed "valid" and the run
// aborted after the solver had allocated the fields.
TEST_F(SimulationConfigTest, EmptyDirectoriesAreRejected) {
    expect_rejected(R"({"output": {"output_dir": ""}})", {"output.output_dir must not be empty"});
    expect_rejected(R"({"checkpoint": {"checkpoint_dir": ""}})",
                    {"checkpoint.checkpoint_dir must not be empty"});
    const auto cfg = SimulationConfig::from_json_string(
        R"({"output": {"output_dir": "."}, "checkpoint": {"checkpoint_dir": "."}})");
    EXPECT_EQ(cfg.output.output_dir, std::filesystem::path("."));
    EXPECT_EQ(cfg.checkpoint.checkpoint_dir, std::filesystem::path("."));
}

// Without multi_gpu the engine runs the whole grid on device_ids[0]
// (SimulationEngine::create_solver): a list of several ids, as the README
// described multi-GPU runs, used one GPU and ignored the others without a
// word. Several ids need multi_gpu; one id runs alone with either value.
TEST_F(SimulationConfigTest, SeveralDevicesRequireMultiGpu) {
    expect_rejected(R"({"gpu": {"device_ids": [0, 1]}})",
                    {"gpu.device_ids lists 2 devices but gpu.multi_gpu is false, so the run would "
                     "use device 0 only"});
    expect_rejected(R"({"gpu": {"device_ids": [3, 0, 0], "multi_gpu": false}})",
                    {"lists 3 devices", "use device 3 only"});
    for (const char* multi : {"true", "false"}) {
        const auto cfg = SimulationConfig::from_json_string(
            std::string(R"({"gpu": {"device_ids": [2], "multi_gpu": )") + multi + "}}");
        EXPECT_EQ(cfg.gpu.device_ids, std::vector<int>{2});
    }
}

// MultiGPUSolver gives the domains of the ids Nx / domains X planes each (one
// more to the first Nx % domains) and needs kMinOwnedPlanes = 2 in every
// domain; --validate-config accepted fewer and the run aborted at startup. The
// last domain owns the fewest: 2 domains - 1 planes leave it one, 2 domains two.
TEST_F(SimulationConfigTest, MultiGpuSplitGivesEveryDomainTwoPlanes) {
    for (int domains = 2; domains <= 5; ++domains) {
        std::string ids;
        for (int g = 0; g < domains; ++g)
            ids += std::string(g ? ", " : "") + "0";
        auto config = [&ids](int Nx) {
            return R"({"grid": {"Nx": )" + std::to_string(Nx) +
                   R"(, "Ny": 8, "Nz": 8}, "gpu": {"device_ids": [)" + ids +
                   R"(], "multi_gpu": true}})";
        };
        expect_rejected(config(2 * domains - 1),
                        {"grid.Nx = " + std::to_string(2 * domains - 1) + " is too small for " +
                         std::to_string(domains) +
                         " domains (gpu.device_ids): the last would own 1 X plane(s), and each "
                         "must own at least 2"});
        EXPECT_EQ(SimulationConfig::from_json_string(config(2 * domains)).grid.Nx, 2 * domains);
    }
}

TEST_F(SimulationConfigTest, UnknownKeyIsRejectedInEverySection) {
    expect_rejected(R"({"physiks": {}})", {"physiks", "<root>"});
    for (const char* section :
         {"physics", "grid", "time", "output", "checkpoint", "gpu", "initial", "boundary"})
        expect_rejected(std::string("{\"") + section + "\": {\"bogus\": 1}}",
                        {"bogus", std::string("\"") + section + "\""});
    expect_rejected(R"({"boundary": {"phi": {"type": "neumann", "vale": 1.0}}})",
                    {"vale", "boundary.phi"});
    expect_rejected(R"({"physics": {"epsilom": 0.05}})", {"epsilom", "physics"});
}

TEST_F(SimulationConfigTest, RemovedBlockSizeIsRejected) {
    // gpu.block_size was parsed but never used (every kernel launches 256
    // threads); accepting it would advertise a setting that does nothing.
    expect_rejected(R"({"gpu": {"block_size": 512}})", {"block_size", "gpu"});
}

TEST_F(SimulationConfigTest, SectionsMustBeObjects) {
    expect_rejected(R"({"physics": 3})", {"physics", "JSON object"});
    expect_rejected(R"({"boundary": {"phi": [1, 2]}})", {"boundary.phi", "JSON object"});
}

TEST_F(SimulationConfigTest, KeysStartingWithUnderscoreAreComments) {
    auto cfg = SimulationConfig::from_json_string(
        R"({"_comment": "x", "physics": {"_note": "y", "epsilon": 0.03},
            "boundary": {"phi": {"_why": "z", "type": "neumann"}}})");
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.03);
    EXPECT_EQ(cfg.boundary.phi_bc.type, BCType::Neumann);
}

// json::parse keeps only the last value of a repeated key, so an earlier
// setting, or a whole earlier section, was dropped without a word:
// {"time": {"scheme": "rk4"}, "time": {"dt": 0.005}} ran Euler. Every object
// of a file, comments included, must name each key once; the message names
// the object as the unknown-key messages name sections, also after nested
// objects and inside arrays. The same key in different objects is no repeat.
TEST_F(SimulationConfigTest, RepeatedKeysAreRejected) {
    auto repeated = [](const std::string& key, const std::string& where) {
        return "config: key \"" + key + "\" appears twice in \"" + where + "\"";
    };
    expect_rejected(R"({"time": {"scheme": "rk4"}, "time": {"dt": 0.005}})",
                    {repeated("time", "<root>")});
    expect_rejected(R"({"time": {"dt": 0.001, "dt": 0.01}})", {repeated("dt", "time")});
    expect_rejected(R"({"physics": {"epsilon": 0.05, "D": 2.0, "epsilon": 0.02}})",
                    {repeated("epsilon", "physics")});
    expect_rejected(R"({"boundary": {"phi": {"type": "neumann", "type": "dirichlet"}}})",
                    {repeated("type", "boundary.phi")});
    expect_rejected(std::string(R"({"boundary": {"u": )") +
                        all_faces(R"(, "x_lo": {"type": "dirichlet"})") + "}}",
                    {repeated("x_lo", "boundary.u")});
    const std::string five_faces = R"("x_lo": {"type": "neumann"}, "x_hi": {"type": "neumann"},
        "y_lo": {"type": "neumann"}, "y_hi": {"type": "neumann"}, "z_lo": {"type": "neumann"})";
    expect_rejected(R"({"boundary": {"phi": {)" + five_faces +
                        R"(, "z_hi": {"type": "neumann", "flux": 0.1, "flux": 0.0}}}})",
                    {repeated("flux", "boundary.phi.z_hi")});
    expect_rejected(R"({"_note": "a", "_note": "b"})", {repeated("_note", "<root>")});
    expect_rejected(R"({"gpu": {"device_ids": [0, {"id": 1, "id": 2}]}})",
                    {repeated("id", "gpu.device_ids[1]")});
    expect_rejected(R"({"boundary": {"phi": {"type": "neumann"}, "u": {"type": "neumann"}},
                        "output": {"frequency": 5}, "output": {}})",
                    {repeated("output", "<root>")});

    const auto cfg = SimulationConfig::from_json_string(
        R"({"output": {"frequency": 7}, "checkpoint": {"frequency": 11},
            "boundary": {"phi": {"type": "neumann"}, "u": {"type": "neumann", "flux": 0.0}}})");
    EXPECT_EQ(cfg.output.frequency, 7);
    EXPECT_EQ(cfg.checkpoint.frequency, 11);
    EXPECT_EQ(cfg.boundary.phi_bc.type, BCType::Neumann);
    EXPECT_EQ(cfg.boundary.u_bc.type, BCType::Neumann);
}

// A face reads only the parameters of its type (dirichlet when "type" is
// omitted): {"flux": 0.0} without a type ran Dirichlet -1 walls, and
// {"type": "neumann", "value": -0.9} ignored the value. Every parameter of
// another type, on a uniform BC and on one face of a per-face BC, of either
// field, must be rejected; each type's own parameters parse.
TEST_F(SimulationConfigTest, ParametersOfAnotherBoundaryTypeAreRejected) {
    struct Type {
        const char* name;
        std::vector<std::string> takes;
    };
    const Type types[] = {{"dirichlet", {"value"}},
                          {"neumann", {"flux"}},
                          {"periodic", {}},
                          {"robin", {"alpha", "beta", "gamma"}}};
    for (const Type& t : types) {
        std::string takes;
        for (const auto& k : t.takes)
            takes += (takes.empty() ? "\"" : ", \"") + k + "\"";
        if (takes.empty())
            takes = "no parameters";
        for (const std::string key : {"value", "flux", "alpha", "beta", "gamma"}) {
            if (std::find(t.takes.begin(), t.takes.end(), key) != t.takes.end())
                continue;
            for (const std::string field : {"phi", "u"}) {
                const std::string bc =
                    std::string(R"({"type": ")") + t.name + R"(", ")" + key + R"(": 0.5})";
                expect_rejected(R"({"boundary": {")" + field + "\": " + bc + "}}",
                                {"config: \"" + key + "\" in \"boundary." + field +
                                 "\" is not a parameter of a " + t.name +
                                 " boundary, which takes " + takes});
                // On z_hi of a per-face BC; z_lo is periodic with a periodic z_hi, so
                // that the parameter is the configuration's only defect.
                const std::string z_lo = std::string(t.name) == "periodic"
                                             ? R"({"type": "periodic"})"
                                             : R"({"type": "neumann"})";
                std::string faces = "{";
                for (const std::string face : {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"})
                    faces += std::string(faces.size() > 1 ? ", " : "") + "\"" + face + "\": " +
                             (face == "z_hi"   ? bc
                              : face == "z_lo" ? z_lo
                                               : std::string(R"({"type": "neumann"})"));
                faces += "}";
                expect_rejected(R"({"boundary": {")" + field + "\": " + faces + "}}",
                                {"config: \"" + key + "\" in \"boundary." + field +
                                 ".z_hi\" is not a parameter of a " + t.name + " boundary"});
            }
        }
    }
    for (const std::string key : {"flux", "alpha", "beta", "gamma"})
        expect_rejected(R"({"boundary": {"phi": {")" + key + R"(": 0.0}}})",
                        {"config: \"" + key +
                         "\" in \"boundary.phi\" is not a parameter of a dirichlet boundary (the "
                         "type when \"type\" is omitted), which takes \"value\""});

    const auto cfg = SimulationConfig::from_json_string(
        R"({"boundary": {"phi": {"value": -0.3},
                         "u": {"type": "robin", "alpha": 2.0, "beta": 0.5, "gamma": -0.4}}})");
    EXPECT_EQ(cfg.boundary.phi_bc.type, BCType::Dirichlet);
    EXPECT_DOUBLE_EQ(cfg.boundary.phi_bc.value, -0.3);
    EXPECT_EQ(cfg.boundary.u_bc.type, BCType::Robin);
    EXPECT_DOUBLE_EQ(cfg.boundary.u_bc.alpha, 2.0);
    EXPECT_DOUBLE_EQ(cfg.boundary.u_bc.beta, 0.5);
    EXPECT_DOUBLE_EQ(cfg.boundary.u_bc.gamma, -0.4);
}

// A per-face object without x_lo used to be read as a uniform BC: the face
// keys were ignored and every face silently kept the default Dirichlet -1.
TEST_F(SimulationConfigTest, PerFaceWithoutXLoIsNotSilentlyUniform) {
    expect_rejected(R"({"boundary": {"phi": {"y_lo": {"type": "periodic"},
                                              "y_hi": {"type": "periodic"}}}})",
                    {"boundary.phi", "missing", "x_lo", "x_hi", "z_lo", "z_hi"});
}

TEST_F(SimulationConfigTest, PerFaceMustNameAllSixFacesAndNothingElse) {
    expect_rejected(R"({"boundary": {"u": {"x_lo": {"type": "neumann"}}}})",
                    {"boundary.u", "missing", "x_hi"});
    expect_rejected(std::string(R"({"boundary": {"phi": )") + all_faces(R"(, "type": "robin")") +
                        "}}",
                    {"\"type\"", "boundary.phi"});
    expect_rejected(std::string(R"({"boundary": {"phi": )") +
                        all_faces(R"(, "x_low": {"type": "neumann"})") + "}}",
                    {"x_low"});
}

TEST_F(SimulationConfigTest, PerFaceParsesEveryFace) {
    const std::string json = R"({"boundary": {
        "phi": {"x_lo": {"type": "neumann", "flux": 0.0}, "x_hi": {"type": "dirichlet", "value": -0.9},
                "y_lo": {"type": "periodic"}, "y_hi": {"type": "periodic"},
                "z_lo": {"type": "robin", "alpha": 1.0, "beta": 0.5, "gamma": 0.2},
                "z_hi": {"type": "neumann"}},
        "u": {"type": "dirichlet", "value": -0.7}}})";
    auto cfg = SimulationConfig::from_json_string(json);
    ASSERT_TRUE(cfg.boundary.per_face);
    const auto& f = cfg.boundary.phi_faces.faces;
    EXPECT_EQ(f[0].type, BCType::Neumann);
    EXPECT_DOUBLE_EQ(f[0].flux, 0.0);
    EXPECT_EQ(f[1].type, BCType::Dirichlet);
    EXPECT_DOUBLE_EQ(f[1].value, -0.9);
    EXPECT_EQ(f[2].type, BCType::Periodic);
    EXPECT_EQ(f[3].type, BCType::Periodic);
    EXPECT_EQ(f[4].type, BCType::Robin);
    EXPECT_DOUBLE_EQ(f[4].beta, 0.5);
    EXPECT_DOUBLE_EQ(f[4].gamma, 0.2);
    EXPECT_EQ(f[5].type, BCType::Neumann);
    // The uniform u applies to all of u's faces in per-face mode.
    for (const auto& face : cfg.boundary.u_faces.faces) {
        EXPECT_EQ(face.type, BCType::Dirichlet);
        EXPECT_DOUBLE_EQ(face.value, -0.7);
    }
    // A face's flux is parsed (phi's must be 0: PhaseFieldBoundaryDataStaysInItsRange).
    const auto u_flux = SimulationConfig::from_json_string(R"({"boundary": {"u": {
        "x_lo": {"type": "neumann", "flux": 0.5}, "x_hi": {"type": "neumann"},
        "y_lo": {"type": "neumann"}, "y_hi": {"type": "neumann"},
        "z_lo": {"type": "neumann"}, "z_hi": {"type": "neumann"}}}})");
    EXPECT_EQ(u_flux.boundary.u_faces.faces[0].type, BCType::Neumann);
    EXPECT_DOUBLE_EQ(u_flux.boundary.u_faces.faces[0].flux, 0.5);
}

// A periodic BC identifies the two faces of an axis. With only one of them
// periodic, the ghost plane on that side copies the far interior while the
// other side does not: the discrete operators become non-symmetric and the
// boundary stops conserving the field. Every face of either field, periodic
// alone, must be rejected; the same axis periodic on both faces is valid.
TEST_F(SimulationConfigTest, PeriodicFaceRequiresPeriodicOppositeFace) {
    static constexpr const char* face_names[] = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"};
    auto faces_with_periodic = [](std::initializer_list<int> periodic) {
        std::string s = "{";
        for (int i = 0; i < 6; ++i) {
            const bool p = std::find(periodic.begin(), periodic.end(), i) != periodic.end();
            s += std::string(i ? "," : "") + "\"" + face_names[i] + "\": {\"type\": \"" +
                 (p ? "periodic" : "neumann") + "\"}";
        }
        return s + "}";
    };
    for (const std::string field : {"phi", "u"}) {
        for (int i = 0; i < 6; ++i) {
            const int opposite = i ^ 1;
            expect_rejected(R"({"boundary": {")" + field + "\": " + faces_with_periodic({i}) + "}}",
                            {"boundary." + field, std::string(face_names[i]) + " is periodic",
                             std::string(face_names[opposite]) + " is not"});
        }
        for (int axis = 0; axis < 3; ++axis) {
            const auto cfg = SimulationConfig::from_json_string(
                R"({"boundary": {")" + field +
                "\": " + faces_with_periodic({2 * axis, 2 * axis + 1}) + "}}");
            const auto& faces = field == "phi" ? cfg.boundary.phi_faces : cfg.boundary.u_faces;
            EXPECT_EQ(faces.get(axis, 0).type, BCType::Periodic);
            EXPECT_EQ(faces.get(axis, 1).type, BCType::Periodic);
        }
    }
}

TEST_F(SimulationConfigTest, BoundaryAcceptsOnlyPhiAndU) {
    expect_rejected(R"({"boundary": {"psi": {"type": "neumann"}}})", {"psi", "boundary"});
}

TEST_F(SimulationConfigTest, ParsesWillmoreBeta) {
    auto cfg = SimulationConfig::from_json_string(
        R"({"physics": {"epsilon": 0.12, "willmore_beta": 1.5}, "time": {"scheme": "euler"}})");
    EXPECT_DOUBLE_EQ(cfg.physics.willmore_beta, 1.5);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.12);
}

// Each Willmore rule rejects on its own, with its own message: a negative
// willmore_beta (a NaN one is not a finite number) at an epsilon where the epsilon rule cannot
// fire, epsilon > 1/15 without willmore_beta, and willmore_beta > 0 with a fully explicit scheme.
TEST_F(SimulationConfigTest, EachWillmoreRuleRejectsOnItsOwn) {
    expect_rejected(R"({"physics": {"epsilon": 0.05, "willmore_beta": -0.5}})",
                    {"willmore_beta must be >= 0"});
    SimulationConfig nan_beta;
    nan_beta.physics.epsilon = 0.05;
    nan_beta.physics.willmore_beta = std::numeric_limits<double>::quiet_NaN();
    try {
        nan_beta.validate();
        ADD_FAILURE() << "NaN willmore_beta accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("physics.willmore_beta = nan is not a finite number"),
                  std::string::npos)
            << e.what();
    }
    expect_rejected(R"({"physics": {"epsilon": 0.12}})", {"exceeds 1/15", "willmore_beta"});
    for (const char* scheme : {"heun", "rk4"})
        expect_rejected(std::string(R"({"physics": {"epsilon": 0.05, "willmore_beta": 1.0}, )") +
                            R"("time": {"scheme": ")" + scheme + R"("}})",
                        {"requires time.scheme"});
}

// Each Robin rule rejects on its own, for phi and u, uniform and on every face:
// alpha = beta = 0, alpha and beta of opposite signs (ghost factor outside
// [0, 1], docs/THEORY.md 9.1). Same signs, or one of them zero (also -0.0),
// are accepted.
TEST_F(SimulationConfigTest, RobinCoefficientsMustNotHaveOppositeSigns) {
    static constexpr const char* face_names[] = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"};
    auto robin = [](const std::string& alpha, const std::string& beta,
                    const std::string& gamma = "0.5") {
        return R"({"type": "robin", "alpha": )" + alpha + R"(, "beta": )" + beta +
               R"(, "gamma": )" + gamma + "}";
    };
    auto per_face = [&](int robin_face, const std::string& bc) {
        std::string s = "{";
        for (int i = 0; i < 6; ++i)
            s += std::string(i ? "," : "") + "\"" + face_names[i] +
                 "\": " + (i == robin_face ? bc : R"({"type": "neumann"})");
        return s + "}";
    };
    const std::pair<const char*, const char*> opposite[] = {
        {"-1.0", "1.0"}, {"1.0", "-0.5"}, {"-1.1", "0.4"}, {"-4.0", "1.0"}};
    const std::pair<const char*, const char*> same[] = {
        {"1.0", "0.5"}, {"-1.0", "-0.5"}, {"0.0", "1.0"}, {"2.0", "0.0"}, {"-0.0", "1.0"}};
    for (const std::string field : {"phi", "u"}) {
        auto with = [&](const std::string& bc) {
            return R"({"boundary": {")" + field + "\": " + bc + "}}";
        };
        expect_rejected(with(robin("0.0", "0.0")),
                        {"Robin BC for " + field + ":", "cannot both be zero"});
        for (const auto& [a, b] : opposite) {
            expect_rejected(with(robin(a, b)), {"Robin BC for " + field + ":", "opposite signs"});
            for (int i = 0; i < 6; ++i)
                expect_rejected(with(per_face(i, robin(a, b))),
                                {"Robin BC for " + field + " face " + std::to_string(i) + ":",
                                 "opposite signs"});
        }
        // gamma 0.5 with alpha 0 is a flux 0.5 / beta, which takes phi outside its range
        // (PhaseFieldBoundaryDataStaysInItsRange); phi's faces here have no flux.
        const std::string gamma = field == "phi" ? "0.0" : "0.5";
        for (const auto& [a, b] : same) {
            EXPECT_NO_THROW((void)SimulationConfig::from_json_string(with(robin(a, b, gamma))))
                << field << " alpha " << a << " beta " << b;
            for (int i = 0; i < 6; ++i)
                EXPECT_NO_THROW(
                    (void)SimulationConfig::from_json_string(with(per_face(i, robin(a, b, gamma)))))
                    << field << " face " << i << " alpha " << a << " beta " << b;
        }
    }
    for (const char* alpha : {"0.0", "-0.0"})
        expect_rejected(R"({"boundary": {"phi": )" + robin(alpha, "1.0") + "}}",
                        {"Robin BC for phi:", "with alpha = 0 the face prescribes the flux"});
}

// A Robin face is filled with its coefficients scaled by a power of two
// (scaled_robin), so its denominator never vanishes and beta*u/h never
// overflows: extreme but meaningful coefficients are accepted (zero flux with
// beta = 1e308 or with the smallest subnormal beta at h = 4, the Dirichlet
// value 1 as 1e-300 u = 1e-300, and the Dirichlet value 2^1024/1.5 = 1.2e308
// as 1.5 * 2^-10 u = 2^1014, whose gamma a scaling with a denominator above 1
// would overflow), and only a ghost constant gamma/(alpha + beta/h) beyond the
// double range is rejected.
TEST_F(SimulationConfigTest, RobinCoefficientsAtTheEdgesOfTheDoubleRange) {
    auto with = [](const std::string& field, double h, const std::string& alpha,
                   const std::string& beta, const std::string& gamma) {
        return R"({"grid": {"dx": )" + std::to_string(h) + R"(, "dy": )" + std::to_string(h) +
               R"(, "dz": )" + std::to_string(h) + R"(}, "boundary": {")" + field +
               R"(": {"type": "robin", "alpha": )" + alpha + R"(, "beta": )" + beta +
               R"(, "gamma": )" + gamma + "}}}";
    };
    for (const std::string field : {"phi", "u"}) {
        EXPECT_NO_THROW(
            (void)SimulationConfig::from_json_string(with(field, 0.4, "0.0", "1e308", "0.0")));
        EXPECT_NO_THROW(
            (void)SimulationConfig::from_json_string(with(field, 4.0, "0.0", "5e-324", "0.0")));
        EXPECT_NO_THROW(
            (void)SimulationConfig::from_json_string(with(field, 0.4, "1e-300", "0.0", "1e-300")));
        // The Dirichlet value 1.2e308 lies outside phi's range [-1, 1].
        if (field == "u")
            EXPECT_NO_THROW((void)SimulationConfig::from_json_string(
                with(field, 0.4, "0.00146484375", "0.0", "1.7555597020139804e+305")));
        else
            expect_rejected(with(field, 0.4, "0.00146484375", "0.0", "1.7555597020139804e+305"),
                            {"Robin BC for phi:", "the range of the phase field"});
        expect_rejected(with(field, 0.4, "1e-200", "0.0", "1e200"),
                        {"Robin BC for " + field + ":", "gamma/(alpha + beta/h) overflows"});
        expect_rejected(with(field, 4.0, "0.0", "5e-324", "0.5"),
                        {"Robin BC for " + field + ":", "gamma/(alpha + beta/h) overflows"});
    }
}

// scaled_robin over alpha and beta from zero and the smallest subnormal to the
// largest double, both signs, three gammas and four spacings: the larger
// scaled term lies in (1/8, 1/2) and the denominator in (1/8, 1) in
// magnitude, and the ghost (gamma + beta*u/h)/(alpha + beta/h) of the kernel
// has the same bits as with the unscaled coefficients whenever every
// intermediate of both is a normal double or zero.
TEST_F(SimulationConfigTest, ScaledRobinCoefficients) {
    const double magnitudes[] = {0.0, 5e-324, 1e-300, 0.00146484375,
                                 0.7, 3.0,    1e300,  std::numeric_limits<double>::max()};
    const double spacings[] = {0.4, 4.0, 1.0 / 3.0, 1e-3};
    auto normal_or_zero = [](double v) { return v == 0.0 || std::fpclassify(v) == FP_NORMAL; };
    struct Ghost {
        double ghost;
        bool normal;
    };
    // The ghost as apply_bc_face_kernel computes it, and whether every
    // intermediate is a normal double or zero.
    auto ghost = [&](double alpha, double beta, double gamma, double h, double u) {
        const double bu = beta * u, buh = bu / h, bh = beta / h, num = gamma + buh,
                     den = alpha + bh, q = num / den;
        bool normal = true;
        for (double v : {alpha, beta, gamma, bu, buh, bh, num, den, q})
            normal = normal && normal_or_zero(v);
        return Ghost{q, normal};
    };
    int compared = 0;
    for (double sign : {1.0, -1.0})
        for (double a : magnitudes)
            for (double b : magnitudes)
                for (double gamma : {0.0, 0.5, -2.0})
                    for (double h : spacings) {
                        if (a == 0.0 && b == 0.0)
                            continue;
                        BoundaryConfig bc;
                        bc.type = BCType::Robin;
                        bc.alpha = sign * a;
                        bc.beta = sign * b;
                        bc.gamma = gamma;
                        SCOPED_TRACE("alpha " + digits(bc.alpha) + " beta " + digits(bc.beta) +
                                     " gamma " + digits(gamma) + " h " + digits(h));
                        const RobinCoefficients r = scaled_robin(bc, h);
                        const double larger = std::max(std::fabs(r.alpha), std::fabs(r.beta / h));
                        EXPECT_GT(larger, 0.125);
                        EXPECT_LT(larger, 0.5);
                        const double den = std::fabs(r.alpha + r.beta / h);
                        EXPECT_GT(den, 0.125);
                        EXPECT_LT(den, 1.0);
                        for (double u : {-0.8, 0.3, 1.0}) {
                            const Ghost unscaled = ghost(bc.alpha, bc.beta, gamma, h, u);
                            const Ghost scaled = ghost(r.alpha, r.beta, r.gamma, h, u);
                            if (!unscaled.normal || !scaled.normal)
                                continue;
                            EXPECT_EQ(scaled.ghost, unscaled.ghost) << "u " << u;
                            ++compared;
                        }
                    }
    // How many of the 2 * 63 * 3 * 4 * 3 = 4536 ghosts stay in the normal range.
    std::printf("ScaledRobinCoefficients: %d ghosts compared bit for bit\n", compared);

    // The preconditions are checked, not assumed.
    BoundaryConfig bc;
    bc.type = BCType::Robin;
    bc.alpha = 0.0;
    bc.beta = 0.0;
    EXPECT_THROW((void)scaled_robin(bc, 0.4), std::invalid_argument);
    bc.alpha = 1.0;
    for (double h : {0.0, -0.4, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW((void)scaled_robin(bc, h), std::invalid_argument) << h;
    for (double bad :
         {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        BoundaryConfig a = bc, b = bc;
        a.alpha = bad;
        b.beta = bad;
        EXPECT_THROW((void)scaled_robin(a, 0.4), std::invalid_argument) << bad;
        EXPECT_THROW((void)scaled_robin(b, 0.4), std::invalid_argument) << bad;
    }
}

namespace {

/// validate() must throw invalid_argument whose message contains every needle.
void expect_invalid(const SimulationConfig& cfg, std::initializer_list<std::string> needles) {
    try {
        cfg.validate();
        ADD_FAILURE() << "accepted";
    } catch (const std::invalid_argument& e) {
        for (const auto& n : needles)
            EXPECT_NE(std::string(e.what()).find(n), std::string::npos)
                << "message \"" << e.what() << "\" lacks \"" << n << "\"";
    }
}

/// Default physics with W0 = d0 = D = a1 = 1 and beta0 = 0, so that tau0 = a2 and
/// lambda = 1 (docs/THEORY.md 5).
SimulationConfig unit_constants() {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 8;
    cfg.physics.W0 = cfg.physics.d0 = cfg.physics.D = cfg.physics.a1 = 1.0;
    cfg.physics.beta0 = 0.0;
    cfg.physics.a2 = 1.0;
    cfg.time.dt = 1e-3;
    return cfg;
}

} // namespace

// Every real parameter must be a finite number. JSON cannot hold NaN or an
// infinity, but a configuration built in code can, and a NaN passes the
// comparisons that should reject it: a NaN D or d0, epsilon, dt, a spacing or
// a seed radius was accepted (and an infinite dt where no stability limit
// applies).
TEST_F(SimulationConfigTest, RealParametersMustBeFinite) {
    const std::pair<const char*, double* (*)(SimulationConfig&)> fields[] = {
        {"physics.delta", [](SimulationConfig& c) { return &c.physics.delta; }},
        {"physics.epsilon", [](SimulationConfig& c) { return &c.physics.epsilon; }},
        {"physics.W0", [](SimulationConfig& c) { return &c.physics.W0; }},
        {"physics.beta0", [](SimulationConfig& c) { return &c.physics.beta0; }},
        {"physics.D", [](SimulationConfig& c) { return &c.physics.D; }},
        {"physics.d0", [](SimulationConfig& c) { return &c.physics.d0; }},
        {"physics.a1", [](SimulationConfig& c) { return &c.physics.a1; }},
        {"physics.a2", [](SimulationConfig& c) { return &c.physics.a2; }},
        {"physics.willmore_beta", [](SimulationConfig& c) { return &c.physics.willmore_beta; }},
        {"grid.dx", [](SimulationConfig& c) { return &c.grid.dx; }},
        {"grid.dy", [](SimulationConfig& c) { return &c.grid.dy; }},
        {"grid.dz", [](SimulationConfig& c) { return &c.grid.dz; }},
        {"time.dt", [](SimulationConfig& c) { return &c.time.dt; }},
        {"time.dt_min", [](SimulationConfig& c) { return &c.time.dt_min; }},
        {"time.dt_max", [](SimulationConfig& c) { return &c.time.dt_max; }},
        {"time.adaptive_tolerance", [](SimulationConfig& c) { return &c.time.adaptive_tolerance; }},
        {"time.cfl_safety", [](SimulationConfig& c) { return &c.time.cfl_safety; }},
        {"time.saturation_threshold",
         [](SimulationConfig& c) { return &c.time.saturation_threshold; }},
        {"initial.seed_radius", [](SimulationConfig& c) { return &c.initial.seed_radius; }},
    };
    const double bad[] = {std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity()};
    for (const auto& [name, of] : fields) {
        for (const double v : bad) {
            SimulationConfig cfg;
            cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 8;
            *of(cfg) = v;
            std::ostringstream text;
            text << std::setprecision(17) << v;
            SCOPED_TRACE(std::string(name) + " = " + text.str());
            expect_invalid(cfg,
                           {std::string(name) + " = " + text.str() + " is not a finite number"});
        }
    }
}

// The derived constants as the kernels receive them, tau0 = W0^3 a1 a2/(d0 D)
// + W0^2 beta0/d0 and lambda = W0 a1/d0, must be finite and positive: d0 or D
// subnormal made tau0 infinite (the first step then divides by it), and a2 = 0,
// a2 < 0 or a negative beta0 removed or reversed the relaxation where the
// Willmore term left no phase-field limit to catch it (review3 #36).
TEST_F(SimulationConfigTest, RelaxationTimeAndCouplingMustBeFiniteAndPositive) {
    const std::string tau0 = "the relaxation time tau0 = W0^3 a1 a2 / (d0 D) + W0^2 beta0 / d0 is ";
    expect_rejected(R"({"physics": {"D": 5e-324, "a1": 1e300}})", {tau0 + "inf"});
    expect_rejected(R"({"physics": {"d0": 5e-324}})", {tau0 + "inf"});
    expect_rejected(R"({"physics": {"epsilon": 0.12, "willmore_beta": 0.6, "a2": 0.0}})",
                    {tau0 + "0 ", "finite and positive"});
    expect_rejected(R"({"physics": {"epsilon": 0.12, "willmore_beta": 0.6, "beta0": -1.0}})",
                    {tau0 + "-1.4", "finite and positive"});
    expect_rejected(R"({"physics": {"a2": -0.5}})", {tau0 + "-0.4", "finite and positive"});
    const std::string lambda = "the coupling constant lambda = W0 a1 / d0 is ";
    expect_rejected(R"({"physics": {"a1": -0.8, "a2": -0.64}})",
                    {lambda + "-1.6", "finite and positive"});
    expect_rejected(R"({"physics": {"a1": 1e308, "a2": 1e-308}})", {lambda + "inf"});
    // A positive subnormal coupling is a value, not a rounding to 0: lambda =
    // 0.01 / 1e308 = 1e-310 with tau0 = 0.01 1e300 / (1e308 1e-10) = 1.
    auto tiny = unit_constants();
    tiny.physics.a1 = 0.01;
    tiny.physics.d0 = 1e308;
    tiny.physics.a2 = 1e300;
    tiny.physics.D = 1e-10;
    ASSERT_EQ(std::fpclassify(tiny.physics.lambda()), FP_SUBNORMAL);
    EXPECT_NO_THROW(tiny.validate());
}

// physics.epsilon must lie in [0, 1/3) (README, docs/THEORY.md 3): compute_An divides by
// 1 - 3 eps, which at the double 1/3 = (2^54 - 1) / (3 2^54) is 0 when 3 eps is rounded on
// its own and 2^-54 when fused, and relaxation_time_range and the liquid-mode analysis
// assume the range. The test this replaces set epsilon 0.5 without willmore_beta, which the
// 1/15 rule and the liquid-mode precondition reject as well, and expected only a throw: it
// passed with this rule deleted (review3 #80). No rule that validate() checks before this
// one rejects these inputs. Each is given with willmore_beta 0 and 1: with 1 the 1/15 rule
// does not apply, and with either the message must be this rule's, which none of the later
// rules' messages contains. The edges are exact: 0 is accepted, the negative subnormal next
// to 0 and the double 1/3 are rejected, and the double below 1/3 passes this rule (without
// willmore_beta the 1/15 rule then rejects it).
TEST_F(SimulationConfigTest, EpsilonMustLieInZeroToOneThird) {
    const std::string rule = "epsilon must be in [0, 1/3)";
    auto with = [](double epsilon, double willmore_beta) {
        SimulationConfig cfg;
        cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 8;
        cfg.physics.epsilon = epsilon;
        cfg.physics.willmore_beta = willmore_beta;
        return cfg;
    };
    for (const double beta : {0.0, 1.0}) {
        for (const double epsilon :
             {-std::numeric_limits<double>::denorm_min(), -0.01, 1.0 / 3.0, 0.5}) {
            SCOPED_TRACE("epsilon " + digits(epsilon) + ", willmore_beta " + digits(beta));
            expect_invalid(with(epsilon, beta), {rule});
        }
    }
    expect_rejected(R"({"physics": {"epsilon": 0.4, "willmore_beta": 1.0}})", {rule});
    EXPECT_NO_THROW(with(0.0, 0.0).validate());
    expect_invalid(with(std::nextafter(1.0 / 3.0, 0.0), 0.0), {"exceeds 1/15"});
}

// W0, D and d0 must be positive (README, physics.*): no test gave one of them a value this
// rule rejects, so each rule could be deleted with every test passing (review3 #80). A zero
// also makes tau0 = W0^3 a1 a2 / (d0 D) + W0^2 beta0 / d0 zero or not finite, which the
// derived-constant rule after these rejects as well. The second input of each keeps tau0
// and lambda = W0 a1 / d0 finite and positive: a negative W0 or d0 with a1 negated, which
// leaves both exactly as they are, and a negative D with beta0 = 2 under IMEX, which has no
// explicit thermal limit. Every later check passes the negative d0 and the negative D, so
// only this rule rejects them; the negative W0 also fails the precondition W0 > 0 of the
// liquid-mode analysis (check_inputs, LiquidModes.cpp), whose message lacks this rule's.
TEST_F(SimulationConfigTest, InterfaceWidthDiffusivityAndCapillaryLengthMustBePositive) {
    SimulationConfig base;
    base.grid.Nx = base.grid.Ny = base.grid.Nz = 8;
    ASSERT_NO_THROW(base.validate());

    SimulationConfig width = base;
    width.physics.W0 = 0.0;
    expect_invalid(width, {"W0 must be positive"});
    width.physics.W0 = -1.0;
    width.physics.a1 = -base.physics.a1;
    ASSERT_EQ(width.physics.tau0(), base.physics.tau0());
    ASSERT_EQ(width.physics.lambda(), base.physics.lambda());
    expect_invalid(width, {"W0 must be positive"});

    SimulationConfig diffusivity = base;
    diffusivity.physics.D = 0.0;
    expect_invalid(diffusivity, {"D must be positive"});
    diffusivity.time.scheme = TimeScheme::IMEX;
    diffusivity.physics.beta0 = 2.0;
    diffusivity.physics.D = 2.0;
    EXPECT_NO_THROW(diffusivity.validate());
    // Only the sign of D differs: tau0 = -a1 a2 + 4 = 3.45, no thermal limit under IMEX,
    // and time.dt within the phase-field limit.
    diffusivity.physics.D = -2.0;
    ASSERT_TRUE(std::isfinite(diffusivity.physics.tau0()) && diffusivity.physics.tau0() > 0.0)
        << diffusivity.physics.tau0();
    ASSERT_EQ(thermal_dt_limit(diffusivity), std::numeric_limits<double>::infinity());
    ASSERT_LE(diffusivity.time.dt, phase_field_dt_limit(diffusivity));
    expect_invalid(diffusivity, {"D must be positive"});

    SimulationConfig capillary = base;
    capillary.physics.d0 = 0.0;
    expect_invalid(capillary, {"d0 must be positive"});
    capillary.physics.d0 = -base.physics.d0;
    capillary.physics.a1 = -base.physics.a1;
    ASSERT_EQ(capillary.physics.tau0(), base.physics.tau0());
    ASSERT_EQ(capillary.physics.lambda(), base.physics.lambda());
    expect_invalid(capillary, {"d0 must be positive"});
}

// time.max_steps >= 1, dt_min > 0, dt_max > 0 and dt_min < dt_max: the engine clamps every
// adaptive step, and the dt an adaptive run restarts with, by std::clamp(dt, dt_min,
// dt_max), whose precondition is dt_min <= dt_max (SimulationEngine.cu). No test broke one
// of these rules, so each could be deleted with every test passing (review3 #80). Each
// input breaks only its rule, except dt_max <= 0: with dt_min > 0 it is not above dt_min
// either, so the order rule after it rejects it as well, and the message must be this
// rule's. The adaptive input keeps dt_min below the stability cap, so no later rule stands
// in. The edges are exact: one step, and dt_min one ulp below dt_max.
TEST_F(SimulationConfigTest, StepCountAndStepBoundsAreChecked) {
    ASSERT_NO_THROW((void)SimulationConfig::from_json_string("{}"));
    for (const char* steps : {"0", "-1"})
        expect_rejected(std::string(R"({"time": {"max_steps": )") + steps + "}}",
                        {"max_steps must be >= 1"});
    EXPECT_EQ(SimulationConfig::from_json_string(R"({"time": {"max_steps": 1}})").time.max_steps,
              1);
    for (const char* dt_min : {"0.0", "-1e-06"})
        expect_rejected(std::string(R"({"time": {"dt_min": )") + dt_min + "}}",
                        {"dt_min must be positive"});
    for (const char* dt_max : {"0.0", "-0.1"})
        expect_rejected(std::string(R"({"time": {"dt_max": )") + dt_max + "}}",
                        {"dt_max must be positive"});

    SimulationConfig order;
    order.grid.Nx = order.grid.Ny = order.grid.Nz = 8;
    order.time.dt_min = order.time.dt_max;
    expect_invalid(order, {"dt_min must be < dt_max"});
    order.time.dt_min = std::nextafter(order.time.dt_max, 0.0);
    EXPECT_NO_THROW(order.validate());
    // An adaptive run whose dt_min = 0.005 exceeds dt_max = 0.001, both below the cap.
    order.time.adaptive = true;
    order.time.dt = 0.001;
    order.time.dt_min = 0.005;
    order.time.dt_max = 0.001;
    ASSERT_LT(order.time.dt_min, order.time.cfl_safety * std::min(thermal_dt_limit(order),
                                                                  phase_field_dt_limit(order)));
    expect_invalid(order, {"dt_min must be < dt_max"});
}

// output.frequency >= 1: the engine writes a frame where step % output.frequency == 0
// (SimulationEngine.cu), an integer division by zero at 0. adaptive_tolerance > 0 with
// adaptive steps: the next dt is the last one times 0.9 adaptive_tolerance / max|dphi|
// within [0.5, 1.5], so a tolerance <= 0 halves dt at every step down to dt_min. At least
// one device id: validate() itself divides grid.Nx by the number of ids, and the solver
// rejects an empty list only when the engine creates it, after the host fields are
// allocated (CudaSolver::activate_device). No test broke one of these rules (review3 #80).
// One frame per step and an adaptive run with the default tolerance are accepted.
TEST_F(SimulationConfigTest, OutputIntervalAdaptiveToleranceAndDeviceListAreChecked) {
    for (const char* frequency : {"0", "-1"})
        expect_rejected(std::string(R"({"output": {"frequency": )") + frequency + "}}",
                        {"output frequency must be >= 1"});
    EXPECT_EQ(
        SimulationConfig::from_json_string(R"({"output": {"frequency": 1}})").output.frequency, 1);
    const std::string adaptive = R"({"time": {"adaptive": true, "dt": 0.001)";
    ASSERT_NO_THROW((void)SimulationConfig::from_json_string(adaptive + "}}"));
    for (const char* tolerance : {"0.0", "-0.01"})
        expect_rejected(adaptive + R"(, "adaptive_tolerance": )" + tolerance + "}}",
                        {"adaptive_tolerance must be positive when adaptive stepping is enabled"});
    for (const char* multi : {"false", "true"})
        expect_rejected(std::string(R"({"gpu": {"device_ids": [], "multi_gpu": )") + multi + "}}",
                        {"At least one GPU device required"});
}

// A cold start sets u = -delta: lambda delta must be a finite double, or the
// coupling lambda u (1 - phi^2)^2 is infinite or NaN in every interior cell at
// the first step (records 85 and 347 of the fuzz campaign). The edge is exact:
// the largest delta with a finite product is accepted, the next one rejected,
// and a restart, which takes u from the checkpoint, is not affected.
TEST_F(SimulationConfigTest, ColdStartTemperatureTimesCouplingMustBeFinite) {
    for (const char* delta : {"1.7976931348623157e308", "-1.7976931348623157e308"})
        expect_rejected(std::string(R"({"physics": {"delta": )") + delta + "}}",
                        {"physics.delta = ", "lambda delta is not a finite double"});
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 8;
    const double lambda = cfg.physics.lambda();
    double edge = std::numeric_limits<double>::max() / lambda;
    while (!std::isfinite(lambda * edge))
        edge = std::nextafter(edge, 0.0);
    while (std::isfinite(lambda * std::nextafter(edge, 2 * edge)))
        edge = std::nextafter(edge, 2 * edge);
    cfg.physics.delta = edge;
    EXPECT_NO_THROW(cfg.validate());
    cfg.physics.delta = std::nextafter(edge, 2 * edge);
    expect_invalid(cfg, {"lambda delta is not a finite double"});
    cfg.checkpoint.restart_file = std::filesystem::path("checkpoints/checkpoint_100.acbin");
    EXPECT_NO_THROW(cfg.validate());
}

// The phase field's boundary data must keep phi in its range [-1, 1]: a
// Dirichlet |value| <= 1, a Neumann flux of 0 and a Robin |gamma| <= |alpha|,
// the exact condition for the face's fill to map [-1, 1] into itself. Ten
// configurations of the fuzz campaign with Robin phi faces far outside the
// range were accepted and failed at run time (tau NaN where the gradient
// overflows A(n), latent heat beyond the Jacobi rounding bound, or a
// divergence). The same data stay valid for u.
TEST_F(SimulationConfigTest, PhaseFieldBoundaryDataStaysInItsRange) {
    const double above_one = std::nextafter(1.0, 2.0);
    auto num = [](double v) {
        std::ostringstream os;
        os << std::setprecision(17) << v;
        return os.str();
    };
    auto dirichlet = [&](double v) { return R"({"type": "dirichlet", "value": )" + num(v) + "}"; };
    auto neumann = [&](double q) { return R"({"type": "neumann", "flux": )" + num(q) + "}"; };
    auto robin = [&](double a, double b, double g) {
        return R"({"type": "robin", "alpha": )" + num(a) + R"(, "beta": )" + num(b) +
               R"(, "gamma": )" + num(g) + "}";
    };
    const std::string inside[] = {
        dirichlet(-1.0), dirichlet(0.0),    dirichlet(1.0),           neumann(0.0),
        neumann(-0.0),   robin(1, 0.5, 1),  robin(1, 0.5, -1),        robin(-1, -0.5, 0.9),
        robin(0, 1, 0),  robin(0, 1, -0.0), robin(1e-300, 0, -1e-300)};
    const std::string outside[] = {dirichlet(above_one),
                                   dirichlet(-above_one),
                                   dirichlet(70.0),
                                   neumann(1e-300),
                                   neumann(-0.5),
                                   robin(1, 0.5, above_one),
                                   robin(-1, -0.5, -above_one),
                                   robin(0, 1, 0.5),
                                   robin(0.0149, 0.05, 10),
                                   robin(1e-16, -0.0, 0.001),
                                   robin(1e-300, 0.001, -1e300)};
    auto uniform = [](const std::string& field, const std::string& bc) {
        return R"({"boundary": {")" + field + "\": " + bc + "}}";
    };
    auto on_face = [](const std::string& field, const std::string& bc) {
        std::string s = R"({"boundary": {")" + field + "\": {";
        for (const std::string face : {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"})
            s += (face == "x_lo" ? "" : ", ") + ("\"" + face + "\": ") +
                 (face == "y_hi" ? bc : std::string(R"({"type": "neumann"})"));
        return s + "}}}";
    };
    for (const auto& bc : inside) {
        for (const std::string field : {"phi", "u"}) {
            EXPECT_NO_THROW((void)SimulationConfig::from_json_string(uniform(field, bc))) << bc;
            EXPECT_NO_THROW((void)SimulationConfig::from_json_string(on_face(field, bc))) << bc;
        }
    }
    for (const auto& bc : outside) {
        SCOPED_TRACE(bc);
        expect_rejected(uniform("phi", bc), {"BC for phi:", "the range of the phase field"});
        expect_rejected(on_face("phi", bc), {"BC for phi face 3:", "the range of the phase field"});
        EXPECT_NO_THROW((void)SimulationConfig::from_json_string(uniform("u", bc)));
        EXPECT_NO_THROW((void)SimulationConfig::from_json_string(on_face("u", bc)));
    }
}

// The single-precision Willmore preconditioner divides every cell by
// tau/dt + willmore_beta d_L: willmore_beta d_L must be a finite float, or
// every preconditioned residual is 0 or NaN (record 407, willmore_beta 1e150).
// The edge is exact: the largest willmore_beta with a finite float product is
// accepted and the next double rejected, with the d_L the solver uses.
TEST_F(SimulationConfigTest, WillmoreDiagonalFitsSinglePrecision) {
    SimulationConfig cfg;
    cfg.grid.Nx = cfg.grid.Ny = cfg.grid.Nz = 8;
    cfg.physics.epsilon = 0.12;
    const double h[3] = {cfg.grid.dx, cfg.grid.dy, cfg.grid.dz};
    const double d_L = willmore_interior_diagonal(cfg.physics.W0, h);
    double edge = static_cast<double>(std::numeric_limits<float>::max()) / d_L;
    while (!std::isfinite(static_cast<float>(edge * d_L)))
        edge = std::nextafter(edge, 0.0);
    while (std::isfinite(static_cast<float>(std::nextafter(edge, 2 * edge) * d_L)))
        edge = std::nextafter(edge, 2 * edge);
    cfg.physics.willmore_beta = edge;
    EXPECT_NO_THROW(cfg.validate()) << "willmore_beta " << edge;
    cfg.physics.willmore_beta = std::nextafter(edge, 2 * edge);
    expect_invalid(cfg, {"willmore_beta d_L = ", "beyond single precision"});
    expect_rejected(R"({"physics": {"epsilon": 0.12, "willmore_beta": 1e150}})",
                    {"willmore_beta d_L = ", "beyond single precision"});
}

// Each Willmore step stores tau/dt per cell as a float and adds the float
// willmore_beta d_L: their float sum must be finite at the largest relaxation
// time and the smallest step (time.dt, and time.dt_min with adaptive steps).
// Record 332 (dt = 5e-324, tau/dt beyond the double range) and record 4 (D =
// 1e-300, tau0 = 1.1e300) were accepted. The edge is exact in tau0 at eps = 0,
// where A = 1 and tau = tau0.
TEST_F(SimulationConfigTest, WillmorePreconditionerDiagonalAtTheSmallestStep) {
    auto cfg = unit_constants();
    cfg.physics.epsilon = 0.0;
    cfg.physics.willmore_beta = 0.25;
    cfg.time.dt = 0.0078125;
    const double h[3] = {cfg.grid.dx, cfg.grid.dy, cfg.grid.dz};
    const float beta_dL = static_cast<float>(0.25 * willmore_interior_diagonal(cfg.physics.W0, h));
    auto diagonal = [&](double tau0) {
        return static_cast<float>(tau0 * (1.0 / cfg.time.dt)) + beta_dL;
    };
    double edge = static_cast<double>(std::numeric_limits<float>::max()) * cfg.time.dt;
    while (!std::isfinite(diagonal(edge)))
        edge = std::nextafter(edge, 0.0);
    while (std::isfinite(diagonal(std::nextafter(edge, 2 * edge))))
        edge = std::nextafter(edge, 2 * edge);
    cfg.physics.a2 = edge; // tau0 = a2
    EXPECT_NO_THROW(cfg.validate()) << "tau0 " << edge;
    cfg.physics.a2 = std::nextafter(edge, 2 * edge);
    expect_invalid(cfg, {"at the smallest step dt = 0.0078125 (time.dt)", "float sum"});
    // Adaptive steps go down to dt_min.
    cfg.physics.a2 = edge;
    cfg.time.adaptive = true;
    cfg.time.dt_min = cfg.time.dt / 2;
    expect_invalid(cfg, {"at the smallest step dt = 0.00390625", "time.dt_min"});
    expect_rejected(
        R"({"physics": {"epsilon": 0.12, "willmore_beta": 0.25},
                        "time": {"dt": 5e-324}})",
        {"at the smallest step dt = 4.9406564584124654e-324", "beyond the double range as well"});
    expect_rejected(R"({"physics": {"epsilon": 0.12, "willmore_beta": 0.25, "D": 1e-300},
                        "time": {"dt": 0.000625}})",
                    {"at the smallest step dt = 0.00062500000000000001", "float sum"});
}

// The relaxation time tau = (tau0 A) A must be a positive finite double for
// every orientation as the kernel computes it: tau0 = DBL_MAX (record 476) or
// 0.88 DBL_MAX at eps 0.15 (record 633) overflowed along <100>, where A = 1 + eps,
// and froze those cells or made the Willmore range NaN; tau0 = 5e-324 at
// eps 0.3 rounds to tau = 0 along <111>. The edges are exact in tau0 for the
// range relaxation_time_range gives (test_KernelRanges), and at eps = 0
// (A = 1) tau0 = DBL_MAX is accepted.
TEST_F(SimulationConfigTest, RelaxationTimeOfEveryOrientationIsAPositiveDouble) {
    auto cfg = unit_constants();
    cfg.physics.epsilon = 0.05;
    double edge = std::numeric_limits<double>::max() / 1.1025;
    while (!(relaxation_time_range(edge, 0.05).hi <= std::numeric_limits<double>::max()))
        edge = std::nextafter(edge, 0.0);
    while (relaxation_time_range(std::nextafter(edge, 2 * edge), 0.05).hi <=
           std::numeric_limits<double>::max())
        edge = std::nextafter(edge, 2 * edge);
    cfg.physics.a2 = edge;
    cfg.time.dt = 1e-3;
    EXPECT_NO_THROW(cfg.validate()) << "tau0 " << edge;
    cfg.physics.a2 = std::nextafter(edge, std::numeric_limits<double>::infinity());
    expect_invalid(cfg, {"the relaxation time tau0 A(n)^2 overflows"});
    cfg.physics.a2 = std::numeric_limits<double>::max();
    cfg.physics.epsilon = 0.0;
    EXPECT_NO_THROW(cfg.validate());

    auto low = unit_constants();
    low.physics.epsilon = 0.3;
    low.physics.willmore_beta = 1.0;
    low.physics.a2 = 5e-324;
    expect_invalid(low, {"the relaxation time tau0 A(n)^2 underflows to 0"});
}

// The parameters a BC type uses must be finite. JSON cannot express NaN or an
// overflowing number (the parser rejects 1e400), so this guards configurations
// built in code.
TEST_F(SimulationConfigTest, BoundaryParametersMustBeFinite) {
    const double bad[] = {std::numeric_limits<double>::infinity(),
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<double>::quiet_NaN()};
    auto expect_finite_rule = [](const SimulationConfig& cfg, const std::string& what) {
        try {
            cfg.validate();
            ADD_FAILURE() << "accepted non-finite " << what;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(what + " must be finite"), std::string::npos)
                << e.what();
        }
    };
    for (double v : bad) {
        for (bool phi : {true, false}) {
            for (int face = -1; face < 6; ++face) {
                SimulationConfig base;
                base.grid.Nx = base.grid.Ny = base.grid.Nz = 10;
                auto& bc = face < 0 ? (phi ? base.boundary.phi_bc : base.boundary.u_bc)
                                    : (phi ? base.boundary.phi_faces : base.boundary.u_faces)
                                          .faces[static_cast<std::size_t>(face)];
                const BoundaryConfig ok = bc;
                bc.type = BCType::Dirichlet;
                bc.value = v;
                expect_finite_rule(base, "value");
                bc = ok;
                bc.type = BCType::Neumann;
                bc.flux = v;
                expect_finite_rule(base, "flux");
                for (const char* what : {"alpha", "beta", "gamma"}) {
                    bc = ok;
                    bc.type = BCType::Robin;
                    bc.alpha = 1.0;
                    bc.beta = 0.5;
                    (std::string(what) == "alpha"  ? bc.alpha
                     : std::string(what) == "beta" ? bc.beta
                                                   : bc.gamma) = v;
                    expect_finite_rule(base, what);
                }
                // A parameter the type does not use is not read.
                bc = ok;
                bc.type = BCType::Dirichlet;
                bc.value = -1.0;
                bc.flux = v;
                bc.alpha = v;
                bc.beta = v;
                bc.gamma = v;
                EXPECT_NO_THROW(base.validate());
            }
        }
    }
}
