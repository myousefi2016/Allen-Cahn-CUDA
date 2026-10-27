#include "core/SimulationConfig.hpp"
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

TEST_F(SimulationConfigTest, ValidationFailsInvalidEpsilon) {
    SimulationConfig cfg;
    cfg.grid.Nx = 10;
    cfg.grid.Ny = 10;
    cfg.grid.Nz = 10;
    cfg.physics.epsilon = 0.5; // >= 1/3
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
        "phi": {"x_lo": {"type": "neumann", "flux": 0.5}, "x_hi": {"type": "dirichlet", "value": -0.9},
                "y_lo": {"type": "periodic"}, "y_hi": {"type": "periodic"},
                "z_lo": {"type": "robin", "alpha": 1.0, "beta": 0.5, "gamma": 0.2},
                "z_hi": {"type": "neumann"}},
        "u": {"type": "dirichlet", "value": -0.7}}})";
    auto cfg = SimulationConfig::from_json_string(json);
    ASSERT_TRUE(cfg.boundary.per_face);
    const auto& f = cfg.boundary.phi_faces.faces;
    EXPECT_EQ(f[0].type, BCType::Neumann);
    EXPECT_DOUBLE_EQ(f[0].flux, 0.5);
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

// Each Willmore rule rejects on its own, with its own message: a negative (or
// NaN) willmore_beta at an epsilon where the epsilon rule cannot fire,
// epsilon > 1/15 without willmore_beta, and willmore_beta > 0 with a fully
// explicit scheme.
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
        EXPECT_NE(std::string(e.what()).find("willmore_beta must be >= 0"), std::string::npos)
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
    auto robin = [](const std::string& alpha, const std::string& beta) {
        return R"({"type": "robin", "alpha": )" + alpha + R"(, "beta": )" + beta +
               R"(, "gamma": 0.5})";
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
        for (const auto& [a, b] : same) {
            EXPECT_NO_THROW((void)SimulationConfig::from_json_string(with(robin(a, b))))
                << field << " alpha " << a << " beta " << b;
            for (int i = 0; i < 6; ++i)
                EXPECT_NO_THROW(
                    (void)SimulationConfig::from_json_string(with(per_face(i, robin(a, b)))))
                    << field << " face " << i << " alpha " << a << " beta " << b;
        }
    }
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
        EXPECT_NO_THROW((void)SimulationConfig::from_json_string(
            with(field, 0.4, "0.00146484375", "0.0", "1.7555597020139804e+305")));
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
