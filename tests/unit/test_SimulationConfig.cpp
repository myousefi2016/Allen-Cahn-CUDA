#include "core/SimulationConfig.hpp"
#include "logging/Logger.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <initializer_list>
#include <string>

using namespace ac;

class SimulationConfigTest : public ::testing::Test {
protected:
    void SetUp() override { Logger::init(spdlog::level::off); }
};

TEST_F(SimulationConfigTest, DefaultValues) {
    SimulationConfig cfg;
    EXPECT_DOUBLE_EQ(cfg.physics.delta, 0.8);
    EXPECT_DOUBLE_EQ(cfg.physics.epsilon, 0.07);
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
            "dt": 0.005,
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
    EXPECT_DOUBLE_EQ(cfg.time.dt, 0.005);
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

std::string all_faces(const std::string& extra = "") {
    std::string s = "{";
    for (const char* f : {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"})
        s += std::string(s.size() > 1 ? "," : "") + "\"" + f + "\": {\"type\": \"neumann\"}";
    return s + extra + "}";
}

} // namespace

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

TEST_F(SimulationConfigTest, BoundaryAcceptsOnlyPhiAndU) {
    expect_rejected(R"({"boundary": {"psi": {"type": "neumann"}}})", {"psi", "boundary"});
}
