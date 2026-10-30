#include "common/Gpu.hpp"
#include "common/TempDir.hpp"
#include "core/SimulationConfig.hpp"
#include "core/SimulationEngine.hpp"
#include "core/StabilityLimits.hpp"
#include "io/CheckpointIO.hpp"
#include "logging/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace ac;

class SimulationEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        AC_GPU_TEST_SETUP();
        Logger::init(spdlog::level::off);
        test_dir_ = ac::test::unique_temp_dir("test_sim_engine");
        std::filesystem::create_directories(test_dir_);
    }

    void TearDown() override { std::filesystem::remove_all(test_dir_); }

    SimulationConfig make_config(int N = 8, int max_steps = 10) {
        SimulationConfig cfg;
        cfg.grid.Nx = N;
        cfg.grid.Ny = N;
        cfg.grid.Nz = N;
        cfg.grid.dx = cfg.grid.dy = cfg.grid.dz = 0.4;

        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.05; // <= 1/15: well-posed without regularisation
        cfg.physics.W0 = 1.0;
        cfg.physics.D = 2.0;
        cfg.physics.d0 = 0.5;

        cfg.time.dt = 0.001;
        cfg.time.max_steps = max_steps;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.time.adaptive = false;
        cfg.time.dt_min = 1e-6;
        cfg.time.dt_max = 0.1;

        cfg.stencil = StencilType::Standard7Point;

        cfg.initial.seed_radius = 2.0;

        // Output to temp dir, frequency higher than max_steps to avoid writes
        cfg.output.output_dir = test_dir_ / "out";
        cfg.output.frequency = max_steps + 100;
        cfg.output.async_io = false;

        cfg.checkpoint.checkpoint_dir = test_dir_ / "ckpt";
        cfg.checkpoint.frequency = max_steps + 100;
        cfg.checkpoint.keep_last = 2;

        cfg.boundary.phi_bc.type = BCType::Neumann;
        cfg.boundary.phi_bc.flux = 0.0;
        cfg.boundary.u_bc.type = BCType::Dirichlet;
        cfg.boundary.u_bc.value = -0.8;

        cfg.gpu.device_ids = {0};
        cfg.gpu.multi_gpu = false;

        return cfg;
    }

    std::filesystem::path test_dir_;
};

TEST_F(SimulationEngineTest, Construction) {
    auto cfg = make_config(8, 10);
    EXPECT_NO_THROW({
        SimulationEngine engine(cfg);
        EXPECT_EQ(engine.grid().Nx(), 8);
        EXPECT_EQ(engine.grid().Ny(), 8);
        EXPECT_EQ(engine.grid().Nz(), 8);
    });
}

// The initial condition is the isotropic (A = 1) equilibrium profile
// phi = -tanh((r - r0) / (sqrt(2) W0)) in physical distance r from the domain
// midpoint, with u = -delta. run() with zero steps returns the fields after
// the solver applied the BCs, so interior cells must equal the profile and
// boundary cells must satisfy the configured Neumann/Dirichlet relations.
TEST_F(SimulationEngineTest, InitializeFieldsPattern) {
    const int N = 12;
    auto cfg = make_config(N, 5);
    cfg.initial.seed_radius = 1.5;
    cfg.time.max_steps = 0;
    SimulationEngine engine(cfg);
    engine.run();

    const auto& phi = engine.phi();
    const auto& u = engine.u();
    const double c = 0.5 * (N - 1);
    const double h = cfg.grid.dx;
    const double inv_sqrt2_W0 = 1.0 / (std::sqrt(2.0) * cfg.physics.W0);
    for (int x = 1; x < N - 1; ++x)
        for (int y = 1; y < N - 1; ++y)
            for (int z = 1; z < N - 1; ++z) {
                double rx = (x - c) * h, ry = (y - c) * h, rz = (z - c) * h;
                double r = std::sqrt(rx * rx + ry * ry + rz * rz);
                double expected = -std::tanh((r - cfg.initial.seed_radius) * inv_sqrt2_W0);
                ASSERT_DOUBLE_EQ(phi(x, y, z), expected) << x << "," << y << "," << z;
            }
    // Zero-flux Neumann phi: every X-face cell copies its inner neighbour.
    for (int y = 0; y < N; ++y)
        for (int z = 0; z < N; ++z) {
            ASSERT_DOUBLE_EQ(phi(0, y, z), phi(1, y, z));
            ASSERT_DOUBLE_EQ(phi(N - 1, y, z), phi(N - 2, y, z));
        }
    // u = -delta everywhere, which also satisfies Dirichlet u = -0.8 = -delta.
    for (std::size_t i = 0; i < u.size(); ++i)
        ASSERT_DOUBLE_EQ(u.data()[i], -cfg.physics.delta);
}

TEST_F(SimulationEngineTest, RunShortSimulation) {
    auto cfg = make_config(8, 5);
    SimulationEngine engine(cfg);

    EXPECT_NO_THROW(engine.run());

    // Verify no NaN in output fields
    const auto& phi = engine.phi();
    const auto& u = engine.u();
    int N = 8;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_FALSE(std::isnan(phi(x, y, z)))
                    << "NaN in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isnan(u(x, y, z)))
                    << "NaN in u at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(phi(x, y, z)))
                    << "Inf in phi at (" << x << "," << y << "," << z << ")";
                EXPECT_FALSE(std::isinf(u(x, y, z)))
                    << "Inf in u at (" << x << "," << y << "," << z << ")";
            }

    // phi should remain bounded in [-1, 1] approximately
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                EXPECT_GE(phi(x, y, z), -2.0);
                EXPECT_LE(phi(x, y, z), 2.0);
            }
}

TEST_F(SimulationEngineTest, AdaptiveTimeStep) {
    auto cfg = make_config(8, 5);
    cfg.time.adaptive = true;
    cfg.time.adaptive_tolerance = 0.01;
    cfg.time.dt_min = 1e-6;
    cfg.time.dt_max = 0.05;
    cfg.time.dt = 0.001;

    SimulationEngine engine(cfg);
    EXPECT_NO_THROW(engine.run());

    // Verify the simulation completed without error and fields are valid
    const auto& phi = engine.phi();
    int N = 8;
    bool all_finite = true;
    for (int x = 0; x < N; ++x)
        for (int y = 0; y < N; ++y)
            for (int z = 0; z < N; ++z) {
                if (std::isnan(phi(x, y, z)) || std::isinf(phi(x, y, z))) {
                    all_finite = false;
                }
            }
    EXPECT_TRUE(all_finite) << "Adaptive stepping produced non-finite values";
}

// A seed wider than the box puts phi > saturation_threshold on the walls from
// the start: the guard must stop the run at the first check (step 1) and
// write that step's checkpoint and output, not run on to max_steps.
TEST_F(SimulationEngineTest, SaturationStopsAtFirstCheckWithCheckpointAndOutput) {
    auto cfg = make_config(8, 10);
    cfg.initial.seed_radius = 2.0; // walls are 1.4 from the centre
    cfg.time.exit_on_saturation = true;
    cfg.time.saturation_threshold = -0.5;
    cfg.time.saturation_check_freq = 1;

    SimulationEngine engine(cfg);
    ASSERT_NO_THROW(engine.run());

    EXPECT_TRUE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_1.acbin"));
    EXPECT_TRUE(std::filesystem::exists(cfg.output.output_dir / "output_1.vts"));
    EXPECT_FALSE(std::filesystem::exists(cfg.output.output_dir / "output_2.vts"));
    EXPECT_FALSE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_2.acbin"));
}

// The same with Dirichlet phi = -1 walls (default.json, benchmark_small, the
// k8s configurations): the boundary cells always hold -1, so the guard must
// read the cells next to the walls to see the solid there.
TEST_F(SimulationEngineTest, SaturationStopsWithDirichletWalls) {
    auto cfg = make_config(8, 10);
    cfg.boundary.phi_bc.type = BCType::Dirichlet;
    cfg.boundary.phi_bc.value = -1.0;
    cfg.initial.seed_radius = 2.0; // the cells next to the walls are 1.0 from the centre
    cfg.time.exit_on_saturation = true;
    cfg.time.saturation_threshold = -0.5;
    cfg.time.saturation_check_freq = 1;

    SimulationEngine engine(cfg);
    ASSERT_NO_THROW(engine.run());

    EXPECT_TRUE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_1.acbin"));
    EXPECT_TRUE(std::filesystem::exists(cfg.output.output_dir / "output_1.vts"));
    EXPECT_FALSE(std::filesystem::exists(cfg.output.output_dir / "output_2.vts"));
}

// Control for the test above: a seed well inside the box never trips the
// guard, so the run reaches max_steps.
TEST_F(SimulationEngineTest, NoSaturationRunsToMaxSteps) {
    auto cfg = make_config(16, 10);
    cfg.initial.seed_radius = 1.0; // walls are 3.0 from the centre
    cfg.time.saturation_check_freq = 1;
    cfg.output.frequency = 10;

    SimulationEngine engine(cfg);
    ASSERT_NO_THROW(engine.run());
    EXPECT_TRUE(std::filesystem::exists(cfg.output.output_dir / "output_10.vts"));
}

// SIGINT/SIGTERM set g_shutdown_requested: the engine finishes the current
// step, writes a checkpoint and output for it and stops.
TEST_F(SimulationEngineTest, ShutdownRequestWritesCheckpointAndStops) {
    auto cfg = make_config(8, 50);
    SimulationEngine engine(cfg);
    g_shutdown_requested.store(true);
    EXPECT_NO_THROW(engine.run());
    g_shutdown_requested.store(false);

    EXPECT_TRUE(
        CheckpointIO::is_valid_checkpoint(cfg.checkpoint.checkpoint_dir / "checkpoint_1.acbin"));
    EXPECT_TRUE(std::filesystem::exists(cfg.output.output_dir / "output_1.vts"));
    EXPECT_FALSE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_2.acbin"));
}

// A resume continues after its checkpoint's step and never writes the frames up
// to that step again, so they must be on disk before the checkpoint. Here the
// frame of the checkpoint's step cannot be written (its temporary file is
// /dev/full, as on a full disk) with asynchronous output, at a regular
// checkpoint step and where a shutdown request and the saturation guard end the
// run: each run must stop with the write error and without that checkpoint.
// Before, the checkpoint was committed while the frame was still queued (at a
// shutdown or saturation even before it was queued), and the failure surfaced
// only afterwards. With the frame writable, both are written.
TEST_F(SimulationEngineTest, NoCheckpointBeforeTheFramesUpToItsStep) {
    ASSERT_TRUE(std::filesystem::exists("/dev/full")) << "the test needs /dev/full";
    for (const std::string ending : {"checkpoint step", "shutdown", "saturation"}) {
        for (const bool writable : {false, true}) {
            SCOPED_TRACE(ending + (writable ? ", frame writable" : ", frame fails"));
            std::filesystem::remove_all(test_dir_);
            auto cfg = make_config(8, 4);
            cfg.output.async_io = true;
            cfg.output.format = "raw";
            cfg.time.exit_on_saturation = ending == "saturation";
            cfg.time.saturation_check_freq = 1;
            cfg.initial.seed_radius = ending == "saturation" ? 2.0 : 1.0;
            const int step = ending == "checkpoint step" ? 2 : 1;
            if (ending == "checkpoint step")
                cfg.output.frequency = cfg.checkpoint.frequency = 2;
            std::filesystem::create_directories(cfg.output.output_dir);
            const auto frame =
                cfg.output.output_dir / ("output_" + std::to_string(step) + "_phi.raw");
            if (!writable)
                std::filesystem::create_symlink("/dev/full", frame.string() + ".tmp");
            const auto checkpoint =
                cfg.checkpoint.checkpoint_dir / ("checkpoint_" + std::to_string(step) + ".acbin");

            SimulationEngine engine(cfg);
            g_shutdown_requested.store(ending == "shutdown");
            bool threw = false;
            try {
                engine.run();
            } catch (const std::runtime_error& e) {
                threw = true;
                EXPECT_NE(std::string(e.what()).find("output write failed for step " +
                                                     std::to_string(step)),
                          std::string::npos)
                    << e.what();
            }
            g_shutdown_requested.store(false);
            EXPECT_EQ(threw, !writable);
            EXPECT_EQ(std::filesystem::exists(checkpoint), writable);
            EXPECT_EQ(std::filesystem::exists(frame), writable);
        }
    }
}

// The writer saves the checkpoints on its thread: one that fails there (its
// directory has become a regular file) must still stop the run with that error,
// at the next write or at the end of the run, like a failed frame.
TEST_F(SimulationEngineTest, FailedCheckpointStopsTheRun) {
    auto cfg = make_config(8, 4);
    cfg.output.async_io = true;
    cfg.checkpoint.frequency = 2;
    SimulationEngine engine(cfg);
    std::filesystem::remove_all(cfg.checkpoint.checkpoint_dir);
    std::ofstream(cfg.checkpoint.checkpoint_dir) << "not a directory";
    try {
        engine.run();
        ADD_FAILURE() << "the run finished although its checkpoint failed";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("checkpoint write failed for step 2"),
                  std::string::npos)
            << e.what();
    }
}

// dt*D/h^2 = 12.5 is far beyond explicit-Euler stability: the fields overflow
// to Inf/NaN within a few hundred steps. The run must fail with an error at
// the first checkpoint and must not write a checkpoint of the diverged state
// (the resume helper would pick it as the latest one).
TEST_F(SimulationEngineTest, DivergedRunFailsWithoutWritingItsState) {
    auto cfg = make_config(8, 1000);
    cfg.time.dt = 1.0;
    cfg.time.exit_on_saturation = false;
    cfg.checkpoint.frequency = 500;
    cfg.output.frequency = 500;

    SimulationEngine engine(cfg);
    try {
        engine.run();
        FAIL() << "a diverged run finished without an error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("diverged"), std::string::npos) << e.what();
    }
    EXPECT_FALSE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_500.acbin"));
    EXPECT_FALSE(std::filesystem::exists(cfg.output.output_dir / "output_500.vts"));
}

// Same instability with no output or checkpoint step inside the run: the
// final-state check in run() must still catch it.
TEST_F(SimulationEngineTest, DivergedRunFailsEvenWithoutOutputSteps) {
    auto cfg = make_config(8, 1000);
    cfg.time.dt = 1.0;
    cfg.time.exit_on_saturation = false;

    SimulationEngine engine(cfg);
    EXPECT_THROW(engine.run(), std::runtime_error);
}

// phi()/u() hold the final state after run(), not the initial condition.
TEST_F(SimulationEngineTest, RunLeavesFinalStateOnHost) {
    auto cfg = make_config(12, 0);
    cfg.initial.seed_radius = 1.2;
    SimulationEngine initial(cfg);
    initial.run();

    cfg.time.max_steps = 20;
    cfg.output.output_dir = test_dir_ / "out20";
    cfg.checkpoint.checkpoint_dir = test_dir_ / "ckpt20";
    SimulationEngine engine(cfg);
    engine.run();

    double moved = 0.0;
    for (std::size_t i = 0; i < engine.phi().size(); ++i)
        moved = std::max(moved, std::abs(engine.phi().data()[i] - initial.phi().data()[i]));
    EXPECT_GT(moved, 1e-6);
}

// A configured restart_file that does not exist stops the run before any
// step; the old engine cold-started and its retention rotated real
// checkpoints away.
TEST_F(SimulationEngineTest, MissingRestartFileFailsWithoutTouchingCheckpoints) {
    auto cfg = make_config(8, 20);
    cfg.checkpoint.frequency = 5;
    cfg.checkpoint.keep_last = 1;
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto existing = cfg.checkpoint.checkpoint_dir / "checkpoint_100.acbin";
    CheckpointIO::write(existing, 100, 1.0, 0.001, grid, phi, u);
    cfg.checkpoint.restart_file = cfg.checkpoint.checkpoint_dir / "checkpoint_10O.acbin"; // typo

    SimulationEngine engine(cfg);
    EXPECT_THROW(engine.run(), std::runtime_error);
    EXPECT_TRUE(CheckpointIO::is_valid_checkpoint(existing));
    EXPECT_FALSE(std::filesystem::exists(cfg.checkpoint.checkpoint_dir / "checkpoint_5.acbin"));
}

// A checkpoint with the right dimensions but a different spacing describes
// another physical problem and must be rejected.
TEST_F(SimulationEngineTest, RestoreRejectsDifferentGridSpacing) {
    auto cfg = make_config(8, 20);
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid other(Dim3{8, 8, 8}, Spacing{0.5, 0.4, 0.4});
    FieldData phi(other, "phi"), u(other, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto file = cfg.checkpoint.checkpoint_dir / "foreign.acbin";
    CheckpointIO::write(file, 10, 0.1, 0.001, other, phi, u);
    cfg.checkpoint.restart_file = file;

    SimulationEngine engine(cfg);
    try {
        engine.run();
        FAIL() << "restored a checkpoint with dx = 0.5 into a dx = 0.4 run";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("spacing"), std::string::npos) << e.what();
    }
}

// validate() accepts every max_steps >= 1, INT_MAX included: the loop ends there. A
// restart from step INT_MAX - 2 runs exactly the steps INT_MAX - 1 and INT_MAX (a
// checkpoint after each), a restart from INT_MAX none.
TEST_F(SimulationEngineTest, RunEndsAtTheLargestMaxSteps) {
    const int last = std::numeric_limits<int>::max();
    for (const int from : {last - 2, last}) {
        SCOPED_TRACE("restart from step " + std::to_string(from));
        auto cfg = make_config(8, 10);
        cfg.time.max_steps = last;
        cfg.output.frequency = last;
        cfg.checkpoint.frequency = 1;
        cfg.checkpoint.keep_last = 3;
        cfg.checkpoint.checkpoint_dir = test_dir_ / ("ckpt_" + std::to_string(from));
        cfg.output.output_dir = test_dir_ / ("out_" + std::to_string(from));
        ASSERT_NO_THROW(cfg.validate());
        Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
        FieldData phi(grid, "phi"), u(grid, "u");
        phi.fill(-1.0);
        u.fill(-0.8);
        const auto file = test_dir_ / ("restart_" + std::to_string(from) + ".acbin");
        CheckpointIO::write(file, from, 0.1, 0.001, grid, phi, u);
        cfg.checkpoint.restart_file = file;

        SimulationEngine engine(cfg);
        ASSERT_NO_THROW(engine.run());
        std::vector<std::string> written;
        if (std::filesystem::exists(cfg.checkpoint.checkpoint_dir))
            for (const auto& entry :
                 std::filesystem::directory_iterator(cfg.checkpoint.checkpoint_dir))
                written.push_back(entry.path().filename().string());
        std::sort(written.begin(), written.end());
        if (from == last - 2) {
            EXPECT_EQ(written, (std::vector<std::string>{"checkpoint_2147483646.acbin",
                                                         "checkpoint_2147483647.acbin"}));
            EXPECT_EQ(
                CheckpointIO::read(cfg.checkpoint.checkpoint_dir / "checkpoint_2147483647.acbin")
                    .step,
                last);
        } else {
            EXPECT_TRUE(written.empty());
        }
    }
}

// A fixed-dt run continues with its configured dt, whatever dt the checkpoint
// was written with (a checkpoint of an earlier configuration, or the same run
// after time.dt was changed): the configured dt is the one validate() checked
// and the frequencies count. The next checkpoint records the dt that was used.
TEST_F(SimulationEngineTest, FixedDtRestartUsesTheConfiguredDt) {
    auto cfg = make_config(8, 15);
    cfg.checkpoint.frequency = 5;
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto file = test_dir_ / "earlier.acbin";
    CheckpointIO::write(file, 10, 0.1, 0.01, grid, phi, u);
    cfg.checkpoint.restart_file = file;
    ASSERT_EQ(cfg.time.dt, 0.001);

    SimulationEngine engine(cfg);
    engine.run();
    const auto next = CheckpointIO::read(cfg.checkpoint.checkpoint_dir / "checkpoint_15.acbin");
    double time = 0.1;
    for (int s = 0; s < 5; ++s)
        time += 0.001;
    EXPECT_EQ(next.dt, 0.001);
    EXPECT_EQ(next.time, time);
}

// An adaptive run continues from the dt it had adapted to: the first step
// after the restart uses the checkpoint's dt.
TEST_F(SimulationEngineTest, AdaptiveRestartContinuesFromTheCheckpointDt) {
    auto cfg = make_config(8, 11);
    cfg.time.adaptive = true;
    cfg.checkpoint.frequency = 11;
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto file = test_dir_ / "adapted.acbin";
    CheckpointIO::write(file, 10, 0.1, 0.0005, grid, phi, u);
    cfg.checkpoint.restart_file = file;

    SimulationEngine engine(cfg);
    engine.run();
    const auto next = CheckpointIO::read(cfg.checkpoint.checkpoint_dir / "checkpoint_11.acbin");
    EXPECT_EQ(next.dt, 0.0005);
    EXPECT_EQ(next.time, 0.1 + 0.0005);
}

// Adaptive steps never exceed cfl_safety times the explicit stability limits,
// even when dt_min is larger (validate() rejects such a dt_min; the engine
// must not depend on it): the cap is applied after the dt_min / dt_max clamp.
TEST_F(SimulationEngineTest, AdaptiveStepsNeverExceedTheStabilityCap) {
    auto cfg = make_config(8, 4);
    cfg.time.adaptive = true;
    cfg.time.cfl_safety = 0.9;
    const double cap =
        cfg.time.cfl_safety * std::min(thermal_dt_limit(cfg), phase_field_dt_limit(cfg));
    cfg.time.dt_min = 2.0 * cap;
    cfg.time.dt_max = 4.0 * cap;
    cfg.checkpoint.frequency = 1;
    cfg.checkpoint.keep_last = 4;
    SimulationEngine engine(cfg);
    engine.run();
    for (int step = 2; step <= 4; ++step) {
        const auto data = CheckpointIO::read(cfg.checkpoint.checkpoint_dir /
                                             ("checkpoint_" + std::to_string(step) + ".acbin"));
        EXPECT_EQ(data.dt, cap) << "step " << step;
    }
}

// When phi does not change (here the liquid at rest, u in equilibrium with
// its Dirichlet walls), tolerance / max|dphi| is infinite and dt grows by the
// largest factor, 1.5, per step until the stability cap.
TEST_F(SimulationEngineTest, AdaptiveDtGrowsToTheCapWhenPhiIsAtRest) {
    auto cfg = make_config(8, 18);
    cfg.time.adaptive = true;
    cfg.checkpoint.frequency = 1;
    cfg.checkpoint.keep_last = 8;
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto file = test_dir_ / "rest.acbin";
    CheckpointIO::write(file, 10, 0.1, 0.001, grid, phi, u);
    cfg.checkpoint.restart_file = file;
    const double cap =
        cfg.time.cfl_safety * std::min(thermal_dt_limit(cfg), phase_field_dt_limit(cfg));

    SimulationEngine engine(cfg);
    engine.run();
    double expected = 0.001; // step 11, the first after the restart, keeps it
    for (int step = 11; step <= 18; ++step) {
        if (step > 11)
            expected = std::min(1.5 * expected, cap);
        const auto data = CheckpointIO::read(cfg.checkpoint.checkpoint_dir /
                                             ("checkpoint_" + std::to_string(step) + ".acbin"));
        EXPECT_EQ(data.dt, expected) << "step " << step;
    }
    EXPECT_EQ(expected, cap) << "the run is too short to reach the cap";
}

// The first step after an adaptive restart obeys the same bounds as every
// adapted step: the checkpoint's dt is clamped to [dt_min, dt_max] and then
// capped at cfl_safety times the stability limits (a checkpoint written with
// another configuration's dt, here 1.0, would otherwise run unchecked).
TEST_F(SimulationEngineTest, AdaptiveRestartIsCappedLikeEveryAdaptedStep) {
    auto cfg = make_config(8, 11);
    cfg.time.adaptive = true;
    cfg.checkpoint.frequency = 11;
    std::filesystem::create_directories(cfg.checkpoint.checkpoint_dir);
    Grid grid(Dim3{8, 8, 8}, Spacing{0.4, 0.4, 0.4});
    FieldData phi(grid, "phi"), u(grid, "u");
    phi.fill(-1.0);
    u.fill(-0.8);
    const auto file = test_dir_ / "foreign_dt.acbin";
    CheckpointIO::write(file, 10, 0.1, 1.0, grid, phi, u);
    cfg.checkpoint.restart_file = file;
    const double cap =
        cfg.time.cfl_safety * std::min(thermal_dt_limit(cfg), phase_field_dt_limit(cfg));
    ASSERT_LT(cap, cfg.time.dt_max);

    SimulationEngine engine(cfg);
    engine.run();
    const auto next = CheckpointIO::read(cfg.checkpoint.checkpoint_dir / "checkpoint_11.acbin");
    EXPECT_EQ(next.dt, cap);
    EXPECT_EQ(next.time, 0.1 + cap);
}

TEST_F(SimulationEngineTest, ConfigValidationRejectsInvalid) {
    auto cfg = make_config(8, 10);
    cfg.output.format = "invalid_format";
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    auto cfg2 = make_config(8, 10);
    cfg2.checkpoint.keep_last = 0;
    EXPECT_THROW(cfg2.validate(), std::invalid_argument);

    auto cfg3 = make_config(8, 10);
    cfg3.initial.seed_radius = -1.0;
    EXPECT_THROW(cfg3.validate(), std::invalid_argument);
}
