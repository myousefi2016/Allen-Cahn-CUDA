#include "core/SimulationConfig.hpp"
#include "core/Grid.hpp"
#include "core/FieldData.hpp"
#include "cuda/CudaSolver.cuh"
#include "cuda/CudaUtils.cuh"
#include "io/CheckpointIO.hpp"
#include "logging/Logger.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <filesystem>
#include <string>

using namespace ac;
using namespace ac::cuda;

class CheckpointRestartTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::init(spdlog::level::off);

        // Create a unique temporary directory for this test run
        tmp_dir_ = std::filesystem::temp_directory_path() / "ac_test_checkpoint";
        std::filesystem::create_directories(tmp_dir_);
    }

    void TearDown() override {
        // Clean up temporary files
        std::error_code ec;
        std::filesystem::remove_all(tmp_dir_, ec);
    }

    SimulationConfig make_config() {
        SimulationConfig cfg;
        cfg.grid.Nx = 16;
        cfg.grid.Ny = 16;
        cfg.grid.Nz = 16;
        cfg.grid.dx = 0.4;
        cfg.grid.dy = 0.4;
        cfg.grid.dz = 0.4;
        cfg.time.dt = 0.001;
        cfg.time.max_steps = 50;
        cfg.time.scheme = TimeScheme::Euler;
        cfg.physics.delta = 0.8;
        cfg.physics.epsilon = 0.07;
        cfg.physics.W0 = 1.0;
        cfg.physics.D = 2.0;
        cfg.physics.d0 = 0.5;
        cfg.initial.seed_radius = 3.0;
        cfg.stencil = StencilType::Standard7Point;
        cfg.boundary.phi_bc = {BCType::Dirichlet, -1.0, 0.0, 0.0, 0.0, 0.0};
        cfg.boundary.u_bc = {BCType::Dirichlet, -cfg.physics.delta, 0.0, 0.0, 0.0, 0.0};
        cfg.output.frequency = 10000; // Suppress output
        cfg.checkpoint.frequency = 0; // Managed manually in tests
        return cfg;
    }

    void make_sphere_ic(FieldData& phi, FieldData& u, int N, double r0, double delta) {
        double cx = N / 2.0, cy = N / 2.0, cz = N / 2.0;
        for (int x = 0; x < N; ++x)
            for (int y = 0; y < N; ++y)
                for (int z = 0; z < N; ++z) {
                    double r = std::sqrt((x - cx) * (x - cx) +
                                         (y - cy) * (y - cy) +
                                         (z - cz) * (z - cz));
                    phi(x, y, z) = (r < r0) ? 1.0 : -1.0;
                    u(x, y, z) = (r < r0) ? 0.0 : -delta * (1.0 - std::exp(-(r - r0)));
                }
    }

    std::filesystem::path tmp_dir_;
};

/// Run 50 steps continuously, then run 25 + checkpoint + restart + 25.
/// The final states must match to within floating-point round-off.
TEST_F(CheckpointRestartTest, ExactReproducibility)
{
    const int N = 16;
    const int total_steps = 50;
    const int mid_step = 25;
    const double dt = 0.001;

    auto cfg = make_config();
    cfg.validate();

    Grid grid = cfg.make_grid();

    // --- Reference run: 50 continuous steps ---
    FieldData phi0(grid, "phi"), u0(grid, "u");
    make_sphere_ic(phi0, u0, N, cfg.initial.seed_radius, cfg.physics.delta);

    CudaSolver ref_solver(cfg);
    ref_solver.initialize(phi0, u0);
    for (int s = 0; s < total_steps; ++s) {
        ref_solver.step(dt);
    }

    FieldData phi_ref(grid, "phi_ref"), u_ref(grid, "u_ref");
    ref_solver.copy_phi_to_host(phi_ref);
    ref_solver.copy_u_to_host(u_ref);

    // --- First half: run 25 steps, then save checkpoint ---
    CudaSolver first_half(cfg);
    first_half.initialize(phi0, u0);
    for (int s = 0; s < mid_step; ++s) {
        first_half.step(dt);
    }

    FieldData phi_mid(grid, "phi_mid"), u_mid(grid, "u_mid");
    first_half.copy_phi_to_host(phi_mid);
    first_half.copy_u_to_host(u_mid);

    // Write checkpoint at step 25
    auto ckpt_path = tmp_dir_ / "checkpoint_step25.bin";
    double sim_time = mid_step * dt;
    CheckpointIO::write(ckpt_path, mid_step, sim_time, dt, grid, phi_mid, u_mid);

    // --- Second half: restore from checkpoint and run remaining 25 steps ---
    auto restored = CheckpointIO::read(ckpt_path);
    ASSERT_EQ(restored.step, mid_step);
    ASSERT_DOUBLE_EQ(restored.time, sim_time);
    ASSERT_DOUBLE_EQ(restored.dt, dt);
    ASSERT_EQ(restored.grid.Nx(), N);
    ASSERT_EQ(restored.grid.Ny(), N);
    ASSERT_EQ(restored.grid.Nz(), N);

    CudaSolver second_half(cfg);
    second_half.initialize(restored.phi, restored.u);
    for (int s = 0; s < (total_steps - mid_step); ++s) {
        second_half.step(dt);
    }

    FieldData phi_restarted(grid, "phi_restart"), u_restarted(grid, "u_restart");
    second_half.copy_phi_to_host(phi_restarted);
    second_half.copy_u_to_host(u_restarted);

    // --- Compare: continuous vs checkpoint-restart ---
    double max_phi_diff = 0.0;
    double max_u_diff = 0.0;
    for (std::size_t i = 0; i < phi_ref.size(); ++i) {
        double pd = std::abs(phi_ref.data()[i] - phi_restarted.data()[i]);
        double ud = std::abs(u_ref.data()[i] - u_restarted.data()[i]);
        max_phi_diff = std::max(max_phi_diff, pd);
        max_u_diff = std::max(max_u_diff, ud);
    }

    // Checkpoint/restart through host copy should be bit-exact or very close.
    // Allow a small tolerance for any floating-point reordering effects.
    EXPECT_LT(max_phi_diff, 1e-10)
        << "Phi mismatch after checkpoint/restart. Max diff = " << max_phi_diff;
    EXPECT_LT(max_u_diff, 1e-10)
        << "U mismatch after checkpoint/restart. Max diff = " << max_u_diff;

    // Verify the fields are actually non-trivial (simulation did something)
    double phi_range = 0.0;
    for (std::size_t i = 0; i < phi_ref.size(); ++i) {
        phi_range = std::max(phi_range, std::abs(phi_ref.data()[i]));
    }
    EXPECT_GT(phi_range, 0.5) << "Phi field should have meaningful variation";
}

/// Verify that a checkpoint file can be written, validated, and read back
/// with correct metadata and field contents.
TEST_F(CheckpointRestartTest, CheckpointFileValid)
{
    const int N = 16;
    const double dt = 0.001;
    const int step = 10;
    const double sim_time = step * dt;

    auto cfg = make_config();
    cfg.validate();

    Grid grid = cfg.make_grid();
    FieldData phi(grid, "phi"), u(grid, "u");
    make_sphere_ic(phi, u, N, cfg.initial.seed_radius, cfg.physics.delta);

    // Evolve a few steps so the fields are non-trivial
    CudaSolver solver(cfg);
    solver.initialize(phi, u);
    for (int s = 0; s < step; ++s) {
        solver.step(dt);
    }
    solver.copy_phi_to_host(phi);
    solver.copy_u_to_host(u);

    // Write checkpoint
    auto ckpt_path = tmp_dir_ / "checkpoint_validity.bin";
    CheckpointIO::write(ckpt_path, step, sim_time, dt, grid, phi, u);

    // Verify file exists and is valid
    ASSERT_TRUE(std::filesystem::exists(ckpt_path));
    ASSERT_TRUE(CheckpointIO::is_valid_checkpoint(ckpt_path));

    // Verify file has reasonable size: header + 2 * N^3 * sizeof(double)
    auto file_size = std::filesystem::file_size(ckpt_path);
    std::size_t expected_min = sizeof(CheckpointIO::Header) +
                               2 * static_cast<std::size_t>(N * N * N) * sizeof(double);
    EXPECT_GE(file_size, expected_min);

    // Read it back
    auto restored = CheckpointIO::read(ckpt_path);

    // Verify metadata
    EXPECT_EQ(restored.step, step);
    EXPECT_DOUBLE_EQ(restored.time, sim_time);
    EXPECT_DOUBLE_EQ(restored.dt, dt);
    EXPECT_EQ(restored.grid.Nx(), N);
    EXPECT_EQ(restored.grid.Ny(), N);
    EXPECT_EQ(restored.grid.Nz(), N);
    EXPECT_DOUBLE_EQ(restored.grid.dx(), cfg.grid.dx);
    EXPECT_DOUBLE_EQ(restored.grid.dy(), cfg.grid.dy);
    EXPECT_DOUBLE_EQ(restored.grid.dz(), cfg.grid.dz);

    // Verify field data matches exactly (binary round-trip)
    ASSERT_EQ(restored.phi.size(), phi.size());
    ASSERT_EQ(restored.u.size(), u.size());

    for (std::size_t i = 0; i < phi.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored.phi.data()[i], phi.data()[i])
            << "Phi mismatch at index " << i;
    }
    for (std::size_t i = 0; i < u.size(); ++i) {
        EXPECT_DOUBLE_EQ(restored.u.data()[i], u.data()[i])
            << "U mismatch at index " << i;
    }

    // Verify an invalid file is rejected
    auto bad_path = tmp_dir_ / "not_a_checkpoint.bin";
    {
        std::ofstream ofs(bad_path, std::ios::binary);
        ofs << "this is not a checkpoint";
    }
    EXPECT_FALSE(CheckpointIO::is_valid_checkpoint(bad_path));
}
