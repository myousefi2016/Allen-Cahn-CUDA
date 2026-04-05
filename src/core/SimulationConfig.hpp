#pragma once

#include "core/Grid.hpp"

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ac {

// ── Boundary condition types ───────────────────────────────────────────────
enum class BCType { Dirichlet, Neumann, Periodic, Robin };

/// Per-face boundary condition specification.
struct BoundaryConfig {
    BCType type = BCType::Dirichlet;
    Real value = -1.0;    ///< Dirichlet value
    Real flux = 0.0;      ///< Neumann flux
    Real alpha = 1.0;     ///< Robin: alpha*u + beta*du/dn = gamma
    Real beta = 0.0;
    Real gamma = 0.0;
};

// ── Physics parameters ─────────────────────────────────────────────────────
struct PhysicsParams {
    Real delta = 0.8;          ///< Dimensionless undercooling
    Real epsilon = 0.07;       ///< Anisotropy strength
    Real W0 = 1.0;             ///< Interface width parameter
    Real beta0 = 0.0;          ///< Kinetic coefficient
    Real D = 2.0;              ///< Thermal diffusivity
    Real d0 = 0.5;             ///< Capillary length
    Real a1 = 1.25 / std::sqrt(2.0);
    Real a2 = 0.64;

    [[nodiscard]] Real lambda() const { return W0 * a1 / d0; }
    [[nodiscard]] Real tau0() const {
        return (W0 * W0 * W0 * a1 * a2) / (d0 * D) + (W0 * W0 * beta0) / d0;
    }
};

// ── Grid parameters ────────────────────────────────────────────────────────
struct GridParams {
    int Nx = 600, Ny = 600, Nz = 600;
    Real dx = 0.4, dy = 0.4, dz = 0.4;
};

// ── Time integration ───────────────────────────────────────────────────────
enum class TimeScheme { Euler, Heun, RK4, IMEX };

struct TimeParams {
    Real dt = 0.01;
    Real dt_min = 1e-6;
    Real dt_max = 0.1;
    int max_steps = 6000;
    TimeScheme scheme = TimeScheme::Euler;
    bool adaptive = false;
    Real adaptive_tolerance = 0.01;  ///< Max allowed |dphi| per step
    Real cfl_safety = 0.9;
};

// ── Stencil type ───────────────────────────────────────────────────────────
enum class StencilType { Standard7Point, Isotropic27Point };

// ── Output parameters ──────────────────────────────────────────────────────
struct OutputParams {
    int frequency = 100;
    std::filesystem::path output_dir = "./out";
    std::string format = "vts";  ///< "vts" or "raw"
    bool async_io = true;
};

// ── Checkpoint parameters ──────────────────────────────────────────────────
struct CheckpointParams {
    int frequency = 500;
    std::filesystem::path checkpoint_dir = "./checkpoints";
    int keep_last = 3;  ///< Rolling checkpoint count
    std::optional<std::filesystem::path> restart_file;
};

// ── GPU parameters ─────────────────────────────────────────────────────────
struct GPUParams {
    std::vector<int> device_ids = {0};
    int block_size_1d = 256;
    bool multi_gpu = false;
};

// ── Initial condition ──────────────────────────────────────────────────────
struct InitialCondition {
    Real seed_radius = 5.0;
};

// ── Boundary conditions (per-face: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) ──
struct BoundaryParams {
    BoundaryConfig phi_bc;  ///< Applied to all faces for phi
    BoundaryConfig u_bc;    ///< Applied to all faces for u
};

// ── Top-level configuration ────────────────────────────────────────────────
struct SimulationConfig {
    PhysicsParams physics;
    GridParams grid;
    TimeParams time;
    StencilType stencil = StencilType::Standard7Point;
    OutputParams output;
    CheckpointParams checkpoint;
    GPUParams gpu;
    InitialCondition initial;
    BoundaryParams boundary;

    /// Load from a JSON file.
    static SimulationConfig from_json(const std::filesystem::path& path);

    /// Load from a JSON string.
    static SimulationConfig from_json_string(const std::string& json_str);

    /// Validate all parameters, compute derived quantities.
    void validate() const;

    /// Construct a Grid from the grid parameters.
    [[nodiscard]] Grid make_grid() const;
};

} // namespace ac
