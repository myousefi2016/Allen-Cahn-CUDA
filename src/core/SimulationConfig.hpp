#pragma once

#include "core/Grid.hpp"

#include <array>
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
    Real value = -1.0; ///< Dirichlet value
    Real flux = 0.0;   ///< Neumann flux
    Real alpha = 1.0;  ///< Robin: alpha*u + beta*du/dn = gamma
    Real beta = 0.0;
    Real gamma = 0.0;
};

/// The coefficients of a Robin face as the ghost fill uses them, for the
/// spacing ds normal to the face: alpha, beta and gamma times the same power
/// of two 2^-k, k chosen so that max(|alpha|, |beta|/ds) lies in (1/8, 1/2).
/// With alpha*beta >= 0 the denominator alpha + beta/ds then lies in (1/8, 1)
/// in magnitude: it cannot vanish, the scaled gamma is smaller in magnitude
/// than the ghost constant gamma/(alpha + beta/ds), and |beta*u/ds| < |u|/2.
/// A power-of-two scaling is exact, so the ghost
/// (gamma + beta*u/ds)/(alpha + beta/ds) has the same bits as with the
/// unscaled coefficients whenever every intermediate of both is a normal
/// double or zero. Throws std::invalid_argument unless alpha and beta are
/// finite and not both zero and ds is positive and finite.
struct RobinCoefficients {
    Real alpha = 0.0;
    Real beta = 0.0;
    Real gamma = 0.0;
};
[[nodiscard]] RobinCoefficients scaled_robin(const BoundaryConfig& bc, Real ds);

// ── Physics parameters ─────────────────────────────────────────────────────
struct PhysicsParams {
    Real delta = 0.8;    ///< Dimensionless undercooling
    Real epsilon = 0.05; ///< Anisotropy strength (> 1/15 requires willmore_beta > 0)
    Real W0 = 1.0;       ///< Interface width parameter
    Real beta0 = 0.0;    ///< Kinetic coefficient
    Real D = 2.0;        ///< Thermal diffusivity
    Real d0 = 0.5;       ///< Capillary length
    /// Thin-interface constants of this model (coupling (1 - phi^2)^2, latent heat
    /// dphi/dt / 2): a1 = I/J = 5 sqrt(2)/8 and a2 = (K + J F)/(2 I) = 47/75, with
    /// which lambda() gives the capillary length d0 and tau0() the kinetic
    /// coefficient beta0. With willmore_beta > 0 only along orientations with
    /// A(n) = 1: elsewhere the regularisation changes the flat profile they are
    /// integrals of (docs/THEORY.md 4.3, 5.2).
    Real a1 = 1.25 / std::sqrt(2.0);
    Real a2 = 47.0 / 75.0;
    /// Willmore regularisation strength beta >= 0 (dimensionless): adds
    /// (beta/2) * int omega^2, omega = W0^2 lap(phi) + phi - phi^3, to the free
    /// energy. omega uses the isotropic Laplacian, so on the planar equilibrium
    /// profile (width W0 A(n)) it is (1 - A(n)^-2)(phi - phi^3), zero only
    /// where A(n) = 1: for epsilon > 0 flat interfaces with A(n) != 1 change
    /// too, not only curved interfaces and corners (corner length
    /// W0 sqrt(beta / |S|), S the most negative gradient-energy stiffness;
    /// docs/THEORY.md section 4.3).
    /// Required (> 0) when epsilon > 1/15, where the gradient energy is not
    /// convex and the unregularised equation is ill-posed.
    Real willmore_beta = 0.0;

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

/// Whether every dimension is >= 1 and Nx * Ny * Nz fits int, the kernels' 32-bit cell
/// index; computed without overflow (the product itself can exceed 2^63).
[[nodiscard]] bool grid_points_fit_int(const GridParams& grid);

// ── Time integration ───────────────────────────────────────────────────────
enum class TimeScheme { Euler, Heun, RK4, IMEX };

struct TimeParams {
    Real dt = 0.01;
    Real dt_min = 1e-6;
    Real dt_max = 0.1;
    int max_steps = 6000;
    TimeScheme scheme = TimeScheme::Euler;
    bool adaptive = false;
    Real adaptive_tolerance = 0.01; ///< Max allowed |dphi| per step
    Real cfl_safety = 0.9;          ///< In (0, 1]: adaptive dt <= cfl_safety * stability limit
    // Saturation guard: once the solid (phi > saturation_threshold) reaches
    // the cells next to a wall, the run no longer represents growth into an
    // unbounded melt (the wall's boundary condition now acts on the solid), so
    // the engine writes a final checkpoint and output and exits cleanly.
    // A check reduces those cells on the GPU and copies one double per domain
    // to the host, not the field (CudaSolver::compute_boundary_max_phi).
    bool exit_on_saturation = true;
    Real saturation_threshold = -0.5; ///< phi above this next to a wall => saturated
    int saturation_check_freq = 100;  ///< check every N steps
};

// ── Stencil type ───────────────────────────────────────────────────────────
enum class StencilType { Standard7Point, Isotropic27Point };

// ── Output parameters ──────────────────────────────────────────────────────
struct OutputParams {
    int frequency = 100;
    std::filesystem::path output_dir = "./out";
    std::string format = "vts"; ///< "vts" or "raw"
    bool async_io = true;
};

// ── Checkpoint parameters ──────────────────────────────────────────────────
struct CheckpointParams {
    int frequency = 500; ///< Steps between checkpoints; 0 disables periodic checkpoints
    std::filesystem::path checkpoint_dir = "./checkpoints";
    int keep_last = 3; ///< Rolling checkpoint count
    std::optional<std::filesystem::path> restart_file;
};

// ── GPU parameters ─────────────────────────────────────────────────────────
/// One domain per entry of device_ids (an id may repeat) when multi_gpu is true and there
/// are several; otherwise the whole grid runs on device_ids[0], and validate() rejects a
/// list of several ids, of which all but the first would be ignored.
struct GPUParams {
    std::vector<int> device_ids = {0};
    bool multi_gpu = false;
};

/// X planes every domain of a multi-GPU split must own (MultiGPUSolver): the BC at an X
/// wall reads the plane next to it, which the first and the last domain must own, as must
/// the periodic wrap sources, global planes 1 and Nx - 2; the neighbours' halos are copied
/// from owned planes (MultiGPUSolver asserts that its halo width does not exceed this).
/// The split gives every domain Nx / domains planes and one more to the first
/// Nx % domains, so the last domain owns the fewest.
inline constexpr int kMinOwnedPlanes = 2;

// ── Initial condition ──────────────────────────────────────────────────────
struct InitialCondition {
    Real seed_radius = 2.0; ///< Physical length (same units as W0 and grid spacing)
};

/// Face identifiers for per-face boundary conditions.
enum class Face : int { XLo = 0, XHi = 1, YLo = 2, YHi = 3, ZLo = 4, ZHi = 5 };

/// Per-face boundary conditions for a single field.
/// Supports uniform (same for all 6 faces) or per-face specification.
struct PerFaceBoundary {
    std::array<BoundaryConfig, 6> faces; ///< [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]

    /// Construct with uniform BC on all faces.
    static PerFaceBoundary uniform(const BoundaryConfig& bc) {
        PerFaceBoundary p;
        p.faces.fill(bc);
        return p;
    }

    /// Access by face index.
    [[nodiscard]] const BoundaryConfig& operator[](Face f) const {
        return faces[static_cast<std::size_t>(f)];
    }
    [[nodiscard]] BoundaryConfig& operator[](Face f) { return faces[static_cast<std::size_t>(f)]; }

    /// Access by axis (0-2) and side (0=lo, 1=hi).
    [[nodiscard]] const BoundaryConfig& get(int axis, int side) const {
        return faces[static_cast<std::size_t>(axis * 2 + side)];
    }
};

// ── Boundary conditions (per-face: x_lo, x_hi, y_lo, y_hi, z_lo, z_hi) ──
struct BoundaryParams {
    BoundaryConfig phi_bc;     ///< Uniform BC for phi (backward compat)
    BoundaryConfig u_bc;       ///< Uniform BC for u (backward compat)
    PerFaceBoundary phi_faces; ///< Per-face BCs for phi
    PerFaceBoundary u_faces;   ///< Per-face BCs for u
    bool per_face = false;     ///< Whether per-face BCs are specified
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
