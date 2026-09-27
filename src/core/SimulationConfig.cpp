#include "core/SimulationConfig.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ac {

using json = nlohmann::json;

// ── JSON deserialization helpers ───────────────────────────────────────────

static TimeScheme parse_time_scheme(const std::string& s) {
    if (s == "euler")
        return TimeScheme::Euler;
    if (s == "heun")
        return TimeScheme::Heun;
    if (s == "rk4")
        return TimeScheme::RK4;
    if (s == "imex")
        return TimeScheme::IMEX;
    throw std::invalid_argument("Unknown time scheme: " + s);
}

static StencilType parse_stencil(const std::string& s) {
    if (s == "7pt" || s == "standard")
        return StencilType::Standard7Point;
    if (s == "27pt" || s == "isotropic")
        return StencilType::Isotropic27Point;
    throw std::invalid_argument("Unknown stencil type: " + s);
}

static BCType parse_bc_type(const std::string& s) {
    if (s == "dirichlet")
        return BCType::Dirichlet;
    if (s == "neumann")
        return BCType::Neumann;
    if (s == "periodic")
        return BCType::Periodic;
    if (s == "robin")
        return BCType::Robin;
    throw std::invalid_argument("Unknown boundary condition type: " + s);
}

/// Throws unless `obj` is a JSON object.
static void require_object(const json& obj, const std::string& section) {
    if (!obj.is_object())
        throw std::invalid_argument("config: \"" + section + "\" must be a JSON object");
}

/// Throws if `obj` has a key that is neither in `allowed` nor a comment (a key
/// starting with '_'). A silently ignored key is a setting the user believes
/// is in effect but is not (a typo, a removed option, a misplaced key).
static void require_known_keys(const json& obj, const std::string& section,
                               std::initializer_list<std::string_view> allowed) {
    require_object(obj, section);
    std::string unknown;
    for (const auto& item : obj.items()) {
        const std::string& key = item.key();
        if (!key.empty() && key.front() == '_')
            continue;
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
            unknown += (unknown.empty() ? "\"" : ", \"") + key + "\"";
    }
    if (!unknown.empty()) {
        std::string list;
        for (std::string_view a : allowed)
            list += (list.empty() ? "" : ", ") + std::string(a);
        throw std::invalid_argument("config: unknown key(s) " + unknown + " in \"" + section +
                                    "\" (allowed: " + list + "; keys starting with '_' are " +
                                    "comments)");
    }
}

static BoundaryConfig parse_boundary_config(const json& j, const std::string& section) {
    require_known_keys(j, section, {"type", "value", "flux", "alpha", "beta", "gamma"});
    BoundaryConfig bc;
    if (j.contains("type"))
        bc.type = parse_bc_type(j["type"].get<std::string>());
    if (j.contains("value"))
        bc.value = j["value"].get<Real>();
    if (j.contains("flux"))
        bc.flux = j["flux"].get<Real>();
    if (j.contains("alpha"))
        bc.alpha = j["alpha"].get<Real>();
    if (j.contains("beta"))
        bc.beta = j["beta"].get<Real>();
    if (j.contains("gamma"))
        bc.gamma = j["gamma"].get<Real>();
    return bc;
}

/// boundary.phi / boundary.u: either one BC object applied to every face, or
/// an object with exactly the six face keys (x_lo, x_hi, y_lo, y_hi, z_lo,
/// z_hi), each a BC object. Any face key selects the per-face form, which
/// then must name all six faces and nothing else.
static void parse_field_boundary(const json& f, const std::string& section, BoundaryConfig& uniform,
                                 PerFaceBoundary& faces, bool& per_face) {
    static constexpr std::string_view face_names[] = {"x_lo", "x_hi", "y_lo",
                                                      "y_hi", "z_lo", "z_hi"};
    require_object(f, section);
    bool any_face = false;
    for (std::string_view name : face_names)
        any_face = any_face || f.contains(std::string(name));
    if (!any_face) {
        uniform = parse_boundary_config(f, section);
        faces = PerFaceBoundary::uniform(uniform);
        return;
    }
    require_known_keys(f, section, {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"});
    std::string missing;
    for (std::string_view name : face_names)
        if (!f.contains(std::string(name)))
            missing += (missing.empty() ? "" : ", ") + std::string(name);
    if (!missing.empty()) {
        throw std::invalid_argument("config: per-face \"" + section +
                                    "\" must specify all six faces; missing: " + missing);
    }
    for (std::size_t i = 0; i < 6; ++i) {
        const std::string name(face_names[i]);
        faces.faces[i] = parse_boundary_config(f[name], section + "." + name);
    }
    per_face = true;
}

static SimulationConfig parse_config(const json& j) {
    require_known_keys(j, "<root>",
                       {"physics", "grid", "time", "stencil", "output", "checkpoint", "gpu",
                        "initial", "boundary"});
    SimulationConfig cfg;

    // Physics
    if (j.contains("physics")) {
        const auto& p = j["physics"];
        require_known_keys(p, "physics",
                           {"delta", "epsilon", "W0", "beta0", "D", "d0", "a1", "a2"});
        if (p.contains("delta"))
            cfg.physics.delta = p["delta"].get<Real>();
        if (p.contains("epsilon"))
            cfg.physics.epsilon = p["epsilon"].get<Real>();
        if (p.contains("W0"))
            cfg.physics.W0 = p["W0"].get<Real>();
        if (p.contains("beta0"))
            cfg.physics.beta0 = p["beta0"].get<Real>();
        if (p.contains("D"))
            cfg.physics.D = p["D"].get<Real>();
        if (p.contains("d0"))
            cfg.physics.d0 = p["d0"].get<Real>();
        if (p.contains("a1"))
            cfg.physics.a1 = p["a1"].get<Real>();
        if (p.contains("a2"))
            cfg.physics.a2 = p["a2"].get<Real>();
    }

    // Grid
    if (j.contains("grid")) {
        const auto& g = j["grid"];
        require_known_keys(g, "grid", {"Nx", "Ny", "Nz", "dx", "dy", "dz"});
        if (g.contains("Nx"))
            cfg.grid.Nx = g["Nx"].get<int>();
        if (g.contains("Ny"))
            cfg.grid.Ny = g["Ny"].get<int>();
        if (g.contains("Nz"))
            cfg.grid.Nz = g["Nz"].get<int>();
        if (g.contains("dx"))
            cfg.grid.dx = g["dx"].get<Real>();
        if (g.contains("dy"))
            cfg.grid.dy = g["dy"].get<Real>();
        if (g.contains("dz"))
            cfg.grid.dz = g["dz"].get<Real>();
    }

    // Time
    if (j.contains("time")) {
        const auto& t = j["time"];
        require_known_keys(t, "time",
                           {"dt", "dt_min", "dt_max", "max_steps", "scheme", "adaptive",
                            "adaptive_tolerance", "cfl_safety", "exit_on_saturation",
                            "saturation_threshold", "saturation_check_freq"});
        if (t.contains("dt"))
            cfg.time.dt = t["dt"].get<Real>();
        if (t.contains("dt_min"))
            cfg.time.dt_min = t["dt_min"].get<Real>();
        if (t.contains("dt_max"))
            cfg.time.dt_max = t["dt_max"].get<Real>();
        if (t.contains("max_steps"))
            cfg.time.max_steps = t["max_steps"].get<int>();
        if (t.contains("scheme"))
            cfg.time.scheme = parse_time_scheme(t["scheme"].get<std::string>());
        if (t.contains("adaptive"))
            cfg.time.adaptive = t["adaptive"].get<bool>();
        if (t.contains("adaptive_tolerance"))
            cfg.time.adaptive_tolerance = t["adaptive_tolerance"].get<Real>();
        if (t.contains("cfl_safety"))
            cfg.time.cfl_safety = t["cfl_safety"].get<Real>();
        if (t.contains("exit_on_saturation"))
            cfg.time.exit_on_saturation = t["exit_on_saturation"].get<bool>();
        if (t.contains("saturation_threshold"))
            cfg.time.saturation_threshold = t["saturation_threshold"].get<Real>();
        if (t.contains("saturation_check_freq"))
            cfg.time.saturation_check_freq = t["saturation_check_freq"].get<int>();
    }

    // Stencil
    if (j.contains("stencil")) {
        cfg.stencil = parse_stencil(j["stencil"].get<std::string>());
    }

    // Output
    if (j.contains("output")) {
        const auto& o = j["output"];
        require_known_keys(o, "output", {"frequency", "output_dir", "format", "async_io"});
        if (o.contains("frequency"))
            cfg.output.frequency = o["frequency"].get<int>();
        if (o.contains("output_dir"))
            cfg.output.output_dir = o["output_dir"].get<std::string>();
        if (o.contains("format"))
            cfg.output.format = o["format"].get<std::string>();
        if (o.contains("async_io"))
            cfg.output.async_io = o["async_io"].get<bool>();
    }

    // Checkpoint
    if (j.contains("checkpoint")) {
        const auto& c = j["checkpoint"];
        require_known_keys(c, "checkpoint",
                           {"frequency", "checkpoint_dir", "keep_last", "restart_file"});
        if (c.contains("frequency"))
            cfg.checkpoint.frequency = c["frequency"].get<int>();
        if (c.contains("checkpoint_dir"))
            cfg.checkpoint.checkpoint_dir = c["checkpoint_dir"].get<std::string>();
        if (c.contains("keep_last"))
            cfg.checkpoint.keep_last = c["keep_last"].get<int>();
        if (c.contains("restart_file"))
            cfg.checkpoint.restart_file = c["restart_file"].get<std::string>();
    }

    // GPU
    if (j.contains("gpu")) {
        const auto& g = j["gpu"];
        require_known_keys(g, "gpu", {"device_ids", "multi_gpu"});
        if (g.contains("device_ids"))
            cfg.gpu.device_ids = g["device_ids"].get<std::vector<int>>();
        if (g.contains("multi_gpu"))
            cfg.gpu.multi_gpu = g["multi_gpu"].get<bool>();
    }

    // Initial condition
    if (j.contains("initial")) {
        const auto& ic = j["initial"];
        require_known_keys(ic, "initial", {"seed_radius"});
        if (ic.contains("seed_radius"))
            cfg.initial.seed_radius = ic["seed_radius"].get<Real>();
    }

    // Boundary conditions
    if (j.contains("boundary")) {
        const auto& b = j["boundary"];
        require_known_keys(b, "boundary", {"phi", "u"});
        if (b.contains("phi"))
            parse_field_boundary(b["phi"], "boundary.phi", cfg.boundary.phi_bc,
                                 cfg.boundary.phi_faces, cfg.boundary.per_face);
        if (b.contains("u"))
            parse_field_boundary(b["u"], "boundary.u", cfg.boundary.u_bc, cfg.boundary.u_faces,
                                 cfg.boundary.per_face);
    }

    return cfg;
}

SimulationConfig SimulationConfig::from_json(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path.string());
    }
    json j = json::parse(file);
    auto cfg = parse_config(j);
    cfg.validate();
    spdlog::info("Configuration loaded from {}", path.string());
    return cfg;
}

SimulationConfig SimulationConfig::from_json_string(const std::string& json_str) {
    json j = json::parse(json_str);
    auto cfg = parse_config(j);
    cfg.validate();
    return cfg;
}

void SimulationConfig::validate() const {
    // Grid validation
    if (grid.Nx < 3 || grid.Ny < 3 || grid.Nz < 3) {
        throw std::invalid_argument("Grid dimensions must be >= 3");
    }
    if (grid.dx <= 0.0 || grid.dy <= 0.0 || grid.dz <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }
    // Kernels index cells with 32-bit int (idx3d, linear_to_3d).
    if (static_cast<long long>(grid.Nx) * grid.Ny * grid.Nz > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(
            "Grid has " + std::to_string(static_cast<long long>(grid.Nx) * grid.Ny * grid.Nz) +
            " points; at most " + std::to_string(std::numeric_limits<int>::max()) +
            " are supported (32-bit cell indices)");
    }

    // 27-point isotropic stencil requires cubic grid spacing
    if (stencil == StencilType::Isotropic27Point) {
        if (std::abs(grid.dx - grid.dy) > 1e-14 || std::abs(grid.dx - grid.dz) > 1e-14) {
            throw std::invalid_argument(
                "27-point isotropic stencil requires cubic grid spacing (dx == dy == dz)");
        }
    }

    // Physics validation
    if (physics.W0 <= 0.0)
        throw std::invalid_argument("W0 must be positive");
    if (physics.D <= 0.0)
        throw std::invalid_argument("D must be positive");
    if (physics.d0 <= 0.0)
        throw std::invalid_argument("d0 must be positive");
    if (physics.epsilon < 0.0 || physics.epsilon >= 1.0 / 3.0) {
        throw std::invalid_argument("epsilon must be in [0, 1/3)");
    }

    // Time validation
    if (time.dt <= 0.0)
        throw std::invalid_argument("dt must be positive");
    if (time.max_steps < 1)
        throw std::invalid_argument("max_steps must be >= 1");
    if (time.dt_min <= 0.0)
        throw std::invalid_argument("dt_min must be positive");
    if (time.dt_max <= 0.0)
        throw std::invalid_argument("dt_max must be positive");
    if (time.dt_min >= time.dt_max)
        throw std::invalid_argument("dt_min must be < dt_max");
    if (time.cfl_safety <= 0.0)
        throw std::invalid_argument("cfl_safety must be positive");

    // CFL check (thermal diffusion + phase-field effective diffusivity)
    Real min_dx = std::min({grid.dx, grid.dy, grid.dz});
    Real inv_h2_sum = 3.0 / (min_dx * min_dx);
    Real thermal_cfl = time.cfl_safety / (2.0 * physics.D * inv_h2_sum);
    Real A_max = 1.0 + physics.epsilon;
    Real D_phi = physics.W0 * physics.W0 * A_max * A_max / physics.tau0();
    Real phi_cfl = time.cfl_safety / (2.0 * D_phi * inv_h2_sum);
    Real cfl_dt = std::min(thermal_cfl, phi_cfl);
    if (time.dt > cfl_dt && !time.adaptive) {
        spdlog::warn("dt={:.6e} exceeds CFL limit={:.6e} (min of thermal and phase-field). "
                     "Consider enabling adaptive time stepping.",
                     time.dt, cfl_dt);
    }

    // Output
    if (output.frequency < 1)
        throw std::invalid_argument("output frequency must be >= 1");
    if (output.format != "vts" && output.format != "raw") {
        throw std::invalid_argument("output format must be 'vts' or 'raw', got: " + output.format);
    }

    // Checkpoint (frequency 0 disables periodic checkpoints; see
    // CheckpointManager::should_checkpoint)
    if (checkpoint.frequency < 0)
        throw std::invalid_argument("checkpoint frequency must be >= 0 (0 disables checkpoints)");
    if (checkpoint.keep_last < 1)
        throw std::invalid_argument("checkpoint keep_last must be >= 1");
    if (checkpoint.restart_file.has_value() && checkpoint.restart_file->empty())
        throw std::invalid_argument("checkpoint restart_file must not be empty (omit it instead)");

    // Initial condition
    if (initial.seed_radius <= 0.0)
        throw std::invalid_argument("seed_radius must be positive");

    // Adaptive stepping
    if (time.adaptive && time.adaptive_tolerance <= 0.0) {
        throw std::invalid_argument(
            "adaptive_tolerance must be positive when adaptive stepping is enabled");
    }

    // Robin BC validation
    auto validate_bc = [](const BoundaryConfig& bc, const std::string& name) {
        if (bc.type == BCType::Robin) {
            if (bc.alpha == 0.0 && bc.beta == 0.0) {
                throw std::invalid_argument("Robin BC for " + name +
                                            ": alpha and beta cannot both be zero");
            }
        }
    };
    validate_bc(boundary.phi_bc, "phi");
    validate_bc(boundary.u_bc, "u");
    for (std::size_t i = 0; i < 6; ++i) {
        validate_bc(boundary.phi_faces.faces[i], "phi face " + std::to_string(i));
        validate_bc(boundary.u_faces.faces[i], "u face " + std::to_string(i));
    }

    // GPU
    if (gpu.device_ids.empty())
        throw std::invalid_argument("At least one GPU device required");

    spdlog::info("Configuration validated: {}x{}x{} grid, dt={}, {} scheme, {} stencil", grid.Nx,
                 grid.Ny, grid.Nz, time.dt,
                 time.scheme == TimeScheme::Euler  ? "Euler"
                 : time.scheme == TimeScheme::Heun ? "Heun"
                 : time.scheme == TimeScheme::RK4  ? "RK4"
                                                   : "IMEX",
                 stencil == StencilType::Standard7Point ? "7pt" : "27pt");
}

Grid SimulationConfig::make_grid() const {
    return Grid(Dim3{grid.Nx, grid.Ny, grid.Nz}, Spacing{grid.dx, grid.dy, grid.dz});
}

} // namespace ac
