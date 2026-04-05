#include "core/SimulationConfig.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>

namespace ac {

using json = nlohmann::json;

// ── JSON deserialization helpers ───────────────────────────────────────────

static TimeScheme parse_time_scheme(const std::string& s) {
    if (s == "euler") return TimeScheme::Euler;
    if (s == "heun")  return TimeScheme::Heun;
    if (s == "rk4")   return TimeScheme::RK4;
    if (s == "imex")  return TimeScheme::IMEX;
    throw std::invalid_argument("Unknown time scheme: " + s);
}

static StencilType parse_stencil(const std::string& s) {
    if (s == "7pt" || s == "standard") return StencilType::Standard7Point;
    if (s == "27pt" || s == "isotropic") return StencilType::Isotropic27Point;
    throw std::invalid_argument("Unknown stencil type: " + s);
}

static BCType parse_bc_type(const std::string& s) {
    if (s == "dirichlet") return BCType::Dirichlet;
    if (s == "neumann")   return BCType::Neumann;
    if (s == "periodic")  return BCType::Periodic;
    if (s == "robin")     return BCType::Robin;
    throw std::invalid_argument("Unknown boundary condition type: " + s);
}

static BoundaryConfig parse_boundary_config(const json& j) {
    BoundaryConfig bc;
    if (j.contains("type"))  bc.type = parse_bc_type(j["type"].get<std::string>());
    if (j.contains("value")) bc.value = j["value"].get<Real>();
    if (j.contains("flux"))  bc.flux  = j["flux"].get<Real>();
    if (j.contains("alpha")) bc.alpha = j["alpha"].get<Real>();
    if (j.contains("beta"))  bc.beta  = j["beta"].get<Real>();
    if (j.contains("gamma")) bc.gamma = j["gamma"].get<Real>();
    return bc;
}

static SimulationConfig parse_config(const json& j)
{
    SimulationConfig cfg;

    // Physics
    if (j.contains("physics")) {
        const auto& p = j["physics"];
        if (p.contains("delta"))   cfg.physics.delta   = p["delta"].get<Real>();
        if (p.contains("epsilon")) cfg.physics.epsilon = p["epsilon"].get<Real>();
        if (p.contains("W0"))      cfg.physics.W0      = p["W0"].get<Real>();
        if (p.contains("beta0"))   cfg.physics.beta0   = p["beta0"].get<Real>();
        if (p.contains("D"))       cfg.physics.D       = p["D"].get<Real>();
        if (p.contains("d0"))      cfg.physics.d0      = p["d0"].get<Real>();
        if (p.contains("a1"))      cfg.physics.a1      = p["a1"].get<Real>();
        if (p.contains("a2"))      cfg.physics.a2      = p["a2"].get<Real>();
    }

    // Grid
    if (j.contains("grid")) {
        const auto& g = j["grid"];
        if (g.contains("Nx")) cfg.grid.Nx = g["Nx"].get<int>();
        if (g.contains("Ny")) cfg.grid.Ny = g["Ny"].get<int>();
        if (g.contains("Nz")) cfg.grid.Nz = g["Nz"].get<int>();
        if (g.contains("dx")) cfg.grid.dx = g["dx"].get<Real>();
        if (g.contains("dy")) cfg.grid.dy = g["dy"].get<Real>();
        if (g.contains("dz")) cfg.grid.dz = g["dz"].get<Real>();
    }

    // Time
    if (j.contains("time")) {
        const auto& t = j["time"];
        if (t.contains("dt"))        cfg.time.dt        = t["dt"].get<Real>();
        if (t.contains("dt_min"))    cfg.time.dt_min    = t["dt_min"].get<Real>();
        if (t.contains("dt_max"))    cfg.time.dt_max    = t["dt_max"].get<Real>();
        if (t.contains("max_steps")) cfg.time.max_steps = t["max_steps"].get<int>();
        if (t.contains("scheme"))    cfg.time.scheme    = parse_time_scheme(t["scheme"].get<std::string>());
        if (t.contains("adaptive"))  cfg.time.adaptive  = t["adaptive"].get<bool>();
        if (t.contains("adaptive_tolerance"))
            cfg.time.adaptive_tolerance = t["adaptive_tolerance"].get<Real>();
        if (t.contains("cfl_safety")) cfg.time.cfl_safety = t["cfl_safety"].get<Real>();
    }

    // Stencil
    if (j.contains("stencil")) {
        cfg.stencil = parse_stencil(j["stencil"].get<std::string>());
    }

    // Output
    if (j.contains("output")) {
        const auto& o = j["output"];
        if (o.contains("frequency"))  cfg.output.frequency  = o["frequency"].get<int>();
        if (o.contains("output_dir")) cfg.output.output_dir = o["output_dir"].get<std::string>();
        if (o.contains("format"))     cfg.output.format     = o["format"].get<std::string>();
        if (o.contains("async_io"))   cfg.output.async_io   = o["async_io"].get<bool>();
    }

    // Checkpoint
    if (j.contains("checkpoint")) {
        const auto& c = j["checkpoint"];
        if (c.contains("frequency"))      cfg.checkpoint.frequency      = c["frequency"].get<int>();
        if (c.contains("checkpoint_dir")) cfg.checkpoint.checkpoint_dir = c["checkpoint_dir"].get<std::string>();
        if (c.contains("keep_last"))      cfg.checkpoint.keep_last      = c["keep_last"].get<int>();
        if (c.contains("restart_file"))   cfg.checkpoint.restart_file   = c["restart_file"].get<std::string>();
    }

    // GPU
    if (j.contains("gpu")) {
        const auto& g = j["gpu"];
        if (g.contains("device_ids"))   cfg.gpu.device_ids   = g["device_ids"].get<std::vector<int>>();
        if (g.contains("block_size"))   cfg.gpu.block_size_1d = g["block_size"].get<int>();
        if (g.contains("multi_gpu"))    cfg.gpu.multi_gpu     = g["multi_gpu"].get<bool>();
    }

    // Initial condition
    if (j.contains("initial")) {
        const auto& ic = j["initial"];
        if (ic.contains("seed_radius")) cfg.initial.seed_radius = ic["seed_radius"].get<Real>();
    }

    // Boundary conditions
    if (j.contains("boundary")) {
        const auto& b = j["boundary"];
        // Uniform BC (backward compatible)
        if (b.contains("phi") && b["phi"].is_object() && !b["phi"].contains("x_lo")) {
            cfg.boundary.phi_bc = parse_boundary_config(b["phi"]);
            cfg.boundary.phi_faces = PerFaceBoundary::uniform(cfg.boundary.phi_bc);
        }
        if (b.contains("u") && b["u"].is_object() && !b["u"].contains("x_lo")) {
            cfg.boundary.u_bc = parse_boundary_config(b["u"]);
            cfg.boundary.u_faces = PerFaceBoundary::uniform(cfg.boundary.u_bc);
        }
        // Per-face BC (new format)
        static constexpr const char* face_names[] = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"};
        if (b.contains("phi") && b["phi"].is_object() && b["phi"].contains("x_lo")) {
            cfg.boundary.per_face = true;
            const auto& p = b["phi"];
            for (int i = 0; i < 6; ++i) {
                if (p.contains(face_names[i])) {
                    cfg.boundary.phi_faces.faces[i] = parse_boundary_config(p[face_names[i]]);
                }
            }
        }
        if (b.contains("u") && b["u"].is_object() && b["u"].contains("x_lo")) {
            cfg.boundary.per_face = true;
            const auto& u = b["u"];
            for (int i = 0; i < 6; ++i) {
                if (u.contains(face_names[i])) {
                    cfg.boundary.u_faces.faces[i] = parse_boundary_config(u[face_names[i]]);
                }
            }
        }
    }

    return cfg;
}

SimulationConfig SimulationConfig::from_json(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path.string());
    }
    json j = json::parse(file);
    auto cfg = parse_config(j);
    spdlog::info("Configuration loaded from {}", path.string());
    return cfg;
}

SimulationConfig SimulationConfig::from_json_string(const std::string& json_str)
{
    json j = json::parse(json_str);
    return parse_config(j);
}

void SimulationConfig::validate() const
{
    // Grid validation
    if (grid.Nx < 3 || grid.Ny < 3 || grid.Nz < 3) {
        throw std::invalid_argument("Grid dimensions must be >= 3");
    }
    if (grid.dx <= 0.0 || grid.dy <= 0.0 || grid.dz <= 0.0) {
        throw std::invalid_argument("Grid spacing must be positive");
    }

    // Physics validation
    if (physics.W0 <= 0.0) throw std::invalid_argument("W0 must be positive");
    if (physics.D <= 0.0)  throw std::invalid_argument("D must be positive");
    if (physics.d0 <= 0.0) throw std::invalid_argument("d0 must be positive");
    if (physics.epsilon < 0.0 || physics.epsilon >= 1.0 / 3.0) {
        throw std::invalid_argument("epsilon must be in [0, 1/3)");
    }

    // Time validation
    if (time.dt <= 0.0) throw std::invalid_argument("dt must be positive");
    if (time.max_steps < 1) throw std::invalid_argument("max_steps must be >= 1");

    // CFL check
    Real min_dx = std::min({grid.dx, grid.dy, grid.dz});
    Real cfl_dt = time.cfl_safety * min_dx * min_dx / (2.0 * physics.D * 3.0);
    if (time.dt > cfl_dt && !time.adaptive) {
        spdlog::warn("dt={:.6e} exceeds CFL limit={:.6e} for explicit diffusion. "
                     "Consider enabling adaptive time stepping.", time.dt, cfl_dt);
    }

    // Output
    if (output.frequency < 1) throw std::invalid_argument("output frequency must be >= 1");

    // GPU
    if (gpu.device_ids.empty()) throw std::invalid_argument("At least one GPU device required");
    if (gpu.block_size_1d < 32 || gpu.block_size_1d > 1024) {
        throw std::invalid_argument("block_size must be in [32, 1024]");
    }

    spdlog::info("Configuration validated: {}x{}x{} grid, dt={}, {} scheme, {} stencil",
                 grid.Nx, grid.Ny, grid.Nz, time.dt,
                 time.scheme == TimeScheme::Euler ? "Euler" :
                 time.scheme == TimeScheme::Heun  ? "Heun" :
                 time.scheme == TimeScheme::RK4   ? "RK4" : "IMEX",
                 stencil == StencilType::Standard7Point ? "7pt" : "27pt");
}

Grid SimulationConfig::make_grid() const
{
    return Grid(
        Dim3{grid.Nx, grid.Ny, grid.Nz},
        Spacing{grid.dx, grid.dy, grid.dz}
    );
}

} // namespace ac
