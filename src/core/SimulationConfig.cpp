#include "core/SimulationConfig.hpp"

#include "core/StabilityLimits.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

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

RobinCoefficients scaled_robin(const BoundaryConfig& bc, Real ds) {
    if (!std::isfinite(bc.alpha) || !std::isfinite(bc.beta) ||
        (bc.alpha == 0.0 && bc.beta == 0.0) || !std::isfinite(ds) || !(ds > 0.0)) {
        throw std::invalid_argument("scaled_robin: alpha and beta must be finite and not both "
                                    "zero, and ds positive and finite");
    }
    // The larger of the exponents of alpha and beta/ds, the latter from those
    // of beta and ds so that beta/ds itself cannot overflow or underflow
    // here. |alpha| < 2^(ilogb(alpha) + 1) and |beta|/ds < 2^(ilogb(beta) -
    // ilogb(ds) + 1), so 2 more bring both below 1/2, and the larger stays
    // above 1/8 (above 1/4 for alpha).
    int k = std::numeric_limits<int>::min();
    if (bc.alpha != 0.0)
        k = std::ilogb(bc.alpha);
    if (bc.beta != 0.0)
        k = std::max(k, std::ilogb(bc.beta) - std::ilogb(ds));
    k += 2;
    return {std::ldexp(bc.alpha, -k), std::ldexp(bc.beta, -k), std::ldexp(bc.gamma, -k)};
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
/// v with 17 significant digits: two different doubles never print alike.
static std::string digits17(double v) {
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

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

/// The int at j, which must be a JSON integer (no fraction or exponent, not a
/// boolean or string) in the range of int. nlohmann's get<int>() converts
/// without a check: 3.9 became 3, 4294967300 became 4 and true became 1.
static int read_int(const json& j, const std::string& key) {
    constexpr int lo = std::numeric_limits<int>::min(), hi = std::numeric_limits<int>::max();
    if (j.is_number_unsigned()) {
        const auto v = j.get<json::number_unsigned_t>();
        if (v <= static_cast<json::number_unsigned_t>(hi))
            return static_cast<int>(v);
    } else if (j.is_number_integer()) {
        const auto v = j.get<json::number_integer_t>();
        if (v >= lo && v <= hi)
            return static_cast<int>(v);
    }
    throw std::invalid_argument("config: " + key + " must be an integer from " +
                                std::to_string(lo) + " to " + std::to_string(hi) + ", got " +
                                j.dump());
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
        require_known_keys(
            p, "physics",
            {"delta", "epsilon", "W0", "beta0", "D", "d0", "a1", "a2", "willmore_beta"});
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
        if (p.contains("willmore_beta"))
            cfg.physics.willmore_beta = p["willmore_beta"].get<Real>();
    }

    // Grid
    if (j.contains("grid")) {
        const auto& g = j["grid"];
        require_known_keys(g, "grid", {"Nx", "Ny", "Nz", "dx", "dy", "dz"});
        if (g.contains("Nx"))
            cfg.grid.Nx = read_int(g["Nx"], "grid.Nx");
        if (g.contains("Ny"))
            cfg.grid.Ny = read_int(g["Ny"], "grid.Ny");
        if (g.contains("Nz"))
            cfg.grid.Nz = read_int(g["Nz"], "grid.Nz");
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
            cfg.time.max_steps = read_int(t["max_steps"], "time.max_steps");
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
            cfg.time.saturation_check_freq =
                read_int(t["saturation_check_freq"], "time.saturation_check_freq");
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
            cfg.output.frequency = read_int(o["frequency"], "output.frequency");
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
            cfg.checkpoint.frequency = read_int(c["frequency"], "checkpoint.frequency");
        if (c.contains("checkpoint_dir"))
            cfg.checkpoint.checkpoint_dir = c["checkpoint_dir"].get<std::string>();
        if (c.contains("keep_last"))
            cfg.checkpoint.keep_last = read_int(c["keep_last"], "checkpoint.keep_last");
        if (c.contains("restart_file"))
            cfg.checkpoint.restart_file = c["restart_file"].get<std::string>();
    }

    // GPU
    if (j.contains("gpu")) {
        const auto& g = j["gpu"];
        require_known_keys(g, "gpu", {"device_ids", "multi_gpu"});
        if (g.contains("device_ids")) {
            const auto& ids = g["device_ids"];
            if (!ids.is_array())
                throw std::invalid_argument(
                    "config: gpu.device_ids must be an array of integers, got " + ids.dump());
            cfg.gpu.device_ids.clear();
            for (std::size_t i = 0; i < ids.size(); ++i)
                cfg.gpu.device_ids.push_back(
                    read_int(ids[i], "gpu.device_ids[" + std::to_string(i) + "]"));
        }
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
    if (!(physics.willmore_beta >= 0.0))
        throw std::invalid_argument("physics.willmore_beta must be >= 0");
    // Above 1/15 the Hessian of the gradient energy 1/2 W0^2 A(p)^2 |p|^2 is
    // indefinite (first along <100>): the Allen-Cahn equation has negative
    // diffusion there and its discrete solutions do not converge under grid
    // refinement. The Willmore term restores well-posedness.
    if (physics.epsilon > 1.0 / 15.0 && physics.willmore_beta <= 0.0) {
        throw std::invalid_argument(
            "epsilon = " + std::to_string(physics.epsilon) +
            " exceeds 1/15, where the anisotropic gradient energy is not convex and the "
            "phase-field equation is ill-posed; set physics.willmore_beta > 0 to regularise it "
            "(docs/THEORY.md, Willmore regularisation)");
    }
    // The Willmore term is stiff (fourth order) and is integrated linearly
    // implicitly; only the Euler and IMEX schemes have that phi update.
    if (physics.willmore_beta > 0.0 && time.scheme != TimeScheme::Euler &&
        time.scheme != TimeScheme::IMEX) {
        throw std::invalid_argument(
            "physics.willmore_beta > 0 requires time.scheme \"euler\" or \"imex\" (the "
            "fourth-order Willmore term is treated linearly implicitly in their phase-field "
            "update; heun and rk4 are fully explicit)");
    }
    // In a liquid with a small gradient along n, a perturbation mode with
    // Lam + beta L < 0 grows at every dt (docs/THEORY.md 10.1): beta does not
    // regularise the non-convex gradient energy at this spacing. Close to the
    // smallest beta that does, the search may not decide within its budget.
    std::optional<LiquidModeBound> unstable;
    try {
        unstable = phase_field_unstable_mode(*this);
    } catch (const LiquidModeSearchExhausted& e) {
        throw std::invalid_argument(
            "physics.willmore_beta = " + digits17(physics.willmore_beta) +
            ": the stability analysis cannot decide within its budget whether it regularises "
            "epsilon = " +
            digits17(physics.epsilon) +
            " at this grid spacing: the smallest Lam + beta L over the modes of a liquid with a "
            "small gradient lies in [" +
            digits17(e.lower) + ", " + digits17(e.upper) + "] (the mode n = (" +
            digits17(e.mode.n[0]) + ", " + digits17(e.mode.n[1]) + ", " + digits17(e.mode.n[2]) +
            "), wave numbers (" + digits17(e.mode.theta[0]) + ", " + digits17(e.mode.theta[1]) +
            ", " + digits17(e.mode.theta[2]) +
            ") rad per cell, has at most the upper end), and one below 0 would grow at every dt; "
            "increase willmore_beta (docs/THEORY.md 10.1)");
    }
    if (const auto& mode = unstable) {
        const LiquidMode& m = mode->mode;
        throw std::invalid_argument(
            "physics.willmore_beta = " + digits17(physics.willmore_beta) +
            " does not regularise epsilon = " + digits17(physics.epsilon) +
            " at this grid spacing: in a liquid with a small gradient along n = (" +
            digits17(m.n[0]) + ", " + digits17(m.n[1]) + ", " + digits17(m.n[2]) +
            ") the perturbation with wave numbers (" + digits17(m.theta[0]) + ", " +
            digits17(m.theta[1]) + ", " + digits17(m.theta[2]) +
            ") rad per cell has Lam + beta L <= " + digits17(mode->attained) +
            " < 0 and grows at every dt; increase willmore_beta (docs/THEORY.md 10.1)");
    }

    // Time validation
    if (time.dt <= 0.0)
        throw std::invalid_argument("dt must be positive");
    if (time.max_steps < 1)
        throw std::invalid_argument("max_steps must be >= 1");
    if (time.saturation_check_freq < 1)
        throw std::invalid_argument("saturation_check_freq must be >= 1");
    if (time.dt_min <= 0.0)
        throw std::invalid_argument("dt_min must be positive");
    if (time.dt_max <= 0.0)
        throw std::invalid_argument("dt_max must be positive");
    if (time.dt_min >= time.dt_max)
        throw std::invalid_argument("dt_min must be < dt_max");
    if (!(time.cfl_safety > 0.0 && time.cfl_safety <= 1.0))
        throw std::invalid_argument("cfl_safety must be in (0, 1]: adaptive stepping caps dt at "
                                    "cfl_safety times the stability limit");

    // Explicit stability limits (docs/THEORY.md 10.1): beyond them a
    // perturbation (of the liquid, for the phase field, to within the
    // tolerance of its certified bound) grows every step.
    // They are printed with 17 significant digits, so a rejected
    // dt never shows as equal to its limit. With
    // adaptive stepping the first step uses time.dt and every later step is
    // capped at cfl_safety times the limits; dt_min must not exceed that cap.
    const double thermal_limit = thermal_dt_limit(*this);
    const double phi_limit = phase_field_dt_limit(*this);
    auto check_step = [&](double dt, const std::string& what, double factor) {
        if (dt > factor * thermal_limit) {
            throw std::invalid_argument(
                what + " = " + digits17(dt) + " exceeds " +
                (factor < 1.0 ? "cfl_safety times " : "") + "the stability limit " +
                digits17(thermal_limit) + " of the explicit thermal update (dt D rho <= " +
                digits17(explicit_stability_interval(time.scheme)) +
                ", rho = " + digits17(thermal_spectral_radius(*this)) +
                " for this stencil and spacing; time.scheme \"imex\" treats u implicitly; "
                "docs/THEORY.md 10.1)");
        }
        if (dt > factor * phi_limit) {
            throw std::invalid_argument(
                what + " = " + digits17(dt) + " exceeds " +
                (factor < 1.0 ? "cfl_safety times " : "") + "the stability limit " +
                digits17(phi_limit) +
                " of the explicit phase-field update (a perturbation of the liquid grows every "
                "step above it, to within the tolerance of docs/THEORY.md 10.1)");
        }
    };
    if (time.adaptive) {
        check_step(time.dt, "time.dt (the first adaptive step)", time.cfl_safety);
        check_step(time.dt_min, "time.dt_min", time.cfl_safety);
    } else {
        check_step(time.dt, "time.dt", 1.0);
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

    // Boundary conditions. The parameters a BC type uses must be finite. A
    // Robin face alpha*u + beta*du/dn = gamma fills its ghost cell with
    // (gamma + (beta/h) u_inner) / (alpha + beta/h), h the spacing normal to
    // the face (apply_bc_face_kernel), so u_ghost = c u_inner + const with
    // c = (beta/h) / (alpha + beta/h). alpha and beta of opposite signs are
    // rejected: such a face feeds the field in proportion to its own value
    // (a negative heat-transfer coefficient), and c leaves [0, 1], where the
    // stability limit of the explicit diffusion and the convergence of the
    // IMEX Jacobi sweeps no longer hold (docs/THEORY.md 9.1, 10.1, 10.2).
    // With alpha*beta >= 0 and the scaling of scaled_robin, which the ghost
    // fill uses, the denominator never vanishes; the ghost constant
    // gamma/(alpha + beta/h) must be a finite double.
    const Real spacing[3] = {grid.dx, grid.dy, grid.dz};
    auto validate_bc = [&spacing](const BoundaryConfig& bc, const std::string& name, int first_axis,
                                  int last_axis) {
        auto require_finite = [&](Real v, const char* what) {
            if (!std::isfinite(v))
                throw std::invalid_argument("BC for " + name + ": " + what + " must be finite");
        };
        switch (bc.type) {
        case BCType::Dirichlet:
            require_finite(bc.value, "value");
            break;
        case BCType::Neumann:
            require_finite(bc.flux, "flux");
            break;
        case BCType::Periodic:
            break;
        case BCType::Robin:
            require_finite(bc.alpha, "alpha");
            require_finite(bc.beta, "beta");
            require_finite(bc.gamma, "gamma");
            if (bc.alpha == 0.0 && bc.beta == 0.0) {
                throw std::invalid_argument("Robin BC for " + name +
                                            ": alpha and beta cannot both be zero");
            }
            if ((bc.alpha < 0.0 && bc.beta > 0.0) || (bc.alpha > 0.0 && bc.beta < 0.0)) {
                throw std::invalid_argument(
                    "Robin BC for " + name +
                    ": alpha and beta must not have opposite signs "
                    "(the face would feed the field in proportion to its value; the explicit "
                    "stability limit and the IMEX Jacobi convergence assume alpha*beta >= 0, "
                    "docs/THEORY.md 9.1)");
            }
            // The ghost fill uses scaled_robin's coefficients. Their
            // denominator lies in (1/8, 1) in magnitude, so the scaled gamma
            // is finite whenever the quotient is, and the quotient is the one
            // of the unscaled coefficients: it is not finite exactly when
            // gamma/(alpha + beta/h) exceeds the double range.
            for (int axis = first_axis; axis <= last_axis; ++axis) {
                const RobinCoefficients r = scaled_robin(bc, spacing[axis]);
                if (!std::isfinite(r.gamma / (r.alpha + r.beta / spacing[axis]))) {
                    throw std::invalid_argument(
                        "Robin BC for " + name +
                        ": gamma/(alpha + beta/h) overflows at h = " + digits17(spacing[axis]));
                }
            }
            break;
        }
    };
    validate_bc(boundary.phi_bc, "phi", 0, 2);
    validate_bc(boundary.u_bc, "u", 0, 2);
    for (std::size_t i = 0; i < 6; ++i) {
        const int axis = static_cast<int>(i / 2);
        validate_bc(boundary.phi_faces.faces[i], "phi face " + std::to_string(i), axis, axis);
        validate_bc(boundary.u_faces.faces[i], "u face " + std::to_string(i), axis, axis);
    }

    // A periodic BC identifies the two faces of an axis: the ghost plane at
    // one face copies the interior plane next to the other face. It is
    // therefore a property of the axis, not of one face. With only one face of
    // an axis periodic, one side copies the far interior while the other does
    // not, so the discrete operators are no longer symmetric and the boundary
    // no longer conserves the field (docs/DESIGN.md, section 5.2). A uniform
    // BC is periodic on every face or on none, so only per-face BCs can pair
    // them wrongly.
    if (boundary.per_face) {
        static constexpr const char* axis_names[] = {"x", "y", "z"};
        const std::pair<const PerFaceBoundary*, const char*> fields[] = {
            {&boundary.phi_faces, "phi"}, {&boundary.u_faces, "u"}};
        for (const auto& [faces, field] : fields) {
            for (int axis = 0; axis < 3; ++axis) {
                const bool lo = faces->get(axis, 0).type == BCType::Periodic;
                const bool hi = faces->get(axis, 1).type == BCType::Periodic;
                if (lo == hi)
                    continue;
                const std::string axis_name = axis_names[axis];
                const std::string periodic = axis_name + (lo ? "_lo" : "_hi");
                const std::string other = axis_name + (lo ? "_hi" : "_lo");
                throw std::invalid_argument("boundary." + std::string(field) + ": " + periodic +
                                            " is periodic but " + other +
                                            " is not; a periodic BC must be set on both faces "
                                            "of an axis");
            }
        }
    }

    // GPU
    if (gpu.device_ids.empty())
        throw std::invalid_argument("At least one GPU device required");
    for (const int id : gpu.device_ids) {
        if (id < 0)
            throw std::invalid_argument("gpu.device_ids must be >= 0, got " + std::to_string(id));
    }

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
