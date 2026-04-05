# Allen-Cahn CUDA v2.0 — Phase-Field Simulation of Dendritic Solidification

Production-grade GPU-accelerated simulation of dendritic crystal growth using the Allen-Cahn phase-field model coupled with thermal diffusion.

## Features

- **Modern C++23 / CUDA 20** with full RAII, no raw pointers
- **Multiple time integrators**: Euler, Heun (RK2), RK4, IMEX (implicit diffusion)
- **Adaptive time stepping** with CFL safety
- **Configurable stencils**: 7-point (2nd order) and 27-point isotropic (Kumar 2004)
- **Boundary conditions**: Dirichlet, Neumann, Periodic, Robin
- **Async VTK output** via background thread (never blocks simulation)
- **Checkpoint/restart** with rolling retention policy
- **Multi-GPU support** via domain decomposition
- **JSON configuration** — no recompilation needed to change parameters
- **Comprehensive tests** — unit, integration, and convergence tests

## Dependencies

| Dependency | Version | Required |
|-----------|---------|----------|
| CMake | >= 3.28 | Yes |
| CUDA Toolkit | >= 12.0 | Yes |
| GCC / Clang | C++23 support | Yes |
| nlohmann/json | >= 3.11 | Auto-fetched |
| spdlog | >= 1.12 | Auto-fetched |
| GoogleTest | >= 1.14 | Auto-fetched |
| VTK | >= 9.0 | Optional |
| HDF5 | any | Optional |

## Build

```bash
git clone git@github.com:myousefi2016/Allen-Cahn-CUDA.git
cd Allen-Cahn-CUDA
cmake --preset release
cmake --build build/release
```

For debug builds with tests:
```bash
cmake --preset debug
cmake --build build/debug
ctest --test-dir build/debug --output-on-failure
```

## Run

```bash
mkdir -p out
./build/release/src/allen-cahn-cuda config/default.json
```

With a smaller grid for testing:
```bash
./build/release/src/allen-cahn-cuda config/benchmark_small.json
```

## Configuration

All parameters are specified in JSON. See `config/default.json` for the full reference. Key sections:

```json
{
    "physics": { "delta": 0.8, "epsilon": 0.07, "W0": 1.0, "D": 2.0 },
    "grid":    { "Nx": 600, "Ny": 600, "Nz": 600, "dx": 0.4 },
    "time":    { "dt": 0.01, "max_steps": 6000, "scheme": "euler" },
    "stencil": "7pt",
    "output":  { "frequency": 100, "format": "vts" }
}
```

Available time schemes: `"euler"`, `"heun"`, `"rk4"`, `"imex"`
Available stencils: `"7pt"`, `"27pt"`
Available BCs: `"dirichlet"`, `"neumann"`, `"periodic"`, `"robin"`

## Memory Budget (600^3 grid)

| Component | Old Code | v2.0 |
|-----------|----------|------|
| Fields (phi, u) × 2 | 6.4 GB | 6.4 GB |
| Force arrays (Fx, Fy, Fz) | 4.8 GB | **0 GB** (fused) |
| Index arrays (IDx, IDy, IDz) | 2.4 GB | **0 GB** (computed) |
| **Total** | **13.6 GB** | **6.4 GB** |

## References

1. Y.-T. Kim, N. Provatas, N. Goldenfeld, J. Dantzig, *Universal dynamics of phase-field models for dendritic growth*, Phys. Rev. E 59, R2546 (1999).
2. A. Kumar, *Isotropic finite-differences*, J. Comput. Phys. 201(1), 109-118 (2004).

## License

See [LICENSE](LICENSE).

![Dendritic Growth](https://raw.githubusercontent.com/myousefi2016/Allen-Cahn-CUDA/master/result/img.png)
