# Architecture

## Module Dependency Graph

```
main.cpp
  └── SimulationEngine (ac_engine)
        ├── SimulationConfig (ac_core)
        ├── Grid (ac_core)
        ├── FieldData (ac_core)
        ├── CheckpointManager (ac_core)
        │     └── CheckpointIO (ac_io)
        ├── VTKWriter (ac_io)
        ├── CudaSolver (ac_cuda)
        │     ├── DeviceField<T>
        │     ├── CudaUtils (Stream, Event)
        │     ├── Kernels
        │     │     ├── AllenCahnKernels (fused phase-field + anisotropy)
        │     │     ├── ThermalKernels
        │     │     ├── BoundaryKernels (Dirichlet, Neumann, Periodic, Robin)
        │     │     └── ReductionKernels (for adaptive dt)
        │     └── KernelParams
        └── Logger (ac_logging)
              └── spdlog
```

## Libraries

| Library | Purpose | Dependencies |
|---------|---------|-------------|
| `ac_logging` | spdlog wrapper | spdlog |
| `ac_core` | Config, Grid, FieldData, CheckpointManager | nlohmann/json, ac_logging |
| `ac_cuda` | GPU solver, kernels, RAII wrappers | CUDA, ac_core |
| `ac_io` | VTK writer, checkpoint I/O | VTK 9 (optional), ac_core |
| `ac_engine` | Simulation orchestrator | ac_core, ac_cuda, ac_io, ac_logging |

## Key Design Decisions

1. **RAII everywhere**: `DeviceField<T>` wraps `cudaMalloc`/`cudaFree`. `Stream` and `Event` wrap CUDA objects. No raw `new`/`delete` or manual memory management.

2. **Pointer swap instead of data copy**: Old/new fields are swapped via `std::swap` on the host side (O(1)) rather than a GPU kernel copying N^3 elements.

3. **Index arrays eliminated**: Thread-to-grid mapping computed arithmetically inside each kernel, saving ~2.4 GB GPU memory for 600^3 grids.

4. **Kernel fusion**: `calculateForce` + `allenCahn` fused into one kernel, eliminating 3 intermediate force arrays (~5 GB savings).

5. **Configurable stencils**: 7-point standard and 27-point isotropic Laplacian (Kumar 2004).

6. **Multiple time integration schemes**: Euler, Heun (RK2), RK4, IMEX via strategy pattern.

7. **Async I/O**: Background thread for VTK file writing; simulation never blocks on disk.

8. **JSON configuration**: All parameters externalized to JSON files, no hardcoded `#define` constants.
