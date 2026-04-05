# Architecture Guide

## System Overview

Allen-Cahn CUDA is a GPU-accelerated 3D phase-field simulation implementing the coupled Allen-Cahn and thermal diffusion equations for dendritic solidification. The system is organized into layered libraries with clear separation of concerns.

## Layer Architecture

```mermaid
graph TB
    subgraph "Application"
        Main["main.cpp"]
    end

    subgraph "Engine"
        SE["SimulationEngine"]
    end

    subgraph "Solver"
        IS["ISolver (interface)"]
        CS["CudaSolver"]
        MG["MultiGPUSolver"]
    end

    subgraph "Kernels"
        ACK["AllenCahnKernels"]
        TK["ThermalKernels"]
        BK["BoundaryKernels"]
        RK["ReductionKernels"]
    end

    subgraph "Infrastructure"
        DF["DeviceField (RAII GPU memory)"]
        CU["CudaUtils (Stream, Event)"]
    end

    subgraph "I/O"
        VW["VTKWriter (async)"]
        CP["CheckpointIO"]
    end

    subgraph "Core"
        SC["SimulationConfig"]
        GR["Grid"]
        FD["FieldData"]
    end

    Main --> SE
    SE --> IS
    IS --> CS
    IS --> MG
    MG --> CS
    CS --> ACK & TK & BK & RK
    CS --> DF & CU
    SE --> VW & CP
    VW --> FD
    CP --> FD
    CS --> SC & GR
```

## Module Dependency Graph

```mermaid
graph LR
    ac_logging["ac_logging<br/>(spdlog wrapper)"]
    ac_core["ac_core<br/>(Grid, Config, FieldData,<br/>CheckpointManager,<br/>LRUCache, SpatialHash,<br/>ConcurrentMap)"]
    ac_cuda["ac_cuda<br/>(ISolver, CudaSolver,<br/>MultiGPUSolver, Kernels,<br/>DeviceField, CudaUtils)"]
    ac_io["ac_io<br/>(VTKWriter, CheckpointIO)"]
    ac_engine["ac_engine<br/>(SimulationEngine)"]
    main["allen-cahn-cuda<br/>(executable)"]

    ac_core --> ac_logging
    ac_core --> nlohmann_json
    ac_cuda --> ac_core
    ac_cuda --> CUDA
    ac_io --> ac_core
    ac_io --> VTK9["VTK 9 (optional)"]
    ac_engine --> ac_core & ac_cuda & ac_io & ac_logging
    main --> ac_engine

    style ac_cuda fill:#e1f5fe
    style CUDA fill:#76ff03
```

## Simulation Lifecycle

```mermaid
sequenceDiagram
    participant Main
    participant Config as SimulationConfig
    participant Engine as SimulationEngine
    participant Solver as CudaSolver/MultiGPU
    participant VTK as VTKWriter (bg thread)
    participant Ckpt as CheckpointManager

    Main->>Config: from_json(path)
    Config->>Config: validate()
    Main->>Engine: SimulationEngine(config)
    Engine->>Solver: create_solver()
    Engine->>VTK: VTKWriter(grid, params)

    alt Restart from checkpoint
        Engine->>Ckpt: restore()
        Ckpt-->>Engine: phi, u, step, time
    else Fresh start
        Engine->>Engine: initialize_fields()
    end

    Engine->>Solver: initialize(phi, u)
    Note over Solver: cudaMemcpy H2D

    loop Time Loop (step 1..max_steps)
        Engine->>Solver: step(dt)
        Note over Solver: GPU kernel launches
        alt Output step
            Engine->>Solver: copy_phi_to_host()
            Engine->>Solver: copy_u_to_host()
            Engine->>VTK: write_async(step, time, phi, u)
            Note over VTK: Background thread<br/>writes to disk
        end
        alt Checkpoint step
            Engine->>Ckpt: save(step, time, dt, phi, u)
        end
        alt Adaptive dt
            Engine->>Solver: compute_max_dphi()
            Note over Solver: Parallel reduction
            Engine->>Engine: adapt_time_step(dt)
        end
    end

    Engine->>VTK: flush()
    Engine->>Solver: synchronize()
```

## Solver Class Hierarchy

```mermaid
classDiagram
    class ISolver {
        <<interface>>
        +initialize(phi, u)*
        +step(dt)*
        +compute_max_dphi()* double
        +copy_phi_to_host(out)*
        +copy_u_to_host(out)*
        +apply_boundary_conditions()*
        +synchronize()*
        +stream()* cudaStream_t
    }

    class CudaSolver {
        -phi_old_, phi_new_ : DeviceField
        -u_old_, u_new_ : DeviceField
        -compute_stream_ : Stream
        -scheme_ : TimeScheme
        +step_euler(dt)
        +step_heun(dt)
        +step_rk4(dt)
        +step_imex(dt)
        +phi_data() : double*
        +u_data() : double*
        -apply_bc(field, bc)
        -apply_bc_per_face(field, face_bcs)
    }

    class MultiGPUSolver {
        -domains_ : vector~GPUDomain~
        -halo_width_ : int
        +exchange_halos()
        -copy_slab(dst, src, ...)
        -extract_subdomain(global, local, domain)
    }

    class GPUDomain {
        +device_id : int
        +x_start, x_end : int
        +local_Nx : int
        +halo : int
        +solver : unique_ptr~CudaSolver~
        +halo_stream : Stream
    }

    ISolver <|-- CudaSolver
    ISolver <|-- MultiGPUSolver
    MultiGPUSolver *-- GPUDomain
    GPUDomain *-- CudaSolver
```

## GPU Memory Layout

```mermaid
graph TB
    subgraph "Device Memory (per GPU)"
        subgraph "Primary Fields (always allocated)"
            phi_old["phi_old_ : DeviceField&lt;double&gt;<br/>N = Nx * Ny * Nz"]
            phi_new["phi_new_ : DeviceField&lt;double&gt;"]
            u_old["u_old_ : DeviceField&lt;double&gt;"]
            u_new["u_new_ : DeviceField&lt;double&gt;"]
        end

        subgraph "Heun/RK4/IMEX temporaries"
            phi_tmp["phi_tmp_ : DeviceField&lt;double&gt;"]
            u_tmp["u_tmp_ : DeviceField&lt;double&gt;"]
        end

        subgraph "RK4 only"
            k1["k1_phi_, k1_u_"]
            k2["k2_phi_, k2_u_"]
            k3["k3_phi_, k3_u_"]
            k4["k4_phi_, k4_u_"]
            F["Fx_, Fy_, Fz_"]
        end

        red["d_reduction_result_ (1 element)"]
    end

    subgraph "Pointer Swap (O(1))"
        swap["swap(phi_old_, phi_new_)<br/>std::swap of pointers<br/>No GPU data movement"]
    end
```

## Multi-GPU Halo Exchange

```mermaid
sequenceDiagram
    participant GPU0 as GPU 0 (Domain 0)
    participant GPU1 as GPU 1 (Domain 1)

    Note over GPU0,GPU1: Before each time step

    par Halo Exchange
        GPU0->>GPU1: Right boundary slabs (halo_width=2)<br/>cudaMemcpyPeerAsync for phi and u
        GPU1->>GPU0: Left boundary slabs (halo_width=2)<br/>cudaMemcpyPeerAsync for phi and u
    end

    Note over GPU0,GPU1: Synchronize halo streams

    par Compute
        GPU0->>GPU0: solver->step(dt)
        GPU1->>GPU1: solver->step(dt)
    end
```

## Async I/O Pipeline

```mermaid
sequenceDiagram
    participant Sim as Simulation Thread
    participant Queue as Job Queue (mutex + cv)
    participant BG as VTK Writer Thread
    participant Cache as LRU Stats Cache
    participant Disk as Filesystem

    Sim->>Queue: write_async(step, time, phi_copy, u_copy)
    Note over Sim: Returns immediately

    Sim->>Sim: Continue simulation...

    BG->>Queue: wait for job
    Queue-->>BG: job
    BG->>Cache: compute_statistics(phi_data)
    Cache-->>BG: FieldStatistics (cached or computed)
    BG->>Disk: write VTK/raw file

    Sim->>Queue: flush()
    Note over Sim: Blocks until queue empty
```

## Per-Face Boundary Condition System

```mermaid
graph LR
    subgraph "3D Domain"
        XLo["X- face (0)"]
        XHi["X+ face (1)"]
        YLo["Y- face (2)"]
        YHi["Y+ face (3)"]
        ZLo["Z- face (4)"]
        ZHi["Z+ face (5)"]
    end

    subgraph "PerFaceBoundary"
        arr["faces[6] : BoundaryConfig"]
    end

    subgraph "BoundaryConfig"
        bc["type : BCType<br/>value : Real<br/>flux : Real<br/>alpha, beta, gamma : Real"]
    end

    XLo --> arr
    XHi --> arr
    YLo --> arr
    YHi --> arr
    ZLo --> arr
    ZHi --> arr
    arr --> bc
```

Each face is applied via a 2D kernel launch covering the face's surface. The kernel receives the face-specific BC parameters and applies Dirichlet, Neumann, Periodic, or Robin conditions.

## Thread Safety Model

| Component | Synchronization | Access Pattern |
|-----------|----------------|----------------|
| `LRUCache` | `std::shared_mutex` | Shared reads, exclusive writes |
| `ConcurrentMap` | 16 `std::shared_mutex` stripes | Distributed lock contention |
| `SpatialHash` | None (build-then-query) | Single-threaded build, read-only queries |
| `VTKWriter` | `std::mutex` + `condition_variable` | Producer-consumer queue |
| `DeviceField` | None (single-stream) | One CUDA stream per solver |
| `MultiGPUSolver` | Per-GPU streams, sync barriers | Parallel compute, synchronized halo exchange |

## Build System

```mermaid
graph TB
    subgraph "CMake Targets"
        ac_logging["ac_logging (STATIC)"]
        ac_core["ac_core (STATIC)"]
        ac_cuda["ac_cuda (STATIC, CUDA)"]
        ac_io["ac_io (STATIC)"]
        ac_engine["ac_engine (STATIC)"]
        exe["allen-cahn-cuda (EXE)"]
        unit["unit_tests (EXE)"]
        integ["integration_tests (EXE)"]
    end

    subgraph "External (FetchContent)"
        json["nlohmann_json"]
        spdlog["spdlog"]
        gtest["GoogleTest"]
    end

    subgraph "System (find_package)"
        vtk["VTK 9 (optional)"]
        hdf5["HDF5 (optional)"]
        cuda["CUDA Toolkit"]
    end

    ac_core --> ac_logging & json
    ac_cuda --> ac_core & cuda
    ac_io --> ac_core
    ac_io -.->|optional| vtk & hdf5
    ac_engine --> ac_core & ac_cuda & ac_io & ac_logging
    exe --> ac_engine
    unit --> gtest & ac_core & ac_cuda & ac_io
    integ --> gtest & ac_engine
```
