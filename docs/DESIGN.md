# Design Document

## Mathematical Model

### Allen-Cahn Equation (Phase Field)

The phase-field variable phi evolves according to:

```
tau(n) * dphi/dt = div(W(n)^2 * grad(phi)) + div(|grad(phi)|^2 * W(n) * dW/d(grad(phi))) - dF/dphi
```

where:
- `tau(n) = tau0 * A(n)^2` is the orientation-dependent relaxation time
- `W(n) = W0 * A(n)` is the orientation-dependent interface width
- `A(n) = (1 - 3*eps) * (1 + 4*eps/(1-3*eps) * (nx^4 + ny^4 + nz^4)/(nx^2 + ny^2 + nz^2)^2)` is the cubic anisotropy function with 4-fold symmetry, where `n = grad(phi)/|grad(phi)|`
- `dF/dphi = -phi*(1-phi^2) + lambda*u*(1-phi^2)^2` is the double-well + coupling term

### Thermal Diffusion

The dimensionless temperature u evolves according to:

```
du/dt = D * Laplacian(u) + 0.5 * dphi/dt
```

The `0.5 * dphi/dt` term represents latent heat release at the solidification front.

### Derived Parameters

```
lambda = W0 * a1 / d0       (coupling strength)
tau0 = (W0^3 * a1 * a2) / (d0 * D) + (W0^2 * beta0) / d0
```

where `a1 = 1.25/sqrt(2)`, `a2 = 0.64`, `d0` is capillary length.

## Spatial Discretization

### 7-Point Standard Laplacian

Second-order accurate, uses 6 face neighbors:

```
Lap(phi) = (phi[x+1] + phi[x-1] - 2*phi[x]) / dx^2
         + (phi[y+1] + phi[y-1] - 2*phi[y]) / dy^2
         + (phi[z+1] + phi[z-1] - 2*phi[z]) / dz^2
```

Stencil weights (assuming dx = dy = dz = h):

```mermaid
graph LR
    subgraph "7-Point Stencil (face neighbors)"
        A["1/h^2"] --- C["center: -6/h^2"] --- B["1/h^2"]
    end
```

### 27-Point Isotropic Laplacian (Kumar 2004)

Fourth-order isotropic, uses all 26 neighbors:

```
Lap(phi) = (4 * sum_face + 2 * sum_edge + 1 * sum_corner - 56 * phi_center) / (26 * h^2)
```

| Neighbor Type | Count | Weight (per neighbor) | Total Weight |
|---------------|-------|-----------------------|-------------|
| Face | 6 | 4/26 | 24/26 |
| Edge | 12 | 2/26 | 24/26 |
| Corner | 8 | 1/26 | 8/26 |
| Center | 1 | -56/26 | -56/26 |

Verification: 24 + 24 + 8 - 56 = 0 (consistent discrete Laplacian).

### Gradient Stencils

**2nd-order central differences** (used in production kernels):

```
d(phi)/dx = (phi[x+1] - phi[x-1]) / (2*dx)
```

**4th-order central differences** (available but not used in fused kernel):

```
d(phi)/dx = (-phi[x+2] + 8*phi[x+1] - 8*phi[x-1] + phi[x-2]) / (12*dx)
```

## Time Integration Schemes

### Euler (1st order)

```mermaid
flowchart TB
    A["phi_new = phi_old + dt * RHS_phi(phi_old, u_old)"] --> B["u_new = u_old + 0.5*(phi_new - phi_old) + dt*D*Lap(u_old)"]
    B --> C["Apply BCs"]
    C --> D["swap(old, new)"]
```

Single function evaluation per step. Stability requires `dt < h^2/(2*D*3)` (CFL condition).

### Heun / RK2 (2nd order)

```mermaid
flowchart TB
    A["Predictor: phi_tmp = phi_old + dt*f(phi_old)"] --> B["Corrector: phi_new = phi_tmp + dt*f(phi_tmp)"]
    B --> C["Average: phi_result = 0.5*(phi_old + phi_new)"]
    C --> D["Apply BCs, swap"]
```

Two function evaluations per step. Local truncation error O(dt^3).

### Classical RK4 (4th order)

```mermaid
flowchart TB
    K1["k1 = f(t_n, y_n)"] --> K2["k2 = f(t_n + dt/2, y_n + dt/2 * k1)"]
    K2 --> K3["k3 = f(t_n + dt/2, y_n + dt/2 * k2)"]
    K3 --> K4["k4 = f(t_n + dt, y_n + dt * k3)"]
    K4 --> Combine["y_{n+1} = y_n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)"]
    Combine --> Latent["u_new += 0.5*(phi_new - phi_old)"]
    Latent --> BC["Apply BCs, swap"]
```

Four function evaluations per step. Uses separate `compute_force_kernel` + `allen_cahn_rhs_kernel` + `thermal_rhs_kernel` per stage (not fused, requires Fx, Fy, Fz arrays).

Latent heat is added as a post-processing step because `thermal_rhs_kernel` computes only `D*Lap(u)` for use with the standard RK4 combination formula.

### IMEX (Mixed order)

```mermaid
flowchart TB
    E["Explicit: Allen-Cahn fused kernel"] --> RHS["RHS = u_old + 0.5*(phi_new - phi_old)"]
    RHS --> J["Jacobi solve: (I - dt*D*Lap) u_new = RHS"]
    J --> BC["Apply BCs, swap"]
```

Implicit treatment of thermal diffusion allows larger time steps. The Jacobi iteration solves:

```
u_new[c] = (RHS[c] + dt*D * sum_neighbors/h^2) / (1 + dt*D * 6/h^2)
```

Uses 50 iterations. Convergence guaranteed when spectral radius `rho = 1 - pi^2 * dt * D / L^2 < 1`.

## Kernel Fusion Optimization

### Original Design (13.6 GB for 600^3)

```mermaid
flowchart LR
    subgraph "Kernel 1: computeGradients"
        G["Read phi_old<br/>Write gradients"]
    end
    subgraph "Kernel 2: computeForce"
        F["Read gradients<br/>Write Fx, Fy, Fz"]
    end
    subgraph "Kernel 3: allenCahn"
        AC["Read phi, u, Fx, Fy, Fz<br/>Write phi_new"]
    end
    G --> F --> AC
```

**Memory cost:** phi_old + phi_new + u_old + u_new + Fx + Fy + Fz + IDx + IDy + IDz = 10 arrays

### Fused Design (6.4 GB for 600^3)

```mermaid
flowchart LR
    subgraph "Single Fused Kernel"
        FK["1. Compute gradients at (x,y,z)<br/>2. Compute A(n) ONCE<br/>3. Compute force at (x,y,z)<br/>4. Recompute force at 6 neighbors<br/>5. Compute divergence<br/>6. Update phi_new"]
    end
```

**Trade-off:** Recomputes anisotropy at 6 neighbor points (7x total including center) instead of reading from separate force arrays. This is compute-bound rather than memory-bound, which favors modern GPU architectures.

**Savings:** Eliminates Fx, Fy, Fz (3 arrays) + IDx, IDy, IDz (3 arrays) = 7.2 GB for 600^3.

## GPU Memory Management

### DeviceField RAII Pattern

```mermaid
classDiagram
    class DeviceField~T~ {
        -ptr_ : T*
        -count_ : size_t
        +DeviceField(count)  cudaMalloc
        +~DeviceField()  cudaFree
        +DeviceField(DeviceField&&)  move
        +operator=(DeviceField&&)  move
        +copy_from_host(src, stream)
        +copy_to_host(dst, stream)
        +zero_async(stream)
        +data() : T*
        +bytes() : size_t
        +friend swap(a, b)  O(1)
    }
    note for DeviceField "Copy ctor/assignment: DELETED\nAll GPU memory is RAII-managed\nswap() exchanges pointers only"
```

### Pointer Swap Strategy

Instead of a GPU kernel copying `N^3` doubles:

```cpp
// OLD: O(N) GPU kernel
swap_kernel<<<grid, block>>>(phi_old, phi_new, N);

// NEW: O(1) host-side pointer swap
swap(phi_old_, phi_new_);  // std::swap on pointers
```

## Parallel Reduction Algorithm

Used for `compute_max_dphi()` (adaptive time stepping).

```mermaid
graph TB
    subgraph "Block Reduction (256 threads)"
        L0["Thread loads |a[i] - b[i]| into shared memory"]
        L1["128: sdata[tid] = max(sdata[tid], sdata[tid+128])"]
        L2["64: sdata[tid] = max(sdata[tid], sdata[tid+64])"]
        L3["Warp reduction (32..1): volatile sdata, no __syncthreads"]
        L4["Thread 0 writes block result to global"]
    end

    subgraph "Grid Reduction"
        G["Second kernel: reduce block results to single value"]
    end

    L0 --> L1 --> L2 --> L3 --> L4 --> G
```

**Complexity:** O(N/P) work per thread, O(log P) steps per block, where P = block size.

## Data Structure Design

### LRU Cache

```mermaid
classDiagram
    class LRUCache~Key, Value~ {
        -capacity_ : size_t
        -order_ : list~pair~Key,Value~~
        -map_ : unordered_map~Key, list_iterator~
        -mutex_ : shared_mutex
        -hit_count_, miss_count_ : size_t
        +put(key, value)  O(1)
        +get(key) : optional~Value~  O(1)
        +contains(key) : bool  O(1)
        +erase(key) : bool  O(1)
        +hit_rate() : double
    }
```

| Operation | Time | Lock Type |
|-----------|------|-----------|
| `put()` | O(1) amortized | Exclusive |
| `get()` | O(1) | Exclusive (promotes) |
| `contains()` | O(1) | Shared (no promote) |
| `erase()` | O(1) | Exclusive |

**Implementation:** Doubly-linked list maintains LRU ordering (front = MRU, back = LRU). Hash map provides O(1) lookup by key to list iterator. On access, the entry is spliced to the front. On eviction, the back entry is removed.

### Spatial Hash Map

```mermaid
classDiagram
    class SpatialHash~Value~ {
        -cell_size_ : Real
        -inv_cell_size_ : Real
        -buckets_ : unordered_map~CellCoord, vector~Value~~
        +insert(x, y, z, value)  O(1)
        +query_cell(x, y, z) : vector~Value~*  O(1)
        +query_radius(x, y, z, r) : vector~Value~  O(r^3)
        +build_from_field(accessor, predicate)  O(N)
    }

    class CellCoord {
        +cx, cy, cz : int
    }

    class CellCoordHash {
        +operator()(CellCoord) : size_t
    }

    SpatialHash --> CellCoord
    SpatialHash --> CellCoordHash
```

**Hash function:** FNV-1a mixing for 3D integer coordinates:

```
h = 14695981039346656037
h = (h XOR cx) * 1099511628211
h = (h XOR cy) * 1099511628211
h = (h XOR cz) * 1099511628211
```

### Concurrent Hash Map

```mermaid
classDiagram
    class ConcurrentMap~Key, Value, NumStripes~ {
        -stripes_ : array~Stripe, NumStripes~
        +put(key, value)  O(1)
        +get(key) : optional~Value~  O(1)
        +update(key, fn)  O(1)
        +get_or_insert(key, default) : Value  O(1)
        +snapshot() : vector~pair~  O(N)
    }

    class Stripe {
        -mutex : shared_mutex
        -map : unordered_map~Key, Value~
    }

    ConcurrentMap *-- Stripe : "16 stripes"
```

**Design:** 16 independent stripes, each with its own `shared_mutex` and `unordered_map`. Key is hashed and modulo-distributed across stripes. This reduces lock contention by 16x compared to a single-mutex approach.

| Operation | Time | Lock Type |
|-----------|------|-----------|
| `put()` | O(1) | Exclusive (one stripe) |
| `get()` | O(1) | Shared (one stripe) |
| `update()` | O(1) | Exclusive (one stripe) |
| `size()` | O(stripes) | Shared (all stripes) |
| `snapshot()` | O(N) | Shared (all stripes) |

## Boundary Condition Implementation

### Face Enumeration

```
Face Index:  0     1     2     3     4     5
Face Name:  X_Lo  X_Hi  Y_Lo  Y_Hi  Z_Lo  Z_Hi
Axis:        0     0     1     1     2     2
Side:        0     1     0     1     0     1
```

### Robin BC Derivation

Robin condition: `alpha * u + beta * du/dn = gamma`

Using one-sided finite difference for du/dn at the boundary:
```
du/dn ~ (u_bnd - u_inner) / (sign * ds)
```

Substituting:
```
alpha * u_bnd + beta * (u_bnd - u_inner) / (sign * ds) = gamma
u_bnd * (alpha + beta/(sign*ds)) = gamma + beta * u_inner / (sign*ds)
u_bnd = (gamma + beta*u_inner/(sign*ds)) / (alpha + beta/(sign*ds))
```

Division-by-zero protection: falls back to `u_inner` when denominator < 1e-30.

## Performance Optimization Summary

| Optimization | Technique | Impact |
|-------------|-----------|--------|
| Index elimination | Compute from threadIdx | -2.4 GB (600^3) |
| Kernel fusion | Recompute force at neighbors | -4.8 GB (600^3) |
| Pointer swap | std::swap on host pointers | O(N) -> O(1) per step |
| Anisotropy dedup | Compute A(n) once per thread (was 8x) | ~7x fewer transcendentals |
| Warp reduction | Unrolled last warp, no __syncthreads | 2x faster reduction |
| Async I/O | Background writer thread | Zero simulation blocking |
| LRU stats cache | Hash map + linked list | O(1) cached field statistics |
| Striped concurrent map | 16 independent mutexes | 16x less lock contention |
| Spatial hashing | FNV-1a 3D bucketing | O(1) region queries |
| RAII GPU memory | DeviceField destructor | Zero memory leaks |
