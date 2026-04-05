# Physics Documentation

## Governing Equations

This code solves the coupled Allen-Cahn phase-field model for dendritic solidification, following the formulation of Kim, Provatas, Goldenfeld, and Dantzig (1999).

### Phase-Field Equation (Allen-Cahn)

```
τ(n) ∂φ/∂t = ∇·(W(n)² ∇φ) + ∂/∂x_i (|∇φ|² W(n) ∂W(n)/∂(∂φ/∂x_i)) - f'(φ) + λu g'(φ)
```

Where:
- `φ` is the phase-field order parameter: φ = 1 (solid), φ = -1 (liquid)
- `W(n)` = W₀ · A(n) is the anisotropic interface width
- `τ(n)` = τ₀ · A(n)² is the anisotropic relaxation time
- `f(φ) = -φ²/2 + φ⁴/4` is the double-well potential
- `g(φ) = φ - 2φ³/3 + φ⁵/5` is the coupling function

### Thermal Diffusion Equation

```
∂u/∂t = D ∇²u + (1/2) ∂φ/∂t
```

Where:
- `u` is the dimensionless temperature (supersaturation)
- `D` is the thermal diffusivity
- The `(1/2) ∂φ/∂t` term represents latent heat release during solidification

### Anisotropy Function

Cubic crystallographic anisotropy (4-fold symmetry):

```
A(n) = (1 - 3ε)(1 + (4ε/(1-3ε)) · (φ_x⁴ + φ_y⁴ + φ_z⁴)/(φ_x² + φ_y² + φ_z²)²)
```

Where ε is the anisotropy strength (must be in [0, 1/3)).

Special cases:
- Along crystal axes (e.g., n = (1,0,0)): A = 1 + ε
- Along diagonals (e.g., n = (1,1,1)/√3): A = 1 - 5ε/3

## Numerical Methods

### Spatial Discretization

**7-point stencil** (standard, 2nd order):
```
∇²φ ≈ (φ_{i+1} + φ_{i-1} - 2φ_i)/dx² + (y terms) + (z terms)
```

**27-point isotropic stencil** (Kumar 2004, 4th order isotropic):
```
∇²φ ≈ (4·Σ_face + 2·Σ_edge + 1·Σ_corner - 56·φ_center) / (26·h²)
```

### Time Integration

| Scheme | Order | Stages | Stability | Memory |
|--------|-------|--------|-----------|--------|
| Euler | 1 | 1 | Conditional | Baseline |
| Heun (RK2) | 2 | 2 | Conditional | +1 field |
| RK4 | 4 | 4 | Conditional | +4 fields |
| IMEX | 1-2 | 1-2 | Unconditional (diffusion) | +1 field |

### CFL Condition

For the explicit thermal diffusion:
```
dt < (dx²) / (2·D·3) × safety_factor
```

### Adaptive Time Stepping

Based on max|Δφ| per step:
```
dt_new = dt_old × min(1.5, max(0.5, 0.9 × target/max_dphi))
```
Clamped to [dt_min, dt_max] and the CFL limit.

## References

1. Y.-T. Kim, N. Provatas, N. Goldenfeld, J. Dantzig, "Universal dynamics of phase-field models for dendritic growth," Phys. Rev. E 59, R2546 (1999).

2. A. Kumar, "Isotropic finite-differences," J. Comput. Phys. 201(1), 109-118 (2004).
