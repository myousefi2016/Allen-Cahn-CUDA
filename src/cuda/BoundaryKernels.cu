#include "cuda/Kernels.cuh"
#include "cuda/CudaUtils.cuh"

namespace ac::cuda {

/// Apply boundary conditions on a single face of the 3D domain.
/// face_axis: 0=X, 1=Y, 2=Z
/// face_side: 0=lo, 1=hi
__global__ void __launch_bounds__(256)
apply_bc_face_kernel(
    double* __restrict__ field,
    int Nx, int Ny, int Nz,
    double dx, double dy, double dz,
    int face_axis, int face_side,
    int bc_type, // 0=Dirichlet, 1=Neumann, 2=Periodic, 3=Robin
    double bc_value, double bc_flux,
    double bc_alpha, double bc_beta, double bc_gamma)
{
    // This kernel is launched as a 2D grid covering two of the three axes.
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    // Determine which axes i,j correspond to and the fixed coordinate
    int dim1_max, dim2_max;
    if (face_axis == 0) { dim1_max = Ny; dim2_max = Nz; }       // YZ face
    else if (face_axis == 1) { dim1_max = Nx; dim2_max = Nz; }   // XZ face
    else { dim1_max = Nx; dim2_max = Ny; }                        // XY face

    if (i >= dim1_max || j >= dim2_max) return;

    // Compute 3D indices
    int x, y, z;
    int fixed_val = (face_side == 0) ? 0 :
        ((face_axis == 0) ? Nx - 1 : (face_axis == 1) ? Ny - 1 : Nz - 1);

    if (face_axis == 0) { x = fixed_val; y = i; z = j; }
    else if (face_axis == 1) { x = i; y = fixed_val; z = j; }
    else { x = i; y = j; z = fixed_val; }

    int c = idx3d(x, y, z, Ny, Nz);

    switch (bc_type) {
    case 0: // Dirichlet
        field[c] = bc_value;
        break;
    case 1: { // Neumann (zero-gradient + prescribed flux)
        int nx, ny, nz;
        double ds;
        if (face_axis == 0) {
            nx = (face_side == 0) ? 1 : Nx - 2; ny = y; nz = z;
            ds = dx;
        } else if (face_axis == 1) {
            nx = x; ny = (face_side == 0) ? 1 : Ny - 2; nz = z;
            ds = dy;
        } else {
            nx = x; ny = y; nz = (face_side == 0) ? 1 : Nz - 2;
            ds = dz;
        }
        double sign = (face_side == 0) ? -1.0 : 1.0;
        field[c] = field[idx3d(nx, ny, nz, Ny, Nz)] + sign * bc_flux * ds;
        break;
    }
    case 2: { // Periodic
        int px, py, pz;
        if (face_axis == 0) {
            px = (face_side == 0) ? Nx - 2 : 1; py = y; pz = z;
        } else if (face_axis == 1) {
            px = x; py = (face_side == 0) ? Ny - 2 : 1; pz = z;
        } else {
            px = x; py = y; pz = (face_side == 0) ? Nz - 2 : 1;
        }
        field[c] = field[idx3d(px, py, pz, Ny, Nz)];
        break;
    }
    case 3: { // Robin: alpha*u + beta*du/dn = gamma
        // Approximate du/dn with one-sided difference
        int nx, ny, nz;
        double ds;
        if (face_axis == 0) {
            nx = (face_side == 0) ? 1 : Nx - 2; ny = y; nz = z;
            ds = dx;
        } else if (face_axis == 1) {
            nx = x; ny = (face_side == 0) ? 1 : Ny - 2; nz = z;
            ds = dy;
        } else {
            nx = x; ny = y; nz = (face_side == 0) ? 1 : Nz - 2;
            ds = dz;
        }
        double sign = (face_side == 0) ? -1.0 : 1.0;
        double u_inner = field[idx3d(nx, ny, nz, Ny, Nz)];
        // Robin: alpha*u_bnd + beta*(u_bnd - u_inner)/(sign*ds) = gamma
        // => u_bnd * (alpha + beta/(sign*ds)) = gamma + beta*u_inner/(sign*ds)
        double denom = bc_alpha + bc_beta / (sign * ds);
        if (fabs(denom) > 1e-30) {
            field[c] = (bc_gamma + bc_beta * u_inner / (sign * ds)) / denom;
        } else {
            field[c] = u_inner;
        }
        break;
    }
    default:
        field[c] = bc_value;
        break;
    }
}

// ── Launch wrapper ─────────────────────────────────────────────────────────

static void launch_bc_face(
    double* field, const KernelParams& params,
    int axis, int side,
    const BoundaryConfig& bc,
    cudaStream_t stream)
{
    int dim1, dim2;
    if (axis == 0) { dim1 = params.Ny; dim2 = params.Nz; }
    else if (axis == 1) { dim1 = params.Nx; dim2 = params.Nz; }
    else { dim1 = params.Nx; dim2 = params.Ny; }

    dim3 block(16, 16);
    dim3 grid(
        (static_cast<unsigned>(dim1) + block.x - 1) / block.x,
        (static_cast<unsigned>(dim2) + block.y - 1) / block.y
    );

    apply_bc_face_kernel<<<grid, block, 0, stream>>>(
        field, params.Nx, params.Ny, params.Nz,
        params.dx, params.dy, params.dz,
        axis, side, static_cast<int>(bc.type),
        bc.value, bc.flux,
        bc.alpha, bc.beta, bc.gamma);
}

void launch_boundary_conditions(
    double* field, const KernelParams& params,
    BCType bc_type, double bc_value, double bc_flux,
    double bc_alpha, double bc_beta, double bc_gamma,
    cudaStream_t stream)
{
    BoundaryConfig bc;
    bc.type = bc_type;
    bc.value = bc_value;
    bc.flux = bc_flux;
    bc.alpha = bc_alpha;
    bc.beta = bc_beta;
    bc.gamma = bc_gamma;

    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            launch_bc_face(field, params, axis, side, bc, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_boundary_conditions_per_face(
    double* field, const KernelParams& params,
    const PerFaceBoundary& face_bcs,
    cudaStream_t stream)
{
    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const auto& bc = face_bcs.get(axis, side);
            launch_bc_face(field, params, axis, side, bc, stream);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ac::cuda
