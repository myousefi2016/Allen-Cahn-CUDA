#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <random>
#include <fstream>
#include <cmath>

#include <vtkVersion.h>
#include <vtkSmartPointer.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkStructuredGrid.h>
#include <vtkPointData.h>
#include <vtkDoubleArray.h>

// set a 3D volume
// To compile it with nvcc execute: nvcc -O2 -o set3d set3d.cu
//define the data set size (cubic volume)
#define DATAXSIZE 128
#define DATAYSIZE 128
#define DATAZSIZE 128
//define the chunk sizes that each threadblock will work on

using namespace std;

vtkSmartPointer<vtkDoubleArray> convertArrayToVTK(double phi[][DATAYSIZE][DATAXSIZE], char* name)
{

  int counter = 0;
  vtkSmartPointer<vtkDoubleArray> phiVTK =
          vtkSmartPointer<vtkDoubleArray>::New();

  phiVTK->SetNumberOfComponents(1);
  phiVTK->SetNumberOfTuples(DATAXSIZE * DATAYSIZE * DATAZSIZE);

  for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
  for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
  for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {

          phiVTK->SetValue(counter, phi[idx][idy][idz]);
          counter++;

  }
  }
  }

  phiVTK->SetName(name);

  return phiVTK;

}

vtkSmartPointer<vtkPoints> createVTKGrid()
{

 vtkSmartPointer<vtkPoints> points =
    vtkSmartPointer<vtkPoints>::New();

  for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
  for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
  for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {

   points->InsertNextPoint(idx, idy, idz);

  }
  }
  }

  return points;

}

void writeVTKFile(std::vector<vtkSmartPointer<vtkDoubleArray>> Arrays, vtkSmartPointer<vtkPoints> points, int t)
{

   string name = "./out/output_" + to_string(t) + ".vtk";

   vtkSmartPointer<vtkStructuredGrid> structuredGrid =
    vtkSmartPointer<vtkStructuredGrid>::New();

  structuredGrid->SetDimensions(DATAXSIZE,DATAYSIZE,DATAZSIZE);
  structuredGrid->SetPoints(points);
  for (int i = 0; i < Arrays.size(); i++)
  {
  structuredGrid->GetPointData()->AddArray(Arrays[i]);
  }

  vtkSmartPointer<vtkXMLStructuredGridWriter> writer =
    vtkSmartPointer<vtkXMLStructuredGridWriter>::New();

  writer->SetFileName(name.c_str());

  writer->SetInputData(structuredGrid);

  writer->Update();

}

// for cuda error checking
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            return 1; \
        } \
    } while (0)

void computeIDs(int *IDx, int *IDy, int *IDz)
{
     int counter = 0;
     for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
     for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
      IDx[counter] = idx;
      IDy[counter] = idy;
      IDz[counter] = idz;
      counter++;
    }
     }
      }
}

__device__ double dFphi(double phi, double u, double lambda)
{

  return (-phi*(1.0-phi*phi)+lambda*u*(1.0-phi*phi)*(1.0-phi*phi));

}

__device__ double GradientX(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double phix = (phi[x+1][y][z] - phi[x-1][y][z]) / (2.0*dx);

  return phix;

}

__device__ double GradientY(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double phiy = (phi[x][y+1][z] - phi[x][y-1][z]) / (2.0*dy);

  return phiy;

}

__device__ double GradientZ(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double phiz = (phi[x][y][z+1] - phi[x][y][z-1]) / (2.0*dz);

  return phiz;

}

__device__ double Divergence(double phix[][DATAYSIZE][DATAXSIZE], double phiy[][DATAYSIZE][DATAXSIZE], double phiz[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double div = GradientX(phix,dx,dy,dz,x,y,z) + GradientY(phiy,dx,dy,dz,x,y,z) + GradientZ(phiz,dx,dy,dz,x,y,z);

  return div;

}

__device__ double Laplacian(double phi[][DATAYSIZE][DATAXSIZE], double dx, double dy, double dz, int x, int y, int z)
{

  double phixx = (phi[x+1][y][z] + phi[x-1][y][z] - 2.0*phi[x][y][z]) / (dx*dx);
  double phiyy = (phi[x][y+1][z] + phi[x][y-1][z] - 2.0*phi[x][y][z]) / (dy*dy);
  double phizz = (phi[x][y][z+1] + phi[x][y][z-1] - 2.0*phi[x][y][z]) / (dz*dz);

  double result = phixx + phiyy + phizz;

  return result;

}

__device__ double An(double phix, double phiy, double phiz, double epsilon)
{
 if (phix != 0.0 || phiy != 0.0 || phiz != 0.0){
 return ((1.0-3.0*epsilon)*(1.0+(((4.0*epsilon)/(1.0-3.0*epsilon))*((phix*phix*phix*phix+phiy*phiy*phiy*phiy+phiz*phiz*phiz*phiz)/((phix*phix+phiy*phiy+phiz*phiz)*(phix*phix+phiy*phiy+phiz*phiz))))));
 }
 else
 {
 return (1.0-((5.0/3.0)*epsilon));
 }
}

__device__ double Wn(double phix, double phiy, double phiz, double epsilon, double W0)
{

  return (W0*An(phix,phiy,phiz,epsilon));

}

__device__ double taun(double phix, double phiy, double phiz, double epsilon, double tau0)
{

  return (tau0*An(phix,phiy,phiz,epsilon)*An(phix,phiy,phiz,epsilon));

}

__device__ double dFunc(double l, double m, double n)
{
 if (l != 0.0 || m != 0.0 || n != 0.0){
 return (((l*l*l*(m*m+n*n))-(l*(m*m*m*m+n*n*n*n)))/((l*l+m*m+n*n)*(l*l+m*m+n*n)));
 }
 else
 {
 return 0.0;
 }
}

__global__ void calculateForce(double phi[][DATAYSIZE][DATAXSIZE], double Fx[][DATAYSIZE][DATAXSIZE], double Fy[][DATAYSIZE][DATAXSIZE], double Fz[][DATAYSIZE][DATAXSIZE], int *IDx, int *IDy, int *IDz, double dx, double dy, double dz, double epsilon, double W0, double tau0)
{

 unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

 if ((IDx[idx] < (DATAXSIZE-1)) && (IDy[idx] < (DATAYSIZE-1)) && (IDz[idx] < (DATAZSIZE-1)) && (IDx[idx] > (0)) && (IDy[idx] > (0)) && (IDz[idx] > (0))){

  double phix = GradientX(phi,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
  double phiy = GradientY(phi,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
  double phiz = GradientZ(phi,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
  double sqGphi = phix*phix+phiy*phiy+phiz*phiz;

  Fx[IDx[idx]][IDy[idx]][IDz[idx]] = Wn(phix,phiy,phiz,epsilon,W0) *  Wn(phix,phiy,phiz,epsilon,W0) * phix + sqGphi * Wn(phix,phiy,phiz,epsilon,W0) * (16.0*W0*epsilon) * dFunc(phix,phiy,phiz);
  Fy[IDx[idx]][IDy[idx]][IDz[idx]] = Wn(phix,phiy,phiz,epsilon,W0) *  Wn(phix,phiy,phiz,epsilon,W0) * phiy + sqGphi * Wn(phix,phiy,phiz,epsilon,W0) * (16.0*W0*epsilon) * dFunc(phiy,phiz,phix);
  Fz[IDx[idx]][IDy[idx]][IDz[idx]] = Wn(phix,phiy,phiz,epsilon,W0) *  Wn(phix,phiy,phiz,epsilon,W0) * phiz + sqGphi * Wn(phix,phiy,phiz,epsilon,W0) * (16.0*W0*epsilon) * dFunc(phiz,phix,phiy);
 }
 else
 {
  Fx[IDx[idx]][IDy[idx]][IDz[idx]] = 0.0;
  Fy[IDx[idx]][IDy[idx]][IDz[idx]] = 0.0;
  Fz[IDx[idx]][IDy[idx]][IDz[idx]] = 0.0;
 }

}

// device function to set the 3D volume
__global__ void allenCahn(double phinew[][DATAYSIZE][DATAXSIZE], double phiold[][DATAYSIZE][DATAXSIZE], double uold[][DATAYSIZE][DATAXSIZE], double Fx[][DATAYSIZE][DATAXSIZE], double Fy[][DATAYSIZE][DATAXSIZE], double Fz[][DATAYSIZE][DATAXSIZE], int *IDx, int *IDy, int *IDz, double epsilon, double W0, double tau0, double lambda, double dt, double dx, double dy, double dz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

    if ((IDx[idx] < (DATAXSIZE-1)) && (IDy[idx] < (DATAYSIZE-1)) && (IDz[idx] < (DATAZSIZE-1)) && (IDx[idx] > (0)) && (IDy[idx] > (0)) && (IDz[idx] > (0))){

      double phix = GradientX(phiold,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
      double phiy = GradientY(phiold,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
      double phiz = GradientZ(phiold,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]); 
  
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = phiold[IDx[idx]][IDy[idx]][IDz[idx]] + (dt / taun(phix,phiy,phiz,epsilon,tau0)) * (Divergence(Fx,Fy,Fz,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]) - dFphi(phiold[IDx[idx]][IDy[idx]][IDz[idx]],uold[IDx[idx]][IDy[idx]][IDz[idx]],lambda));
      }
}

__global__ void boundaryConditionsPhi(double phinew[][DATAYSIZE][DATAXSIZE], int *IDx, int *IDy, int *IDz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (IDx[idx] == 0){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }
    else if (IDx[idx] == DATAXSIZE-1){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }
    else if (IDy[idx] == 0){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }
    else if (IDy[idx] == DATAYSIZE-1){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }
    else if (IDz[idx] == 0){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }
    else if (IDz[idx] == DATAZSIZE-1){
      phinew[IDx[idx]][IDy[idx]][IDz[idx]] = -1.0;
      }

}

__global__ void thermalEquation(double unew[][DATAYSIZE][DATAXSIZE], double uold[][DATAYSIZE][DATAXSIZE], double phinew[][DATAYSIZE][DATAXSIZE], double phiold[][DATAYSIZE][DATAXSIZE], int *IDx, int *IDy, int *IDz, double D, double dt, double dx, double dy, double dz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

    if ((IDx[idx] < (DATAXSIZE-1)) && (IDy[idx] < (DATAYSIZE-1)) && (IDz[idx] < (DATAZSIZE-1)) && (IDx[idx] > (0)) && (IDy[idx] > (0)) && (IDz[idx] > (0))){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] = uold[IDx[idx]][IDy[idx]][IDz[idx]] + 0.5*(phinew[IDx[idx]][IDy[idx]][IDz[idx]]-phiold[IDx[idx]][IDy[idx]][IDz[idx]]) + dt * D * Laplacian(uold,dx,dy,dz,IDx[idx],IDy[idx],IDz[idx]);
      }
}

__global__ void boundaryConditionsU(double unew[][DATAYSIZE][DATAXSIZE], double delta, int *IDx, int *IDy, int *IDz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (IDx[idx] == 0){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }
    else if (IDx[idx] == DATAXSIZE-1){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }
    else if (IDy[idx] == 0){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }
    else if (IDy[idx] == DATAYSIZE-1){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }
    else if (IDz[idx] == 0){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }
    else if (IDz[idx] == DATAZSIZE-1){
      unew[IDx[idx]][IDy[idx]][IDz[idx]] =  -delta;
      }

}

__global__ void Swap(double cnew[][DATAYSIZE][DATAXSIZE], double cold[][DATAYSIZE][DATAXSIZE], int *IDx, int *IDy, int *IDz)
{
    unsigned idx = blockIdx.x*blockDim.x + threadIdx.x;

    double tmp;    

    if ((IDx[idx] < (DATAXSIZE)) && (IDy[idx] < (DATAYSIZE)) && (IDz[idx] < (DATAZSIZE))) {
     tmp=cnew[IDx[idx]][IDy[idx]][IDz[idx]];
     cnew[IDx[idx]][IDy[idx]][IDz[idx]]=cold[IDx[idx]][IDy[idx]][IDz[idx]];
     cold[IDx[idx]][IDy[idx]][IDz[idx]]=tmp;
    }

}

void initializationPhi(double phi[][DATAYSIZE][DATAXSIZE], double r0)
{
    for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
     for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
      double r = std::sqrt((idx-0.5*DATAXSIZE)*(idx-0.5*DATAXSIZE) + (idy-0.5*DATAYSIZE)*(idy-0.5*DATAYSIZE) + (idz-0.5*DATAZSIZE)*(idz-0.5*DATAZSIZE));
      if (r < r0){
      phi[idx][idy][idz] = 1.0;
      }
      else
      {
      phi[idx][idy][idz] = -1.0;
      }
    }
     }
      }
}

void initializationU(double u[][DATAYSIZE][DATAXSIZE], double r0, double delta)
{
    for (unsigned int idx = 0.0; idx < DATAXSIZE; idx++) {
     for (unsigned int idy = 0.0; idy < DATAYSIZE; idy++) {
      for (unsigned int idz = 0.0; idz < DATAZSIZE; idz++) {
      double r = std::sqrt((idx-0.5*DATAXSIZE)*(idx-0.5*DATAXSIZE) + (idy-0.5*DATAYSIZE)*(idy-0.5*DATAYSIZE) + (idz-0.5*DATAZSIZE)*(idz-0.5*DATAZSIZE));
      if (r < r0) {
      u[idx][idy][idz] = 0.0;
      }
      else
      {
      u[idx][idy][idz] = -delta * (1.0 - std::exp(-(r-r0)));
      }
    }
     }
      }
}

int main(int argc, char *argv[])
{
    double dx = 0.4;
    double dy = 0.4;
    double dz = 0.4;
    double dt = 0.01;
    int t_f = 1000;
    int t_freq = 100;
    double delta = 0.8;
    double r0 = 5.0;
    double epsilon = 0.07;
    double W0 = 1.0;
    double beta0 = 0.0;
    double D = 2.0;
    double d0 = 0.5;
    double a1 = 1.25 / std::sqrt(2.0);
    double a2 = 0.64;
    double lambda = (W0*a1)/(d0);
    double tau0 = ((W0*W0*W0*a1*a2)/(d0*D)) + ((W0*W0*beta0)/(d0));
    vtkSmartPointer<vtkPoints> points = createVTKGrid();
    cudaSetDevice(0.0);
    typedef double nRarray[DATAYSIZE][DATAXSIZE];
    const int BLOCK_SIZE = 1024;
    const int siteCount = DATAXSIZE*DATAYSIZE*DATAZSIZE;
    const int GRID_SIZE = (siteCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
// overall data set sizes
    const int nx = DATAXSIZE;
    const int ny = DATAYSIZE;
    const int nz = DATAZSIZE;
// pointers for data set storage via malloc
    nRarray *phi_host; // storage for result stored on host
    nRarray *u_host;
    int *IDx_host;
    int *IDy_host;
    int *IDz_host;
    nRarray *d_phiold;  // storage for result computed on device
    nRarray *d_phinew;
    nRarray *d_uold;
    nRarray *d_unew;
    nRarray *d_Fx;
    nRarray *d_Fy;
    nRarray *d_Fz;
    int *d_IDx;
    int *d_IDy;
    int *d_IDz;
// allocate storage for data set
    cudaHostAlloc((void**)&phi_host,(nx*ny*nz)*sizeof(double),cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate host buffer");
    cudaHostAlloc((void**)&u_host,(nx*ny*nz)*sizeof(double),cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate host buffer");
    cudaHostAlloc((void**)&IDx_host,(nx*ny*nz)*sizeof(double),cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate host buffer");
    cudaHostAlloc((void**)&IDy_host,(nx*ny*nz)*sizeof(double),cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate host buffer");
    cudaHostAlloc((void**)&IDz_host,(nx*ny*nz)*sizeof(double),cudaHostAllocMapped);
    cudaCheckErrors("Failed to allocate host buffer");
    //if ((phi_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    //if ((u_host = (nRarray *)malloc((nx*ny*nz)*sizeof(double))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    //if ((IDx_host = (int *)malloc((nx*ny*nz)*sizeof(int))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    //if ((IDy_host = (int *)malloc((nx*ny*nz)*sizeof(int))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
    //if ((IDz_host = (int *)malloc((nx*ny*nz)*sizeof(int))) == 0) {fprintf(stderr,"malloc1 Fail \n"); return 1;}
// allocate GPU device buffers
    cudaMalloc((void **) &d_phiold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_phinew, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_uold, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_unew, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_Fx, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_Fy, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_Fz, (nx*ny*nz)*sizeof(double));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_IDx, (nx*ny*nz)*sizeof(int));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_IDy, (nx*ny*nz)*sizeof(int));
    cudaCheckErrors("Failed to allocate device buffer");
    cudaMalloc((void **) &d_IDz, (nx*ny*nz)*sizeof(int));
    cudaCheckErrors("Failed to allocate device buffer");
// compute result

    initializationPhi(phi_host,r0);
    initializationU(u_host,r0,delta);

    std::vector<vtkSmartPointer<vtkDoubleArray>> ArraysInitial;

    ArraysInitial.push_back(convertArrayToVTK(phi_host,"phi"));
    ArraysInitial.push_back(convertArrayToVTK(u_host,"u"));

    writeVTKFile(ArraysInitial,points,0);

    cudaMemcpyAsync(d_phiold, phi_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    cudaMemcpyAsync(d_uold, u_host, ((nx*ny*nz)*sizeof(double)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    computeIDs(IDx_host,IDy_host,IDz_host);

    cudaMemcpyAsync(d_IDx, IDx_host, ((nx*ny*nz)*sizeof(int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    cudaMemcpyAsync(d_IDy, IDy_host, ((nx*ny*nz)*sizeof(int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    cudaMemcpyAsync(d_IDz, IDz_host, ((nx*ny*nz)*sizeof(int)), cudaMemcpyHostToDevice);
    cudaCheckErrors("CUDA memcpy failure");

    double clock_d = double(clock()) / CLOCKS_PER_SEC;

    int t = 0;

    while (t <= t_f) {

    printf("Timestep is: %d\n",t);

    calculateForce<<<GRID_SIZE,BLOCK_SIZE>>>(d_phiold,d_Fx,d_Fy,d_Fz,d_IDx,d_IDy,d_IDz,dx,dy,dz,epsilon,W0,tau0);
    cudaCheckErrors("Kernel launch failure");
    allenCahn<<<GRID_SIZE,BLOCK_SIZE>>>(d_phinew,d_phiold,d_uold,d_Fx,d_Fy,d_Fz,d_IDx,d_IDy,d_IDz,epsilon,W0,tau0,lambda,dt,dx,dy,dz);
    cudaCheckErrors("Kernel launch failure");
    boundaryConditionsPhi<<<GRID_SIZE,BLOCK_SIZE>>>(d_phinew,d_IDx,d_IDy,d_IDz);
    cudaCheckErrors("Kernel launch failure");

    thermalEquation<<<GRID_SIZE,BLOCK_SIZE>>>(d_unew,d_uold,d_phinew,d_phiold,d_IDx,d_IDy,d_IDz,D,dt,dx,dy,dz);
    cudaCheckErrors("Kernel launch failure");
    boundaryConditionsU<<<GRID_SIZE,BLOCK_SIZE>>>(d_unew,delta,d_IDx,d_IDy,d_IDz);
    cudaCheckErrors("Kernel launch failure");

    if (t % t_freq == 0 && t > 0) {

     cudaMemcpyAsync(phi_host, d_phinew, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     cudaMemcpyAsync(u_host, d_unew, ((nx*ny*nz)*sizeof(double)), cudaMemcpyDeviceToHost);
     cudaCheckErrors("CUDA memcpy failure");

     std::vector<vtkSmartPointer<vtkDoubleArray>> Arrays;

     Arrays.push_back(convertArrayToVTK(phi_host,"phi"));
     Arrays.push_back(convertArrayToVTK(u_host,"u"));

     writeVTKFile(Arrays,points,t);

    }
    
    Swap<<<GRID_SIZE,BLOCK_SIZE>>>(d_phinew, d_phiold,d_IDx,d_IDy,d_IDz);
    cudaCheckErrors("Kernel launch failure");

    Swap<<<GRID_SIZE,BLOCK_SIZE>>>(d_unew, d_uold,d_IDx,d_IDy,d_IDz);
    cudaCheckErrors("Kernel launch failure");

    t++;

    }

    cudaThreadSynchronize();
    clock_d = double(clock()) / CLOCKS_PER_SEC - clock_d; 
    printf("GPU time = %.3fms\n",clock_d*1e3);

    free(phi_host);
    free(u_host);
    free(IDx_host);
    free(IDy_host);
    free(IDz_host);
    cudaFree(d_phiold);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_phinew);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_uold);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_unew);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_Fx);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_Fy);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_Fz);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_IDx);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_IDy);
    cudaCheckErrors("cudaFree fail");
    cudaFree(d_IDz);
    cudaCheckErrors("cudaFree fail");
    return 0;
}
