# Allen Cahn CUDA (Phase-Field Simulation of Dendritic Solidification)

Description: This CUDA program does phase-field simulation of dendritic solidification based on these references:

1. [Yung-Tae Kim, Nikolas Provatas, Nigel Goldenfeld, and Jonathan Dantzig, Universal dynamics of phase-field models for dendritic growth, Phys. Rev. E 59, R2546(R)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.59.R2546)

2. [Anand Kumar, Isotropic finite-differences, Journal of Computational Physics, Volume 201, Issue 1, 20 November 2004, Pages 109-118](https://www.sciencedirect.com/science/article/pii/S0021999104002037)

Dependecies: CUDA Toolkit 9.2, GCC 6.3, VTK 8.2, CMake 3.10.0

In order to compile the program use these commands in UNIX shell terminals:

```
git clone git@github.com:myousefi2016/Allen-Cahn-CUDA.git
cd Allen-Cahn-CUDA && mkdir build && cd build
cmake .. && make
```

To run the program after compilation just running this command:

```
mkdir out && ./Allen-Cahn-CUDA
```

It will store the results in out directory as vtk files. Good luck and if you use this piece of code for your research don't forget to give attribute to this github repository.

![alt text](https://raw.githubusercontent.com/myousefi2016/Allen-Cahn-CUDA/master/result/img.png)
