# PDE Solvers

A set of diffeq numerical methods for obtaining solutions. This will include Jacobi relaxation and multigrid extensions, as well as other solution techniques. These require MPI to be installed, to be run with `mpirun` with either MPICH or OpenMPI. The compiler must support C++23 and at least OpenMP 4.5. A hybrid parallelism model is used.

### Details

- Gauss-Seidel vs matrix formulation of the poisson problem: matrix inversion has a complexity of n^3 and there aren't faster methods - (QR, LU and others require n^3)
