## 2D Diffusion Solver

In main.cpp you can find an example showcasing the use of the
the class DiffusionSolver. For now, the code is either serial
or parallelized with OpenMP.

For compilation details, take a look at the makefile.

### Serial code

To run serial code, run the following command in a linux terminal:

```terminal
make serial
```

### Code parallelized with OpenMP

To run the code parallelized with OpenMP, run the follwing command in a linux terminal:

```terminal
make omp
```



### Code parallelized with MPI

Will be upstream shortly.
