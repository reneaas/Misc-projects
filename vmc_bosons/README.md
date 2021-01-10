The vmc_bosons project relies heavily on Armadillo, a library of linear algebra and scientific computing in C++. For the codes to run, you will need to install this library
which is available for free at http://arma.sourceforge.net/.

The codes are parallelized using OpenMP. To achieve maximal speedup, the codes can be compiled on Linux using


```terminal
g++ -o main.out *.cpp -fopenmp -larmadillo -Ofast
```
