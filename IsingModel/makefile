CXXFLAGS1= -O3 -march=native -mtune=native
NAMES = main.cpp ising2d.cpp
CXXFLAGS_OMP= -O3 -fopenmp -march=native
NAMES_OMP = main.cpp ising2d.cpp
CXXFLAGS_MPI= -O3 -march=native -fopenmp
NAMES_MPI= ising2d.cpp main_mpi.cpp

serial: compile execute

compile:
	g++ $(CXXFLAGS1) -o main.out $(NAMES)

execute:
	./main.out


omp: compile_omp execute_omp

compile_omp:
	g++ $(CXXFLAGS_OMP) -o main_omp.out $(NAMES_OMP)

execute_omp:
	./main_omp.out

mpi: compile_mpi execute_mpi

compile_mpi:
	mpicxx $(CXXFLAGS_MPI) -o main_mpi.out $(NAMES_MPI)

execute_mpi:
	mpirun -n 4 ./main_mpi.out
