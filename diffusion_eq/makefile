serial: compile_serial execute_serial

compile_serial:
	g++ -O3 -o serial_main.exe main.cpp diffusion_solver.cpp

execute_serial:
	./serial_main.exe

omp: compile_omp execute_omp

compile_omp:
	g++ -O3 -fopenmp -o omp_main.exe main.cpp diffusion_solver.cpp

execute_omp:
	./omp_main.exe

mpi: compile_mpi execute_mpi

compile_mpi:
	mpic++ -o mpi_main.exe main.cpp diffusion_solver.cpp mpi_diffusion_solver.cpp

execute_mpi:
	mpirun -n 4 ./mpi_main.exe
