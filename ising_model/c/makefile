all: compile run

compile:
	gcc-10 -o main.out initialize_ising.c omp_monte_carlo_metropolis.c main.c -lm -fopenmp -Ofast -march=native

run:
	./main.out
