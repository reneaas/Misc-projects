all: compile run

compile:
	gcc -o main.out initialize_ising.c omp_monte_carlo_metropolis.c main.c -lm -Xpreprocessor -fopenmp -O3 -lomp

run:
	./main.out
