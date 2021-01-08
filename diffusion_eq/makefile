all: compile run

compile:
	g++ -o main.out $(wildcard *.cpp) -O3 -larmadillo -fopenmp -march=native

run:
	./main.out
