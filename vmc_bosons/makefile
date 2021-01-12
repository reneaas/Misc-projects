all: compile run

compile:
	g++-10 -o main.out $(wildcard *.cpp) -Ofast -larmadillo -fopenmp -march=native 

run:
	./main.out
