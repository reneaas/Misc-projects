all: compile link run

compile:
	g++-11 -c $(wildcard *.cpp) -O3 -std=c++11 -fopenmp

link:
	g++-11 -o main.out $(wildcard *.o) -fopenmp -larmadillo

run:
	./main.out
