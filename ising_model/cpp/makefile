all: compile link run

compile:
	g++ -c $(wildcard *.cpp) -O3 -Xpreprocessor -fopenmp -std=c++11

link:
	g++ -o main.out *.o -larmadillo -lomp


run:
	./main.out
