all: compile link run

compile:
	c++ -c $(wildcard *.cpp) -Ofast -std=c++11 -larmadillo -Xpreprocessor -fopenmp

link:
	c++ -o main.out $(wildcard *.o) -Xpreprocessor -fopenmp -lomp -larmadillo

run:
	./main.out
