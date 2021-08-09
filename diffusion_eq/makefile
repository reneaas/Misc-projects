all: compile run

compile:
	g++ -o main.out $(wildcard *.cpp) -O3 -Xpreprocessor -fopenmp -std=c++11 -lomp -larmadillo

run:
	./main.out
