all: compile link run

compile:
	g++-11 -c main.cpp solar_storm.cpp -O3 -fopenmp -std=c++11 -mtune=native

link:
	g++-11 -o main.out main.o solar_storm.o -fopenmp

run:
	./main.out