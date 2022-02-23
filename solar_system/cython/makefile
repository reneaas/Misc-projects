all: compile link run

compile:
	c++ -c main.cpp solar_system.cpp -std=c++11

link:
	c++ -o main.out main.o solar_system.o -larmadillo

run:
	./main.out