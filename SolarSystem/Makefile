CXXFLAGS= -O3 -march=native -mtune=native

all: compile execute plot

compile:
	g++ $(CXXFLAGS) -o main.exe $(wildcard *.cpp)

execute:
	./main.exe


plot:
	python3 plot.py
