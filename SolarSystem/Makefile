all: compile execute plot

compile:
	g++ -O3 -o main.exe $(wildcard *.cpp) -march=native

execute:
	./main.exe


plot:
	python3 plot.py
