all: compile execute

compile:
	g++ -O3 -o main.exe $(wildcard *.cpp) -march=native

execute:
	./main.exe && python3 plot.py
