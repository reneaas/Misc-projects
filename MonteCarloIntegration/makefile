serial: compile_serial execute_serial

parallel: compile_parallel execute_parallel

compile_serial:
	g++ -O3 -o main_serial.exe $(wildcard *.cpp)

execute_serial:
	./main_serial.exe

compile_parallel:
	g++ -O3 -ffast-math -fopenmp -o main_parallel.exe $(wildcard *.cpp)

execute_parallel:
	./main_parallel.exe
