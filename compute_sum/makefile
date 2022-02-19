serial: compile_serial link_serial run

parallel: compile_omp link_omp run

compile_serial:
	g++ -c compute_sum.cpp

link_serial:
	g++ -o main.out *.o

compile_omp:
	g++ -c compute_sum.cpp -Xpreprocessor -fopenmp

link_omp:
	g++ -o main.out *.o -lomp

run:
	./main.out
