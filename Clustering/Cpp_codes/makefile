train_model: compile execute

compile:
	g++ -O3 -o main.exe $(wildcard *.cpp) -fopt-info

execute:
	./main.exe MNIST_training_data.txt MNIST_test_data.txt
