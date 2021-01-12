#include <fstream>
#include <armadillo>
#include <cstdio>


using namespace std;
using namespace arma;


void create_mnist_binary_files(int num_train, int num_val, int num_test);

void create_frankefunction_binary_files();


int main(int argc, char const *argv[]) {

    //Create mnist binary files to be used with armadillo matrices.
    int num_train, num_val, num_test;
    num_train = 0.95*60000;
    num_val = 0.05*60000;
    num_test = 10000;
    create_mnist_binary_files(num_train, num_val, num_test);
    return 0;
}

void create_mnist_binary_files(int num_train, int num_val, int num_test)
{
    int features = 28*28;
    int num_outputs = 10;

    mat X_train = mat(features, num_train);
    mat y_train = mat(num_outputs, num_train);

    mat X_val = mat(features, num_val);
    mat y_val = mat(num_outputs, num_val);

    mat X_test = mat(features, num_test);
    mat y_test = mat(num_outputs, num_test);


    char* infilename_X = "mnist_training_X.txt";
    char* infilename_y = "mnist_training_Y.txt";
    FILE *fp_X = fopen(infilename_X, "r");
    FILE *fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_train; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &X_train(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &y_train(j, i));
        }
    }

    for (int i = 0; i < num_val; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &X_val(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &y_val(j, i));
        }
    }-

    fclose(fp_X);
    fclose(fp_y);

    X_train.save("mnist_X_train.bin");
    y_train.save("mnist_y_train.bin");

    X_val.save("mnist_X_val.bin");
    y_val.save("mnist_y_val.bin");




    infilename_X = "mnist_test_X.txt";
    infilename_y = "mnist_test_Y.txt";
    fp_X = fopen(infilename_X, "r");
    fp_y = fopen(infilename_y, "r");
    for (int i = 0; i < num_test; i++){
        for (int j = 0; j < features; j++){
            fscanf(fp_X, "%lf", &X_test(j, i));
        }
        for (int j = 0; j < num_outputs; j++){
            fscanf(fp_y, "%lf", &y_test(j, i));
        }
    }
    fclose(fp_X);
    fclose(fp_y);

    X_test.save("mnist_X_test.bin");
    y_test.save("mnist_y_test.bin");

}
