import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

mnist = tf.keras.datasets.mnist
#load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
shuffled_indices = np.random.permutation(10000)
testX, testY = testX[shuffled_indices], testY[shuffled_indices]

N_datapoints = 15;
handwritten_digit = 5
indices = np.where(testY == handwritten_digit)
test_data = testX[indices][:N_datapoints]
testing_data = np.zeros((N_datapoints, 28*28))
for i in range(N_datapoints):
    testing_data[i][:] = test_data[i].flat[:]/255
print(np.shape(testing_data))


with open("MNIST_test_data.txt", "w") as outfile:
    outfile.write("Lines: " + str(N_datapoints) + " Linelength: " + str(int(28*28)) + "\n")
    for k in range(N_datapoints):
        for l in range(28*28):
            outfile.write(str(testing_data[k][l]) + "\n")
