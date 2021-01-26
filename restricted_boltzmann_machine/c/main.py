import os
from rbm import rbm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

instruction = input("Train model? y/n\n")

if instruction == "y":
    os.system("make train_model")

#After training (if training is chosen), we initiate reconstruction of an arbitrary data point from the test set.

weights_filename = "weights.txt"
visiblebias_filename = "visiblebias.txt"
hiddenbias_filename = "hiddenbias.txt"

RBM = rbm(nvisible=28*28, nhidden=14*14, nCDsteps = 1, nepochs = 100)

with open(weights_filename, "r") as infile:
    lines = infile.readlines()
    i = 0
    for line in lines:
        vals = line.split()
        for j in range(len(vals)):
            RBM.weights[i][j] = float(vals[j])
        i += 1

print(RBM.weights)

with open(visiblebias_filename, "r") as infile:
    lines = infile.readlines()
    i = 0
    for line in lines:
        vals = line.split()
        RBM.visiblebias[i] = float(vals[0])
        i += 1

with open(hiddenbias_filename, "r") as infile:
    lines = infile.readlines()
    i = 0
    for line in lines:
        vals = line.split()
        RBM.hiddenbias[i] = float(vals[0])
        i += 1

mnist = tf.keras.datasets.mnist
#load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
shuffled_indices = np.random.permutation(10000)
testX, testY = testX[shuffled_indices], testY[shuffled_indices]


test_data = np.zeros(28*28)
test_data[:] = testX[0].flat[:]/255
for i in range(len(test_data)):
    if test_data[i] > 0.5:
        test_data[i] = 1
    else:
        test_data[i] = 0

plt.imshow(testX[0]/255)
plt.colorbar()
plt.show()

RBM.compute_reconstruction(test_data)
image = np.zeros((28,28))
image.flat[:] = RBM.visibleprob[:]

plt.imshow(image);
plt.colorbar();
plt.show()
