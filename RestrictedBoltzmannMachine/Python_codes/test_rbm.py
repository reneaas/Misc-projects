import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from rbm import rbm, make_binary

mnist = tf.keras.datasets.mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
shuffled_indices = np.random.permutation(60000)
trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
shuffled_indices = np.random.permutation(10000)
testX, testY = testX[shuffled_indices], testY[shuffled_indices]

size_of_dataset = 60000;
#Initialize the model
my_rbm = rbm(nvisible=28*28,
            nhidden=14*14,
            eta = 0.1,
            momentum = 0.9,
            nCDsteps = 1,
            nepochs = 300,
            batch_size = 100,
            size_of_dataset = size_of_dataset)

training_data = np.zeros((size_of_dataset, 28*28))
for k in range(size_of_dataset):
    training_data[k][:] = make_binary(trainX[k].flat[:])

start = time.time()
my_rbm.train_model(training_data)
end = time.time()
elapsed_time = end-start;
print("Model was trained for ", elapsed_time, " seconds")
my_rbm.plot_loss()

#Testing
test_data = np.zeros(28*28)
for k in range(1,10):
    example = np.zeros((28,28),float)
    test_data[:] = testX[k-1].flat[:]
    test_data = make_binary(test_data)

    my_rbm.compute_reconstruction(test_data)
    example.flat[:] = my_rbm.visibleprob[:]
    plt.subplot(330 + k)
    plt.imshow(example, cmap = plt.get_cmap("gray"))
plt.figure()
for k in range(1,10):
    plt.subplot(330 + k)
    plt.imshow(testX[k-1], cmap = plt.get_cmap("gray"))
plt.show()
