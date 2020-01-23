"""
This program combines the RBM with a classifier from tensorflow. The RBM is used for dimensionality reduction 
and then the classifier is learnt the training-data procued by the RBM.

"""

from __future__ import absolute_import, division, print_function, unicode_literals
from rbm import rbm, make_binary
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

#Load the data
#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
#train_images = train_images/255.0
#test_images = test_images/255.0

# load dataset
mnist = tf.keras.datasets.mnist
(trainX, trainY), (testX, testY) = mnist.load_data()
shuffled_indices = np.random.permutation(60000)
trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
#plot first few images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

plt.show()


def train_func(nexamples = 1500):
    """
    Trains the RBM.
    """
    start_time = time.time()
    RBM = rbm(28*28, 9*9, nCDsteps = 1, nepochs = 30)
    training_data = np.zeros(28*28, float)
    test_data = np.zeros(np.shape(training_data))
    #Training
    for k in np.arange(nexamples):
        print(k)
        training_data[:] = trainX[k].flat[:]
        training_data = make_binary(training_data)
        RBM.contrastive_divergence(training_data)

    elapsed_time = time.time()-start_time
    print("Trained for", elapsed_time, "seconds")

    #Testing
    for k in range(1,10):
        example = np.zeros((28,28),float)
        test_data[:] = testX[k-1].flat[:]
        test_data = make_binary(test_data)
        RBM.compute_reconstruction(test_data)
        #RBM.compute_hidden(test_data)
        #RBM.compute_visible(RBM.hiddenprob)
        #RBM.compute_hidden(RBM.visibleact)
        example.flat[:] = RBM.visibleprob[:]
        plt.subplot(330 + k)
        plt.imshow(example, cmap = plt.get_cmap("gray"))
    plt.figure()
    for k in range(1,10):
        plt.subplot(330 + k)
        plt.imshow(testX[k-1], cmap = plt.get_cmap("gray"))
    plt.show()
    return RBM, nexamples, start_time

RBM, nexamples, start_time = train_func()

model = keras.Sequential([
        keras.layers.Flatten(input_shape=(int(np.sqrt(RBM.nhidden)),int(np.sqrt(RBM.nhidden)))),              #similar to array.flat funtion in numpy.
        keras.layers.Dense(128,activation=tf.nn.relu),          #
        keras.layers.Dense(10,activation=tf.nn.softmax)         #Returns a layer of 10 different
        ])


#empty_dataset = np.zeros((10000, 100))
training_data = np.zeros((nexamples, int(np.sqrt(RBM.nhidden)),int(np.sqrt(RBM.nhidden))))
label_data = trainY[:nexamples]
input = np.zeros(28*28)
for k in range(nexamples):
    input[:] = trainX[k].flat[:]
    input = make_binary(input)
    RBM.compute_hidden(input)
    #train_data[k] = RBM.hiddenact
    training_data[k].flat = RBM.hiddenact


model.compile(optimizer = 'adam',
                loss='sparse_categorical_crossentropy',
                metrics = ['accuracy'])
model.fit(training_data, label_data,epochs = 5)


#Test the model.
#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print('Test_accuracy:', test_acc)

#predict labels.
N = int(np.sqrt(RBM.nhidden))
test_hidden = np.zeros((10000, N, N))
for k in range(10000):
    RBM.compute_hidden(testX[k].flat[:])
    test_hidden[k].flat[:] = RBM.hiddenact
predictions = model.predict(test_hidden)
most_likely_predictions = np.zeros(10000)
for k in range(len(most_likely_predictions)):
    most_likely_predictions[k] = np.argmax(predictions[k])
#print(most_likely_prediction)
number_of_correct_predictions = most_likely_predictions == testY


#number_of_correct = np.where(number_of_correct == 0)


print("Number of correct predictions:", np.sum(number_of_correct_predictions), "/10 000 test images, or", np.sum(number_of_correct_predictions)/10000*100, " %")
some_prediction = predictions[0]
#print("prediction:", some_prediction)
#print("actual:",testY[0])
#print(np.argmax(predictions[0]))
total_time = time.time() - start_time
print("Total time training and testing: ", total_time, "seconds")


"""
Tasks/ideas:
-: Compare results of Classifier vs RBM + Classifier on the MNIST dataset.

"""


def inpuc_func():
    features = training_data
    labels = label_data
    return features, labels


model = tf.estimator.LinearClassifier
