import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
from matplotlib import animation

from rbm import rbm, make_binary
from clustering import K_means, K_means_NN

mnist = tf.keras.datasets.mnist
#load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()
shuffled_indices = np.random.permutation(60000)
trainX, trainY = trainX[shuffled_indices], trainY[shuffled_indices]
#summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, y=%s' % (testX.shape, testY.shape))
#plot first few images
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))

plt.show()


#Writes the MNIST dataset to a txt file
with open("MNIST_training_data.txt", "w") as outfile:
    for k in range(1000):
        outfile.write(str(trainX[k].flat[:]) + "\n")


def test_K_means_NN():
    """
    test function for the class K_means_NN. The test function essentially summarizes how the class can be used in 4 steps:
    1) Preprocess training data
    2) Initiate an instance of the class
    3) Train it
    4) Make a prediction on test data.
    """
    #Preprocess the dataset and prepare it for use with the class.
    training_data = np.zeros((60000, 28*28))
    for k in range(60000):
        training_data[k] = trainX[k].flat[:]


    #Initate an instance of the class K_means_NN.
    Clusters = K_means_NN(Kneurons = 10, dataset = training_data)

    #Trains the neural network on the dataset
    Clusters.train(nepochs = 500)

    #Test the K_means_NN using test data.
    numbers = [i for i in range(10)]
    ordered_data = {str(i):[] for i in range(10)}
    for number in numbers:
        indices = np.where(testY == number)
        test_data = testX[indices]
        for e in range(np.shape(test_data)[0]):
            ordered_data[str(number)].append(test_data[e])

    for k in range(9):
        plt.subplot(330 + 1 + k)
        plt.imshow(ordered_data["3"][k], cmap = plt.get_cmap("gray"))

    plt.show()

    for number in numbers:
        print("--------------------------Predicting the cluster to which the number " + str(number) + " belongs to ------------------------------")
        for k in range(9):
            activation = Clusters.predict(test_datapoint = ordered_data[str(number)][k].flat[:])
            print(activation)
#test_K_means_NN()


def test_func_RBM():
    """
    Test function to test the RBM-class combined with the K-means class.
    The test function essentially does the following:
    1) Initiates an instance of the RBM-class and trains it on the MNIST dataset.
    2) Tests the RBM using test data from the MNIST dataset and reconstructs the data to test the fidelity of the machine.
    3) Uses the RBM to reduce the dimensionality of the MNIST dataset and use the K-means class to cluster hidden units of the RBM into K clusters.
    4) Then we make predictions on test-data and make it back to visible space to see if all the numbers plotted in fact are the same numbers and truly belong to the same cluster.
    """
    start_time = time.time()
    
    #Train the RBM
    R = rbm(nvisible=28*28, nhidden=8*8, nCDsteps = 5, nepochs = 15)
    training_data = np.zeros(28*28, float)
    test_data = np.zeros(np.shape(training_data))
    number_of_training_examples = 60000 


    #training_data[:] = trainX[k].flat[:]
    #training_data = make_binary(training_data)
    for k in range(number_of_training_examples):
        print(k)
        R.contrastive_divergence(make_binary(trainX[k].flat[:]))

    elapsed_time = time.time()-start_time
    print("Trained for", elapsed_time, "seconds")
    
    #Testing
    for k in range(1,10):
        example = np.zeros((28,28),float)
        test_data[:] = testX[k-1].flat[:]
        test_data = make_binary(test_data)

        R.compute_reconstruction(test_data)
        example.flat[:] = R.visibleprob[:]
        plt.subplot(330 + k)
        plt.imshow(example, cmap = plt.get_cmap("gray"))
    plt.savefig("RBM_reconstructions_" + str(R.nhidden) + "hiddenunits_" + str(R.nCDsteps) + "CDsteps_" + str(R.nepochs) + "epochs_" \
            + str(number_of_training_examples) + "trainingexamples" + ".png")
    plt.figure()
    for k in range(1,10):
        plt.subplot(330 + k)
        plt.imshow(testX[k-1], cmap = plt.get_cmap("gray"))
    plt.show()
    

    #NEW test, initializes arbitrary input and makes lets the RBM dream up some number.
    input_data = testX[np.random.randint(0,10000)].flat[:]
    input_data = input_data/np.max(input_data)
    #input_data = make_binary(input_data)
    R.compute_hidden(input_data)
    N_iterations = 1000000
    image = np.zeros((28,28))
    image.flat[:] = input_data

    fig = plt.figure()
    plt.title("iteration = 0")
    im = plt.imshow(image, cmap = plt.get_cmap("gray"),vmin = 0, vmax = 1)

    def animate(i):
        R.compute_visible(R.hiddenact)
        R.compute_hidden(R.visibleprob)
        image = np.zeros((28,28))
        image.flat[:] = R.visibleprob
        im.set_array(image)
        plt.title("iteration = " + str(i))
        return [im]

    ani = animation.FuncAnimation(
        fig, animate, interval=1, frames = range(5000), blit=False)
    #plt.show()
    ani.save("RBM_dream_ " + str(R.nhidden) + "hiddenunits_" + str(R.nCDsteps) + "CDsteps_" + str(R.nepochs) + "epochs_" \
            + str(number_of_training_examples) + "trainingexamples" + ".mp4", writer = "ffmpeg",fps =500)


test_func_RBM()




def test_func_RBM_Kmeans():
    """
    Test function to test the RBM-class combined with the K-means class.
    The test function essentially does the following:
    1) Initiates an i nstance of the RBM-class and trains it on the MNIST dataset.
    2) Tests the RBM using test data from the MNIST dataset and reconstructs the data to test the fidelity of the machine.
    3) Uses the RBM to reduce the dimensionality of the MNIST dataset and use the K-means class to cluster hidden units of the RBM into K clusters.
    4) Then we make predictions on test-data and make it back to visible space to see if all the numbers plotted in fact are the same numbers and truly belong to the same cluster.
    """
    start_time = time.time()
    #Train the RBM
    n_hidden = 18;
    R = rbm(nvisible=28*28, nhidden=18*18, nCDsteps = 5, nepochs = 30)
    training_data = np.zeros(28*28, float)
    test_data = np.zeros(np.shape(training_data))
    number_of_training_examples = 1000


    #training_data[:] = trainX[k].flat[:]
    #training_data = make_binary(training_data)
    for k in range(number_of_training_examples):
        print(k)
        R.contrastive_divergence(make_binary(trainX[k].flat[:]))

    elapsed_time = time.time()-start_time
    print("Trained for", elapsed_time, "seconds")

    #Testing
    for k in range(1,10):
        example = np.zeros((28,28),float)
        test_data[:] = testX[k-1].flat[:]
        test_data = make_binary(test_data)

        R.compute_reconstruction(test_data)
        example.flat[:] = R.visibleprob[:]
        plt.subplot(330 + k)
        plt.imshow(example, cmap = plt.get_cmap("gray"))
    plt.figure()
    for k in range(1,10):
        plt.subplot(330 + k)
        plt.imshow(testX[k-1], cmap = plt.get_cmap("gray"))
    plt.show()


    #We now use the RBM for dimensionality reduction
    training_data = np.zeros((10000, n_hidden*n_hidden))
    for i in range(np.shape(training_data)[0]):
        R.compute_hidden(make_binary(trainX[i].flat[:]))
        training_data[i] = R.hiddenprob

    #Then we train our clustering algorithm using the K-means algorithm on the data the RBM spits out.
    Cluster_data = K_means(n_hidden*n_hidden, 10, training_data)
    Cluster_data.train(15)

    #Here we test the combined RBM-K-means algorithm.
    test_data = np.zeros((10000, n_hidden*n_hidden))
    for i in range(np.shape(test_data)[0]):
        R.compute_hidden(make_binary(testX[i].flat[:]))
        test_data[i] = R.hiddenprob

    Cluster_data.predict(test_data)
    Image = np.zeros((9,28,28))
    images = Cluster_data.clustered_data["5"]
    for i in range(9):
        R.compute_visible(images[i])
        Image[i].flat[:] = R.visibleprob

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(Image[i], cmap = plt.get_cmap("gray"))

    plt.savefig("reconstructions_after_" + str(number_of_training_examples) + "_training_examples.png")
    plt.show()
#test_func_RBM_Kmeans()



def test_func3():
    """
    Test function for an alternative implementation of the RBM.
    """
    R = rbm2(nvisible=28*28, nhidden=18*18, nCDsteps = 25, nepochs = 15)
    training_data = np.zeros((60000, 28*28))
    #Fill the training_data array with flattened images.
    for k in range(60000):
        training_data[k] = trainX[k].flat[:]

    #Trains the RBM
    R.contrastive_divergence(training_data = training_data, number_of_training_examples = 1000)

    #Tests the RBM
    for k in range(9):
        plt.subplot(330 + 1 + k)
        plt.imshow(testX[k], plt.get_cmap("gray"))
    plt.title("Actual images")
    plt.figure()
    for k in range(9):
        image = np.zeros((28,28))
        test_data = make_binary(testX[k].flat[:])
        R.compute_reconstruction(test_data)
        image.flat[:] = R.visibleprob
        plt.subplot(330 + 1 + k)
        plt.imshow(image, cmap = plt.get_cmap("gray"))
    plt.title("Reconstructed images")
    plt.show()


#test_func3()
