import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


"""
Both classes here were written to cluster lower dimensional data produced by the rbm class in the file 'rbm.py'. However, both classes are written generally 
such that they can be used with whatever data you wish to cluster. Of course, they require you to format the data you want to cluster in the way the classes asks you to, 
but this is specified in the __init__ methods in both classes and thus should be a straight forward manner should you desire to cluster an arbitrary set of data.

"""

class K_means:
    """
    Implements the k-means clustering algorithm.
    """
    def __init__(self, dataset_dimension, k, dataset):
        """
        k: number of clusters 
        dataset_dimension: the dimension of the dataset
        dataset: the dataset to be clustered in k clusters. The dataset should be an N x M matrix where N is the number of datapoints and M is the dimension of each datapoint.
        """
        self.k = k
        self.dataset_dimension = dataset_dimension
        self.dataset = dataset
        self.centroids = np.random.randn(k, dataset_dimension)                      #randomly initialized centroids.
        self.old_centroids = np.zeros(np.shape(self.centroids))                     #will be used to store old centroids to facilitate error computations.
        self.distance_to_centroids = np.zeros(k)                                    #Empty distance array to contain the distances from each cluster to a single datapoint at a time.
        self.N  = np.shape(self.dataset)[0]                                         #The number of datapoints in the dataset.


    def assign_to_cluster(self):
        """
        Assigns each datapoint to a cluster.
        """
        self.clustered_data = {str(i):[] for i in range(10)}                                                                                    #Creates an empty dict with a unique key for each cluster.
        for i in range(self.N):
            for j in range(self.k):
                self.distance_to_centroids[j] = np.sqrt(np.dot(self.dataset[i] - self.centroids[j], self.dataset[i] - self.centroids[j]))       #Computes the distance from the datapoint to each cluster
            belongs_to_cluster_number = np.argmin(self.distance_to_centroids)                                                                   #Picks out the cluster which the datapoint is closest to.
            key = str(belongs_to_cluster_number)
            self.clustered_data[key].append(self.dataset[i])                                                                                    #Appends the datapoint to the cluster it 'belongs to'.


    def assign_to_cluster_test(self, test_data, number_of_test_objects):
        """
        Assigns each datapoint to a cluster. Does essentially the same as the method: assing_to_cluster, but on a test_data set instead.
        """
        self.clustered_data = {str(i):[] for i in range(10)}                                                                            #Creates an empty dict with a unique key for each cluster.
        for i in range(number_of_test_objects):
            for j in range(self.k):
                self.distance_to_centroids[j] = np.sqrt(np.dot(test_data[i] - self.centroids[j],test_data[i] - self.centroids[j]))      #Computes the cistance from the datapoint to each cluster. 
            belongs_to_cluster_number = np.argmin(self.distance_to_centroids)                                                           #Picks out the cluster which the datapoint is closest to.
            key = str(belongs_to_cluster_number)
            self.clustered_data[key].append(test_data[i])                                                                               #Appends the datapoint to the cluster it belongs to.

    def re_adjust_centroids(self):
        """
        Updates the centroid-vectors
        """
        self.old_centroids = self.centroids.copy()                              #Stores the old centroids.
        for j in range(self.k):
            key = str(j)
            self.centroids[j,:] = 0                                             #reset the centroids to compute it anew from the datapoints assigned to each cluster.
            for i in range(len(self.clustered_data[key])):
                self.centroids[j] += self.clustered_data[key][i]                #Computes the new centroids by taking the average of the datapoints assigned to the cluster j.
            if len(self.clustered_data[key]) != 0:
                self.centroids[j] /= len(self.clustered_data[key])              #Just a safety measure in case no datapoints are assigned to the cluster.

    def train(self, max_number_of_epochs):
        """
        Trains the computer on the dataset

        max_number_of_epochs: number of epochs you want to train the machine.
        """
        error = np.linalg.norm(self.centroids - self.old_centroids)                             #Necessary if you want to include an error requirement in the while loop.
        epoch_number = 0                                                                        
        while epoch_number <=  max_number_of_epochs:
            epoch_number += 1                                                                   #Keeps track of which epoch the machine is at    
            self.assign_to_cluster()                                                            #Assigns the datapoints to their respective clusters
            self.re_adjust_centroids()                                                          #Computes the new centroids based on the mean of the datapoints pertaining to each cluster.
            error = np.linalg.norm(self.centroids - self.old_centroids)                         #Computes how far the centroids move each epoch.
            print("epochs:", epoch_number, " error:", error)                                    #Lets us know how it's doing, the idea is that the centroids should move less and less. 


    def predict(self, test_data):
        """
        Clusters never before seen data into K clusters.
        """
        test_data_dimensions = np.shape(test_data)                                              #Extracts the shape of the test_data. 
        N = test_data_dimensions[0]                                                             #Number of test examples
        self.assign_to_cluster_test(test_data, N)                                               #Assigns each test example to the learned clusters.



class K_means_NN():
    """
    This is a class that implements a version of the K-means clustering algorithm based on a neural network consisting of two layers, 
    a visible layer consisting of the data to be placed in its respective cluster and a hidden layer made up of K neurons corresponding to the K clusters the data 
    is to be clusted into. 

    I still haven't done any quantitative measure of this algorithms efficiency relative to the regular K-means algorithm.
    """
    def __init__(self, Kneurons, dataset):
        """
        Kneruons:(K number of neurons)
        dataset: the dataset, should be a two dimensional array as a (N x M)-matrix containing N test examples of length M.
        """
        self.Kneurons = Kneurons;                                                       #The number of neurons, K in total. Corresponds to the number of clusters in the data.
        self.dataset = dataset;                                                         #(N x M)-matrix containing the training data.
        self.nexamples = np.shape(self.dataset)[0]                                      #N training examples.
        self.length_of_examples = 28*28                                                 #The length M of the training examples.
        self.neurons = np.zeros(self.Kneurons)                                          #The K neurons stored in an array of K elements, each element corresponding the activations of the neuron.
        self.weights = np.random.randn(self.Kneurons, self.length_of_examples)*0.1      #The weight matrix , a (K x M)-matrix. This is the parameter adjusted by the learning algorithm.
        self.eta = 0.1                                                                  #Learning rate (or step size if you will).


    def train(self, nepochs):
        """
        Trains the neural network to cluster the data.
        """
        #Normalizes the dataset to ensure it lies on the unit sphere. This is necessary to use this particular algorithm.
        for i in range(self.nexamples):
            sum_of_example = np.sum(self.dataset[i])
            self.dataset[i] /= sum_of_example


        #Compute the activations and update the weights.
        epoch = 0
        self.activations = np.zeros(self.Kneurons)                                                                  #Creates an empty activation array to store the activation of each neuron.
        while epoch < nepochs:
            print("epoch:", epoch)
            epoch += 1
            for e in range(self.nexamples):
                for i in range(self.Kneurons):
                    self.activations[i] = np.dot(self.weights[i].T, self.dataset[e])
                activated_neuron = np.argmax(self.activations)                                                      #The neuron with the highest linear activation wins.
                self.weights[activated_neuron,:] += self.eta*(self.dataset[e,:] - self.weights[activated_neuron,:]) #Update the weights based on the activated neuron.



    def predict(self, test_datapoint):
        """
        Predicts which cluster a test datapoint belongs to based on its training.
        test_datapoint: Should be an array of length M, the same length as the training data.
        """
        for i in range(self.Kneurons):
            self.activations[i] = np.dot(self.weights[i].T, test_datapoint)                                     #Computes the linear activation of each neuron.
        activated_neuron = np.argmax(self.activations)                                                          #Chooses the winning neuron as the predicted cluster.
        return activated_neuron
