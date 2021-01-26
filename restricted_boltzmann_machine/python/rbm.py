#Necessary imports.
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar, IncrementalBar

class rbm:
    """
    This class implement a simple restricted boltzmann machine (RBM) using both learning rate (eta) and momentum to update its parameters;
    weights, visiblebias and hiddenbias, to learn the RBM features contained in the sample distribution it's supposed to learn.
    """
    def __init__(self, nvisible, nhidden, eta, momentum, nCDsteps, nepochs, batch_size, size_of_dataset):
        """
        Implements a simple Restricted Boltzmann Machine.


        nvisible: number of visible nodes
        nhidden: number of hidden nodes.
        eta: Learning rate; set 0.1 <= eta <= 0.5
        momentum: set 0.1 <= momentum <= 0.9.
        nCDsteps: number of iterations in the contrastive divergence algorithm.
        nepochs: number of epochs
        """
        self.nvisible = nvisible
        self.nhidden = nhidden
        self.eta = eta
        self.momentum = momentum                                        #Parameter in the learning rule.
        self.nCDsteps = nCDsteps                                        #Number of CD-steps.
        self.nepochs = nepochs                                          #Number of epochs per training examples.
        self.batch_size = batch_size
        self.size_of_dataset = size_of_dataset
        self.visibleprob = np.zeros(nvisible)                           #Stores an empty array for the visible probabilities.
        self.visibleact = np.zeros(nvisible)                            #Stores an empty array for the activations of the visible nodes.
        self.hiddenprob = np.zeros(nhidden)                             #Stores an empty array for the hidden probabilities.
        self.hiddenact = np.zeros(nhidden)                              #Stores an empty array for the hidden activations.

        #Defines the parameters in the model.
        self.weights = np.random.randn(nvisible, nhidden)*0.01          #Initial weights
        self.visiblebias = np.random.randn(nvisible)*0.01               #Initial visible bias
        self.hiddenbias = np.random.randn(nhidden)*0.01                 #Initial hidden bias
        self.loss = np.zeros(self.nepochs)
        self.data_matrix = np.zeros((self.size_of_dataset, self.nvisible))

    def sigmoid(self, x):
        """Activation function"""
        return 1./(1. + np.exp(-x))


    def compute_visible(self,hidden):
        """Computes P = p(v[i]|h) = 1 for i = 1,2,...,nvisible, and activates node i given that P > u,
        where u is a random number between 0 and 1 sampled from a uniform distribution
        -----------------------------------------Variables-------------------------------------------
        hidden: an array containing the activations of the hidden nodes.
        """
        u = np.random.uniform(0,1.0,self.nvisible)
        self.visibleprob = self.sigmoid(self.visiblebias + np.dot(self.weights, hidden))
        self.visibleact = self.visibleprob > u
        return [self.visibleprob, self.visibleact]

    def compute_hidden(self, visible):
        """Computes P = p(h[j]|v) = 1 for j = 1,2,...,nhidden, and activates node j given that p > u,

        -----------------------------------------Variables-------------------------------------------
        visible: array containing the activations of the visible nodes.
        """
        u = np.random.uniform(0,1,self.nhidden)
        self.hiddenprob = self.sigmoid(self.hiddenbias + np.dot(self.weights.T, visible))
        self.hiddenact = self.hiddenprob > u
        return [self.hiddenprob, self.hiddenact]


    def train_model(self, input_data):
        """
        An implementation of the CD-n algorithm.


        inputs: training data should be a vector V of the same shape as v = np.zeros(nvisible)
        """

        self.data_matrix = input_data
        N = self.nvisible                                    #Scaling factor used in the learning process.
        #Creates empty arrays to store "differentials".
        dW = np.zeros((self.nvisible,self.nhidden))
        dvb = np.zeros(self.nvisible)
        dhb = np.zeros(self.nhidden)

        bar = IncrementalBar("Progress", max = self.nepochs)  #Sets up the progressbar.
        #Trains the RBM using the CD-n algorithm on a single datapoint at the time.
        for epoch in range(self.nepochs):
            bar.next()
            shuffled_indices = np.random.permutation(self.batch_size)
            training_data = self.data_matrix[shuffled_indices]
            error = 0
            for k in range(self.batch_size):
                visible = training_data[k]
                #sample hidden variables
                self.compute_hidden(visible)

                #compute <vh>_0
                CDpos = np.tensordot(visible, self.hiddenprob, axes = 0)     #Tensor product computes a matrix of shape (nvisible x nhidden)
                CDpos_vb = visible                                          #Simply the initial state of the visible nodes.
                CDpos_hb = self.hiddenprob                                  #The first computed state of the hidden nodes.

                #CD-n, if nCDsteps = 1, this is essentially just reconstrunction of the input.
                #Choosing nCDsteps = 1 works alright and is computationally effective.
                for j in range(self.nCDsteps):
                    self.compute_visible(self.hiddenact)
                    self.compute_hidden(self.visibleact)
                #self.compute_visible(self.hiddenprob)
                #self.compute_hidden(self.visibleact)

                #Computes <vh>_n
                CDneg = np.tensordot(self.visibleact, self.hiddenprob, axes = 0)
                CDneg_vb = self.visibleact
                CDneg_hb = self.hiddenprob

                #This is where the learning happens, you can skip the momentum if you want but it speeds up initial learning
                #You can modifiy the class to add decay, that is add -self.decay*dW to the learning rule, or reduce the momentum towards the end of learning.

                #Reconstruction error. It measures how well the RBM reconstructs the data it's shown.
                visible = training_data[k]
                error += np.sum((self.data_matrix[k]-self.visibleact)**2)
                dW = self.eta*(CDpos - CDneg)/N + self.momentum*dW
                self.weights += dW
                dvb = self.eta*(CDpos_vb - CDneg_vb)/N + self.momentum*dvb
                self.visiblebias += dvb
                dhb = self.eta*(CDpos_hb - CDneg_hb)/N + self.momentum*dhb
                self.hiddenbias += dhb
            error /= self.batch_size
            self.loss[epoch] = error
        bar.finish()


    def predict(self, input):
        self.compute_hidden(input)
        self.compute_visible(self.hiddenprob)
        for k in range(self.nCDsteps):
            self.compute_hidden(self.visibleact)
            self.compute_visible(self.hiddenprob)

    def plot_loss(self):
        epochs = np.linspace(0, self.nepochs, self.nepochs)
        plt.plot(epochs, self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
