#Necessary imports.
import numpy as np

class rbm:
    """
    This class implement a simple restricted boltzmann machine (RBM) using both learning rate (eta) and momentum to update its parameters;
    weights, visiblebias and hiddenbias, to learn the RBM features contained in the sample distribution it's supposed to learn. 
    I've mainly applied it to the MNIST dataset of handwritten digits and it reconstructs the images very well after rougly 10000 training examples. 
    It does well after less training examples as well (say 1000), but the fidelity of the reconstructions aren't as good. 
    
    Some advice based on my experience, the baseline for the learning rate and momentum set in the class works well. Furthermore CD-25 (that is nCDsteps = 25) is sufficient to train the algorithm. 
    To speed up the algorithm, use nCDsteps = 1, it works really well too, but you might need more training examples. Usually I've used nepochs = 15 - 30 as that is usually enough. 
    The number of hidden units, nhidden, is also important. If you use nhidden ~ nvisible, you'll get better results on the reconstructions but you will the training will take much longer. 
    with nvisible = 28*28 you can get adequate results with nhidden = 8*8 on the MNIST dataset suggesting that the dimensionality of the dataset can be lowered, but the reconstructions will suffer a
    little bit. 

    """
    def __init__(self, nvisible, nhidden, eta=0.1, momentum = 0.7, nCDsteps = 25, nepochs = 1000):
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
        self.visibleprob = np.zeros(nvisible)                           #Stores an empty array for the visible probabilities.
        self.visibleact = np.zeros(nvisible)                            #Stores an empty array for the activations of the visible nodes.
        self.hiddenprob = np.zeros(nhidden)                             #Stores an empty array for the hidden probabilities.
        self.hiddenact = np.zeros(nhidden)                              #Stores an empty array for the hidden activations.

        #Defines the parameters in the model.
        self.weights = np.random.randn(nvisible, nhidden)*0.01          #Initial weights
        self.visiblebias = np.random.randn(nvisible)*0.01               #Initial visible bias
        self.hiddenbias = np.random.randn(nhidden)*0.01                 #Initial hidden bias.


    def sigmoidal(self, x):
        """Activation function"""
        return 1./(1. + np.exp(-x))


    def compute_visible(self,hidden):
        """Computes P = p(v[i]|h) = 1 for i = 1,2,...,nvisible, and activates node i given that P > u,
        where u is a random number between 0 and 1 sampled from a uniform distribution
        -----------------------------------------Variables-------------------------------------------
        hidden: an array containing the activations of the hidden nodes.
        """
        u = np.random.uniform(0,1.0,self.nvisible)                          
        self.visibleprob = self.sigmoidal(self.visiblebias + np.dot(self.weights, hidden))
        self.visibleact = self.visibleprob > u
        return [self.visibleprob, self.visibleact]

    def compute_hidden(self, visible):
        """Computes P = p(h[j]|v) = 1 for j = 1,2,...,nhidden, and activates node j given that p > u,
        where u is a random number between 0 and 1 sampled from a uniform distribution.
        -----------------------------------------Variables-------------------------------------------
        visible: array containing the activations of the visible nodes.
        """
        u = np.random.uniform(0,1,self.nhidden)
        self.hiddenprob = self.sigmoidal(self.hiddenbias + np.dot(self.weights.T, visible))
        self.hiddenact = self.hiddenprob > u
        return [self.hiddenprob, self.hiddenact]


    def contrastive_divergence(self, inputs):
        """
        An implementation of the CD-n algorithm.


        inputs: training data should be a vector V of the same shape as v = np.zeros(nvisible)
        """

        visible = inputs
        N = np.shape(inputs)[0]                                     #Scaling factor used in the learning process.

        #Creates empty arrays to store "differentials".
        dW = np.zeros((self.nvisible,self.nhidden))
        dvb = np.zeros(self.nvisible)
        dhb = np.zeros(self.nhidden)

        #Trains the RBM using the CD-n algorithm on a single datapoint at the time.
        for epoch in np.arange(self.nepochs):
            #sample hidden variables
            self.compute_hidden(visible)

            #compute <vh>_0
            CDpos = np.tensordot(visible, self.hiddenprob, axes = 0)     #Tensor product computes a matrix of shape (nvisible x nhidden)
            CDpos_vb = visible                                          #Simply the initial state of the visible nodes.
            CDpos_hb = self.hiddenprob                                  #The first computed state of the hidden nodes.

            #CD-n, if nCDsteps = 1, this is essentially just reconstrunction of the input.
            #Choosing nCDsteps = 1 works alright and is computationally effective.
            for j in np.arange(self.nCDsteps):
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
            dW = self.eta*(CDpos - CDneg)/N + self.momentum*dW
            self.weights += dW
            dvb = self.eta*(CDpos_vb - CDneg_vb)/N + self.momentum*dvb
            self.visiblebias += dvb
            dhb = self.eta*(CDpos_hb - CDneg_hb)/N + self.momentum*dhb
            self.hiddenbias += dhb

            #Reconstruction error. It measures how well the RBM reconstructs the data it's shown.  
            error = np.sum((inputs-self.visibleact)**2)
            visible = inputs
        #print("Reconstruction error", error)
        #self.compute_hidden(inputs)
        return error

    def compute_reconstruction(self, input):
        self.compute_hidden(input)
        self.compute_visible(self.hiddenprob)
        return None

def make_binary(data):
    """data: an numpy array of elements between 0 and 255 (pixels). Returns a binary version of the array."""
    data = data/255
    for i in np.arange(len(data.flat)):
        if data.flat[i] > 0.5:
            data.flat[i] = 1
        else:
            data.flat[i] = 0
    return data
