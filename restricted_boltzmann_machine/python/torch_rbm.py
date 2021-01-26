import torch
import numpy as np
from progress.bar import Bar, IncrementalBar
import matplotlib.pyplot as plt
import opt_einsum

class rbm(object):
    def __init__(self, n_visible, n_hidden, eta, mom, nCDsteps, epochs, batch_sz, sz_of_dataset):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.eta = eta
        self.mom = mom                                                  #Parameter in the learning rule.
        self.nCDsteps = nCDsteps                                        #Number of CD-steps.
        self.epochs = epochs                                            #Number of epochs per training examples.
        self.batch_sz = batch_sz
        self.sz_of_dataset = sz_of_dataset

        self.vis_prob = torch.zeros(n_visible)
        self.vis_act = torch.zeros(n_visible)
        self.hid_prob = torch.zeros(n_hidden)                             #Stores an empty array for the hidden probabilities.
        self.hid_act = torch.zeros(n_hidden)                              #Stores an empty array for the hidden activations.


        self.weights = torch.randn(n_visible, n_hidden)*0.01
        self.vis_bias = torch.randn(n_visible)*0.01
        self.hid_bias = torch.randn(n_hidden)*0.01
        self.loss = torch.zeros(epochs)
        self.data_mat = torch.zeros([self.sz_of_dataset, self.n_visible])

    def sigmoid(self, x):
        # return torch.nn.Sigmoid(x)
        return 1./(1. + torch.exp(-x))


    def compute_visible(self, hidden):
        u = torch.rand(self.n_visible)
        # self.vis_prob = self.sigmoid(self.vis_bias + torch.matmul(self.weights, hidden))
        self.vis_prob = self.sigmoid(self.vis_bias + torch.einsum("ij,kj->ki", self.weights, hidden))
        self.vis_act = 1.*(self.vis_prob > u)

    def compute_hidden(self, visible):
        u = torch.rand(self.n_hidden)
        # self.hid_prob = self.sigmoid(self.hid_bias + torch.matmul(self.weights.T, visible))
        self.hid_prob = self.sigmoid(self.hid_bias + torch.einsum("ij,ki->kj", self.weights, visible))
        self.hid_act = 1.*(self.hid_prob > u)

    def train_model(self, input_data):
        self.data_mat = torch.as_tensor(input_data, dtype=torch.float32)
        N = self.n_visible
        dW = torch.zeros((self.batch_sz, self.n_visible,self.n_hidden))
        dvb = torch.zeros([self.batch_sz, self.n_visible])
        dhb = torch.zeros([self.batch_sz, self.n_hidden])

        bar = IncrementalBar("Progress", max = self.epochs)  #Sets up the progressbar.
        for epoch in range(self.epochs):
            bar.next()

            idx = torch.randperm(self.batch_sz)
            training_data = self.data_mat[idx]
            error = 0.;

            self.compute_hidden(training_data)
            self.compute_visible(self.hid_act)

            CDpos = torch.einsum("ki,kj->kij", training_data, self.hid_prob)


            CDpos_vb = training_data
            CDpos_hb = self.hid_prob

            for j in range(self.nCDsteps):
                self.compute_visible(self.hid_act)
                self.compute_hidden(self.vis_act)

            CDneg = torch.einsum("ki,kj->kij", self.vis_act, self.hid_prob)
            CDneg_vb = self.vis_act
            CDneg_hb = self.hid_prob

            tmp = training_data - self.vis_act

            error = np.einsum("ij,ij->", tmp, tmp)/self.batch_sz
            self.loss[epoch] = error

            dW = self.eta*(CDpos - CDneg)/self.batch_sz + self.mom*dW
            self.weights = torch.einsum("ijk->jk",dW)
            dvb = self.eta*(CDpos_vb - CDneg_vb)/self.batch_sz + self.mom*dvb
            self.vis_bias = torch.einsum("ij->j", dvb)
            dhb = self.eta*(CDpos_hb - CDneg_hb)/self.batch_sz + self.mom*dhb
            self.hid_bias = torch.einsum("ij->j", dhb)
        bar.finish()

    def predict(self, input):
        input = torch.as_tensor(input, dtype=torch.float32)
        self.compute_hidden(input)
        self.compute_visible(self.hid_prob)
        for k in range(self.nCDsteps):
            self.compute_hidden(self.vis_act)
            self.compute_visible(self.hid_prob)

    def plot_loss(self):
        epochs = np.linspace(0, self.epochs, self.epochs)
        plt.plot(epochs, self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()
