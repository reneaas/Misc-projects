

## Object-oriented machine learning algorithms
This repository contains work I did during a summer research job at UiO at CINPLA, summer 2019.
The codes are [here](https://github.com/reneaas/SummerProject2019/tree/master/RestrictedBoltzmannMachine).


### Restricted Boltzmann Machine (RBM)
The RBM is a generative model.
It consists of two layers made up neurons, a visible layer and a hidden layer.
Each neuron in the visible layer can only communicate with each neuron in the hiddel layer and vice-versa.

The RBM can learn a probability distribution if it's given enough samples from said distribution.
It does this by adjusting three parameters based on some simple learning rules; the weights, the visible biases and the hidden biases.
The weights represented how well the neurons communicate with each others. The visible biases are only connected to the visible layer,
the hidden biases are only connected to the hidden layer. Their purpose is to help with activations of neurons in case their weights are zero.

There's alot of mathematical details in this model, so I'll leave the wikipedia page here:

https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine

It's an excellent source on the basic mathematics and how to implement the most basic algorithm: Constrastive Divergence (CD-k).


### What can it do?

The RBM is capable of extracting a complicated probability distribution from a dataset by showing it enough examples. Once learnt, you can do the following (not an exhaustive list):

1) Show it never-before-seen test examples thought to belong to the same distribution and make it reconstruct the test examples. You can think of it this way:
You show the machine a picture and ask it to draw what it "sees".

2) Give it a test example and make it map the visible and hidden units back and forth using the trained model. It can then generate new examples of the example you gave it,
or it might give you an entirely new example. For instance, if you've trained an RBM on the MNIST handwritten digit dataset and show it, let's say, a picture of the number 2,
it may at first construct different examples of the number 2, but after a significant number of iterations the information about the initial example is lost due to the fact that
the map isn't bijective, nor deterministic. Sometimes, then, the RBM might construct an example of an entirely different number, for example the number 5.
Often times, it converges to the number zero which I hypothesize is due to the fact that the zero does not look remotely like any other number and thus is easier for the
RBM to distinguish from the other numbers.

3) You can use the RBM as a tool for dimensionality reduction such that other machine learning algorithms can run much faster due to the fact that the number of needed
computations to be performed can be significantly reduced. Once you've trained said machine learning algorithm, you can then combine the two to solve more complicated problems faster.
For instance, you can use the RBM to reduce a datasets dimensionality, that is, produce a new dataset. Then you can use this newly made dataset and, say, cluster it into K clusters
using the K-means algorithm. Once finished you can then use this chain to map data back and forth, predicting which cluster it belongs to and so on.

### Classes


#### Restricted Boltzmann Machine: [rbm.py](https://github.com/reneaas/SummerProject2019/blob/master/RestrictedBoltzmannMachine/rbm.py)
The class rbm implements a simple Restricted Boltzmann Machine, see the .py file for documentation on how to format the training- and test data.

#### Clustering algorithms: [clustering.py](https://github.com/reneaas/SummerProject2019/blob/master/RestrictedBoltzmannMachine/clustering.py)
Implements two different clustering algorithms:

1. K-means algorithm
Implements the simplest K-means clustering algorithm. See the .py file on how to format the training- and test data.

2. Neural Network K-means algorithm
Implements a K-means algorithm bases in a neural network. See the .py file on how to format the traning- and test data.


#### Test file: [testing.py](https://github.com/reneaas/SummerProject2019/blob/master/RestrictedBoltzmannMachine/testing.py)
Includes several examples on the usage of the classes used on the MNIST dataset of handwritten digits. See the .py file for documentation.
