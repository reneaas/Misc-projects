### Restricted Boltzmann Machine in C++

To run the codes, you may find it most useful to run

```terminal
python main.py
```
in a standard linux terminal. You'll be prompted with

```terminal
Train model? y/n
```

Choosing y will reset the model parameters and train the model on a predefined dataset of handwritten digits from MNIST. Once training is done, the trained model parameters will be written to file and read by the python script. These will in turn call upon the python version of the RBM and reconstruct a test image a plot will be shown to the screen. If you type n, the script will read model parameters of an earlier trained model and plot a reconstructed image to screen along with the original.
