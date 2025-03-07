# Overview
This is a simple feedforward neural network implemented using NumPy to solve the XOR problem. The network consists of:

A configurable number of hidden layers
Sigmoid activation function
Forward and backward propagation
Gradient descent optimization

# Requirements
Python 3.x
NumPy

# Installation
Clone the repository:
git clone https://github.com/yourusername/neural-network-xor.git
cd neural-network-xor
Install dependencies (if needed):
pip install numpy

# Usage
Run the script:
python neural_network.py
This will train the network for 10,000 epochs with a learning rate of 0.1.

# Customization
You can modify:
The number of hidden layers and neurons
Learning rate
Number of training epochs

Example:
nn = NeuralNetwork(inputLayerSize=2, hiddenLayerSize=[6], outputLayerSize=1)
nn.train(X_preprocess, Y, epochs=5000, learningRate=0.05)


