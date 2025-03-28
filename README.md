# Neural Network for XOR Problem

## Overview
This repository implements a lightweight, NumPy-based feedforward neural network that solves the classic XOR problem. The project demonstrates core deep learning concepts with clean, educational code.

## Key Features
- **Pure NumPy implementation** - No TensorFlow/PyTorch dependencies
- **Interactive visualizations** - Real-time loss curves and decision boundaries
- **Modular design** - Easily extendable to other logic gates
- **Educational focus** - Detailed comments and math explanations

## Technical Specifications
| Component          | Implementation Details                 |
|--------------------|----------------------------------------|
| Framework          | NumPy 1.21+                            |
| Network Topology   | 2-4-1 (configurable)                   |
| Activation         | Sigmoid (σ(z) = 1/(1+e⁻ᶻ))             |
| Loss Function      | Binary Cross-Entropy                   |
| Optimization       | Gradient Descent (η=0.1)               |

## Installation
```bash
git clone https://github.com/jkdu-O2/Neural-Network-Model.git
cd Neural-Network-Model
pip install numpy matplotlib
```
## Basic Training
```bash
from model import NeuralNetwork
import numpy as np

# Initialize network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# XOR dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# Train the model
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Make predictions
print(nn.predict([[0,1]]))  # Expected output: ~0.98
```
