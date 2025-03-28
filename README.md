# Neural Network for XOR Problem

# Overview
This repository implements a lightweight, NumPy-based feedforward neural network that solves the classic XOR problem. The project demonstrates core deep learning concepts with clean, educational code.

# Key Features
- **Pure NumPy implementation** - No TensorFlow/PyTorch dependencies
- **Interactive visualizations** - Real-time loss curves and decision boundaries
- **Modular design** - Easily extendable to other logic gates
- **Educational focus** - Detailed comments and math explanations

# Technical Specifications
| Component          | Implementation Details                 |
|--------------------|----------------------------------------|
| Framework          | NumPy 1.21+                            |
| Network Topology   | 2-4-1 (configurable)                   |
| Activation         | Sigmoid (σ(z) = 1/(1+e⁻ᶻ))             |
| Loss Function      | Binary Cross-Entropy                   |
| Optimization       | Gradient Descent (η=0.1)               |

# Installation
```bash
git clone https://github.com/jkdu-O2/Neural-Network-Model.git
cd Neural-Network-Model
pip install numpy matplotlib
```
# Basic Training
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
# Visualization
```bash
# Plot training loss
nn.plot_training_loss()

# Visualize decision boundary (requires matplotlib)
nn.plot_decision_boundary()
```
# Mathematical Foundations
## Forward Propagation
```bash
\begin{aligned}
z^{[1]} &= W^{[1]}X + b^{[1]} \\
a^{[1]} &= \sigma(z^{[1]}) \\
z^{[2]} &= W^{[2]}a^{[1]} + b^{[2]} \\
\hat{y} &= a^{[2]} = \sigma(z^{[2]})
\end{aligned}
```
## Backwards propagation
```bash
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{[2]}} &= (a^{[2]} - y) \cdot a^{[1]T} \\
\frac{\partial \mathcal{L}}{\partial W^{[1]}} &= ((W^{[2]T}(a^{[2]} - y)) \odot \sigma'(z^{[1]})) \cdot X^T
\end{aligned}
```
## Loss Function (Binary Cross-Entropy)
```bash
\mathcal{L} = -\frac{1}{m}\sum_{i=1}^m [y^{(i)}\log(a^{[2](i)}) + (1-y^{(i)})\log(1-a^{[2](i)})]
```
# References
```bash
Nielsen, M. A. (2015). Neural Networks and Deep Learning

Goodfellow, I., et al. (2016). Deep Learning (Chapter 6)

CS231n: Backpropagation Notes
```
