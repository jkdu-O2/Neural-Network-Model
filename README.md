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
