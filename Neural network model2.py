import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        """
        Initialize neural network with configurable architecture
        
        Args:
            input_size: Number of input neurons
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases using He initialization
        sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        for i in range(len(sizes)-1):
            limit = np.sqrt(2 / (sizes[i] + sizes[i+1]))
            self.weights.append(np.random.randn(sizes[i], sizes[i+1]) * limit)
            self.biases.append(np.zeros((1, sizes[i+1])))
            
        # Training history storage
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """
        Numerically stable sigmoid activation
        
        Args:
            x: Input array
            derivative: If True, returns derivative
            
        Returns:
            Sigmoid activation or its derivative
        """
        x = np.clip(x, -500, 500)  # Prevent overflow
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig
    
    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Forward propagation through the network
        
        Args:
            X: Input data (m samples x n features)
            
        Returns:
            List of layer activations
        """
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {X.shape[1]}")
            
        activations = [X]
        for w, b in zip(self.weights, self.biases):
            X = self.sigmoid(np.dot(X, w) + b)
            activations.append(X)
        return activations
    
    def backward(self, 
                X: np.ndarray, 
                y: np.ndarray, 
                activations: List[np.ndarray], 
                learning_rate: float) -> None:
        """
        Backpropagation algorithm with gradient descent
        
        Args:
            X: Input data
            y: Target values
            activations: Layer activations from forward pass
            learning_rate: Learning rate for updates
        """
        m = X.shape[0]  # Number of samples
        output = activations[-1]
        
        # Output error
        error = output - y
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradient calculations
            grad_w = np.dot(activations[i].T, error) / m
            grad_b = np.sum(error, axis=0, keepdims=True) / m
            
            # Parameter updates
            self.weights[i] -= learning_rate * grad_w
            self.biases[i] -= learning_rate * grad_b
            
            # Propagate error backward
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.sigmoid(activations[i], derivative=True)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary cross-entropy loss with numerical stability
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted probabilities
            
        Returns:
            Loss value
        """
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Classification accuracy
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities
            
        Returns:
            Accuracy percentage (0-1)
        """
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y_true)
    
    def train(self, 
             X: np.ndarray, 
             y: np.ndarray, 
             epochs: int = 10000, 
             learning_rate: float = 0.1, 
             verbose: bool = True,
             early_stopping: bool = True) -> None:
        """
        Train the neural network with optional early stopping
        
        Args:
            X: Training data
            y: Training labels
            epochs: Maximum training iterations
            learning_rate: Initial learning rate
            verbose: Print progress
            early_stopping: Stop if loss plateaus
        """
        self.loss_history = []
        self.accuracy_history = []
        
        for epoch in range(epochs + 1):
            # Forward and backward pass with learning rate decay
            current_lr = learning_rate * (1 / (1 + 0.01 * epoch))
            activations = self.forward(X)
            self.backward(X, y, activations, current_lr)
            
            # Compute and store metrics
            loss = self.compute_loss(y, activations[-1])
            accuracy = self.compute_accuracy(y, activations[-1])
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            # Early stopping check
            if early_stopping and len(self.loss_history) > 20:
                if abs(np.mean(self.loss_history[-20:-10]) - np.mean(self.loss_history[-10:])) < 1e-6:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress reporting
            if verbose and (epoch % 1000 == 0 or epoch == epochs):
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions from input data
        
        Args:
            X: Input data (m samples x n features)
            
        Returns:
            Predicted probabilities
        """
        return self.forward(X)[-1]
    
    def plot_training_history(self) -> None:
        """
        Plot training loss and accuracy curves
        """
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy")
        
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history)
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundary(self, X: np.ndarray = None, y: np.ndarray = None) -> None:
        """
        Visualize decision boundary for 2D inputs
        
        Args:
            X: Optional input data to plot
            y: Optional labels to plot
        """
        if X is None or y is None:
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y = np.array([[0], [1], [1], [0]])
        
        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        # Predict probabilities
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8, vmin=0, vmax=1)
        plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=50, 
                   cmap="RdBu", edgecolors='white')
        plt.title("Decision Boundary")
        plt.xlabel("Input 1")
        plt.ylabel("Input 2")
        plt.show()


def main():
    """Run XOR classification example"""
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create and train network
    print("Training neural network on XOR problem...")
    nn = NeuralNetwork(input_size=2, hidden_sizes=[4], output_size=1)
    nn.train(X, y, epochs=10000, learning_rate=0.1)
    
    # Visualize results
    print("\nTraining complete. Showing results...")
    nn.plot_training_history()
    nn.plot_decision_boundary()
    
    # Make predictions
    print("\nFinal Predictions:")
    for sample in X:
        pred = nn.predict(sample.reshape(1, -1))[0,0]
        print(f"Input: {sample} -> Prediction: {pred:.4f} ({'✓' if round(pred) == y[np.all(X == sample, axis=1)][0,0] else '✗'})")


if __name__ == "__main__":
    main()
