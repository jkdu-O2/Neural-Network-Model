import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
import time

# === Neural Network ===
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_type="sigmoid", 
                 learning_rate=0.1, momentum=0.9, init_method="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation_type.lower()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.init_method = init_method.lower()
        self.set_hidden_layers(hidden_sizes)
        self.velocity_w = [None] * len(self.weights)
        self.velocity_b = [None] * len(self.biases)

    def set_hidden_layers(self, hidden_sizes):
        """Initialize network with Xavier/Glorot initialization"""
        self.hidden_sizes = hidden_sizes
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i+1]
            
            if self.init_method == "xavier":
                # Xavier/Glorot initialization
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                self.weights.append(np.random.randn(fan_in, fan_out) * scale)
            else:  # Standard initialization
                self.weights.append(np.random.randn(fan_in, fan_out) * 0.1)
                
            self.biases.append(np.zeros((1, fan_out)))
        
        # Reset training history
        self.loss_history = []
        self.accuracy_history = []
        self.training_time = []
        self.epoch_times = []

    def reset(self):
        """Reinitialize network with current architecture"""
        self.set_hidden_layers(self.hidden_sizes)

    def activation_function(self, x, derivative=False):
        """Sigmoid or ReLU activation with derivative support"""
        if self.activation_type == "sigmoid":
            if derivative:
                return x * (1 - x)  # x should be the output of sigmoid
            return 1 / (1 + np.exp(-x))
        elif self.activation_type == "relu":
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)

    def forward_propagation(self, inputs):
        """Forward pass through all layers"""
        layer_outputs = [inputs]
        for weight, bias in zip(self.weights, self.biases):
            inputs = self.activation_function(np.dot(inputs, weight) + bias)
            layer_outputs.append(inputs)
        return layer_outputs

    def predict(self, inputs):
        """Get final network predictions"""
        return self.forward_propagation(inputs)[-1]

    def backward_propagation(self, inputs, targets, layer_outputs):
        """Backpropagation with momentum"""
        output = layer_outputs[-1]
        errors = targets - output
        delta = errors * self.activation_function(output, derivative=True)
        
        for i in range(len(self.weights) - 1, -1, -1):
            inputs_T = layer_outputs[i].T if i > 0 else inputs.T
            
            # Calculate weight updates with momentum
            weight_update = inputs_T.dot(delta)
            if self.velocity_w[i] is None:
                self.velocity_w[i] = weight_update
            else:
                self.velocity_w[i] = self.momentum * self.velocity_w[i] + (1 - self.momentum) * weight_update
                
            bias_update = np.sum(delta, axis=0, keepdims=True)
            if self.velocity_b[i] is None:
                self.velocity_b[i] = bias_update
            else:
                self.velocity_b[i] = self.momentum * self.velocity_b[i] + (1 - self.momentum) * bias_update
            
            # Apply updates
            self.weights[i] += self.learning_rate * self.velocity_w[i]
            self.biases[i] += self.learning_rate * self.velocity_b[i]
            
            # Propagate error backward if not in first layer
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.activation_function(layer_outputs[i], derivative=True)

    def compute_accuracy(self, predictions, targets):
        """Calculate classification accuracy"""
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes == targets)

    def train(self, inputs, targets, epochs):
        """Train the network for given epochs"""
        self.loss_history = []
        self.accuracy_history = []
        self.epoch_times = []
        
        # Normalize inputs (helps with convergence)
        inputs = (inputs - np.mean(inputs, axis=0)) / (np.std(inputs, axis=0) + 1e-8)
        
        for epoch in range(epochs + 1):
            t0 = time.time()
            
            # Forward and backward pass
            layer_outputs = self.forward_propagation(inputs)
            self.backward_propagation(inputs, targets, layer_outputs)
            
            # Calculate metrics
            output = layer_outputs[-1]
            loss = np.mean((targets - output) ** 2)  # MSE loss
            accuracy = self.compute_accuracy(output, targets)
            
            t1 = time.time()
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self.epoch_times.append(t1 - t0)
            
            # Progress reporting
            if epoch % 100 == 0 or epoch == epochs:
                print(f"Epoch {epoch:4d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}, Time = {t1-t0:.4f}s")

# === GUI Layer ===
class InteractiveNNGUI:
    def __init__(self):
        # XOR dataset
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.Y = np.array([[0], [1], [1], [0]])
        
        # Default network parameters
        self.input_size = 2
        self.hidden_sizes = [4]  # 2-4-1 architecture (17 parameters)
        self.output_size = 1
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.epochs = 10000
        self.activation = "sigmoid"
        
        # Initialize network
        self.nn = NeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            activation_type=self.activation,
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )
        
        self.setup_gui()

    def setup_gui(self):
        """Create interactive GUI with controls"""
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        plt.suptitle("Neural Network XOR Classifier (2-4-1 Architecture)", fontsize=14)
        gs = gridspec.GridSpec(3, 2, height_ratios=[3, 0.5, 0.5], width_ratios=[1, 1])

        # Create plots
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_boundary = self.fig.add_subplot(gs[0, 1])
        
        # Control panel
        control_panel = self.fig.add_subplot(gs[1:, :])
        control_panel.axis('off')
        
        # Sliders
        self.slider_lr = Slider(
            plt.axes([0.15, 0.25, 0.35, 0.03]), 
            'Learning Rate', 0.001, 1.0, valinit=self.learning_rate
        )
        self.slider_momentum = Slider(
            plt.axes([0.15, 0.20, 0.35, 0.03]), 
            'Momentum', 0.0, 0.99, valinit=self.momentum
        )
        self.slider_epochs = Slider(
            plt.axes([0.15, 0.15, 0.35, 0.03]), 
            'Epochs', 100, 20000, valinit=self.epochs, valstep=100
        )
        
        # Dropdown (simulated with buttons)
        self.activation_btn = Button(
            plt.axes([0.55, 0.25, 0.15, 0.05]), 
            f'Activation: {self.activation.capitalize()}'
        )
        
        # Textbox and buttons
        self.textbox = TextBox(
            plt.axes([0.55, 0.15, 0.15, 0.05]), 
            'Hidden Layers', initial="4"
        )
        self.train_btn = Button(
            plt.axes([0.75, 0.15, 0.1, 0.05]), 
            'Train', color='lightgoldenrodyellow'
        )
        self.reset_btn = Button(
            plt.axes([0.75, 0.25, 0.1, 0.05]), 
            'Reset', color='lightcoral'
        )

        # Event handlers
        self.slider_lr.on_changed(self.update_learning_rate)
        self.slider_momentum.on_changed(self.update_momentum)
        self.slider_epochs.on_changed(self.update_epochs)
        self.activation_btn.on_clicked(self.toggle_activation)
        self.textbox.on_submit(self.update_hidden_layers)
        self.train_btn.on_clicked(self.start_training)
        self.reset_btn.on_clicked(self.reset_network)

        self.initialize_plots()

    def toggle_activation(self, event):
        """Switch between sigmoid and ReLU activation"""
        self.activation = "relu" if self.activation == "sigmoid" else "sigmoid"
        self.activation_btn.label.set_text(f'Activation: {self.activation.capitalize()}')
        self.nn.activation_type = self.activation

    def update_learning_rate(self, val):
        self.learning_rate = val
        self.nn.learning_rate = val

    def update_momentum(self, val):
        self.momentum = val
        self.nn.momentum = val

    def update_epochs(self, val):
        self.epochs = int(val)

    def update_hidden_layers(self, text):
        try:
            self.hidden_sizes = list(map(int, text.strip().split(",")))
            print(f"Updated architecture to: {self.input_size}-{'x'.join(map(str, self.hidden_sizes))}-{self.output_size}")
        except ValueError:
            print("Please enter comma-separated integers (e.g. '4,3')")

    def reset_network(self, event):
        """Reinitialize network with current parameters"""
        self.nn = NeuralNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            output_size=self.output_size,
            activation_type=self.activation,
            learning_rate=self.learning_rate,
            momentum=self.momentum
        )
        print("Network reset with current parameters")

    def start_training(self, event):
        """Train the network and update visualizations"""
        self.train_btn.label.set_text("Training...")
        self.train_btn.color = 'yellow'
        plt.pause(0.1)  # Allow GUI to update
        
        try:
            print(f"\nStarting training with:")
            print(f"- Architecture: {self.input_size}-{'x'.join(map(str, self.hidden_sizes))}-{self.output_size}")
            print(f"- Learning rate: {self.learning_rate:.3f}, Momentum: {self.momentum:.2f}")
            print(f"- Activation: {self.activation}, Epochs: {self.epochs}")
            
            self.nn.train(self.X, self.Y, self.epochs)
            self.update_plots()
            
            final_loss = self.nn.loss_history[-1]
            final_acc = self.nn.accuracy_history[-1]
            print(f"\nTraining complete!")
            print(f"Final loss: {final_loss:.6f}, Accuracy: {final_acc:.4f}")
            print(f"Total training time: {sum(self.nn.epoch_times):.2f} seconds")
            
        finally:
            self.train_btn.label.set_text("Train")
            self.train_btn.color = 'lightgoldenrodyellow'
            plt.pause(0.1)

    def initialize_plots(self):
        """Set up initial plot appearance"""
        for ax in [self.ax_loss, self.ax_boundary]:
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        self.ax_loss.set_title("Training Metrics")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Value")
        
        self.ax_boundary.set_title("Decision Boundary")
        self.ax_boundary.set_xlabel("Input 1")
        self.ax_boundary.set_ylabel("Input 2")

    def update_plots(self):
        """Update all visualizations with current network state"""
        # Loss and Accuracy
        self.ax_loss.clear()
        self.ax_loss.plot(self.nn.loss_history, label='Loss (MSE)', color='red')
        self.ax_loss.plot(self.nn.accuracy_history, label='Accuracy', color='blue')
        self.ax_loss.legend()
        self.ax_loss.set_ylim(0, 1.1)
        
        # Decision Boundary
        self.plot_decision_boundary()
        
        plt.draw()

    def plot_decision_boundary(self):
        """Visualize the network's decision boundaries"""
        self.ax_boundary.clear()
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                           np.linspace(y_min, y_max, 100))
        Z = self.nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Decision boundary
        self.ax_boundary.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
        # Training points
        self.ax_boundary.scatter(self.X[:, 0], self.X[:, 1], c=self.Y.ravel(), 
                               cmap="RdBu", edgecolors='k', s=100)
        self.ax_boundary.set_title(f"Decision Boundary ({self.activation.capitalize()} Activation)")

# === Main Execution ===
if __name__ == "__main__":
    print("XOR Neural Network Classifier")
    print("Key Features:")
    print("- 2-4-1 architecture (17 trainable parameters)")
    print("- Xavier initialization + data normalization")
    print("- Momentum-accelerated backpropagation")
    print("- Supports both sigmoid and ReLU activations")
    
    gui = InteractiveNNGUI()
    plt.show(block=True)
