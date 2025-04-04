import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
import time

# === Neural Network ===
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation_type="sigmoid"):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation_type
        self.set_hidden_layers(hidden_sizes)

    def set_hidden_layers(self, hidden_sizes):
        self.hidden_sizes = hidden_sizes
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        rng = np.random.default_rng()
        self.weights = [rng.standard_normal((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.biases = [rng.standard_normal((1, size)) for size in sizes[1:]]
        self.loss_history = []
        self.accuracy_history = []
        self.training_time = []

    def reset(self):
        self.set_hidden_layers(self.hidden_sizes)

    def activation_function(self, x, derivative=False):
        if self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-x)) if not derivative else x * (1 - x)
        elif self.activation_type == "relu":
            return np.maximum(0, x) if not derivative else (x > 0).astype(float)

    def forwardPropagation(self, inputs):
        layer_outputs = [inputs]
        for weight, bias in zip(self.weights, self.biases):
            inputs = self.activation_function(np.dot(inputs, weight) + bias)
            layer_outputs.append(inputs)
        return layer_outputs

    def predict(self, inputs):
        return self.forwardPropagation(inputs)[-1]

    def backwardPropagation(self, inputs, targets, layer_outputs, learning_rate):
        output = layer_outputs[-1]
        errors = targets - output
        delta = errors * self.activation_function(output, derivative=True)
        for i in range(len(self.weights) - 1, -1, -1):
            inputs_T = layer_outputs[i].T if i > 0 else inputs.T
            self.weights[i] += learning_rate * inputs_T.dot(delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.activation_function(layer_outputs[i], derivative=True)

    def compute_accuracy(self, predictions, targets):
        predicted_classes = (predictions > 0.5).astype(int)
        return np.mean(predicted_classes == targets)

    def train(self, inputs, targets, epochs, learning_rate):
        self.loss_history = []
        self.accuracy_history = []
        self.training_time = []

        for epoch in range(epochs + 1):
            t0 = time.time()
            layer_outputs = self.forwardPropagation(inputs)
            self.backwardPropagation(inputs, targets, layer_outputs, learning_rate)
            output = layer_outputs[-1]
            loss = np.mean((targets - output) ** 2)
            accuracy = self.compute_accuracy(output, targets)
            t1 = time.time()
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            self.training_time.append(t1 - t0)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

# === GUI Layer ===
class InteractiveNNGUI:
    def __init__(self):
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.Y = np.array([[0], [1], [1], [0]])
        self.input_size = 2
        self.hidden_sizes = [4]
        self.output_size = 1
        self.learning_rate = 0.1
        self.epochs = 1000
        self.nn = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        self.setup_gui()

    def setup_gui(self):
        plt.ion()
        self.fig = plt.figure(figsize=(11, 9))
        gs = gridspec.GridSpec(5, 2, height_ratios=[4, 0.5, 0.5, 0.5, 1])

        # Create plots
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_boundary = self.fig.add_subplot(gs[0, 1])
        self.ax_time = self.fig.add_subplot(gs[1, :])
        
        # Setup sliders
        self.slider_lr = Slider(self.fig.add_subplot(gs[2, :]), 'Learning Rate', 0.01, 1.0,
                                valinit=self.learning_rate, valstep=0.01)
        self.slider_epochs = Slider(self.fig.add_subplot(gs[3, :]), 'Epochs', 100, 10000,
                                    valinit=self.epochs, valstep=100)
        
        # Setup button and textbox
        self.button = Button(self.fig.add_axes([0.45, 0.01, 0.1, 0.05]), 'Train', 
                           color='#f5deb3', hovercolor='#c4a000')
        self.textbox = TextBox(self.fig.add_axes([0.82, 0.01, 0.15, 0.05]), 
                             'Hidden Layers', initial="4")

        # Connect events
        self.slider_lr.on_changed(self.update_learning_rate)
        self.slider_epochs.on_changed(self.update_epochs)
        self.button.on_clicked(self.start_training)
        self.textbox.on_submit(self.update_hidden_layers)

        self.initialize_plots()
        self.fig.tight_layout(pad=2.0)

    def initialize_plots(self):
        for ax in [self.ax_loss, self.ax_boundary, self.ax_time]:
            ax.clear()
            ax.grid(True)
        
        self.ax_loss.set_title("Loss & Accuracy")
        self.ax_boundary.set_title("Decision Boundary")
        self.ax_time.set_title("Training Time")

    def update_learning_rate(self, val):
        self.learning_rate = val

    def update_epochs(self, val):
        self.epochs = int(val)

    def update_hidden_layers(self, text):
        try:
            self.hidden_sizes = list(map(int, text.strip().split(",")))
            self.nn.set_hidden_layers(self.hidden_sizes)
        except ValueError:
            print("Please enter comma-separated integers (e.g. '4,3')")

    def start_training(self, event):
        self.button.disabled = True  # Disable during training
        self.button.label.set_text("Training...")
        plt.pause(0.1)  # Allow GUI to update
        
        try:
            self.nn.train(self.X, self.Y, self.epochs, self.learning_rate)
            self.update_plots()
        finally:
            self.button.disabled = False
            self.button.label.set_text("Train")
            plt.pause(0.1)  # Ensure button state updates

    def update_plots(self):
        # Loss and Accuracy
        self.ax_loss.clear()
        self.ax_loss.plot(self.nn.loss_history, label='Loss')
        self.ax_loss.plot(self.nn.accuracy_history, label='Accuracy')
        self.ax_loss.legend()
        
        # Decision Boundary
        self.plot_decision_boundary()
        
        # Time Complexity
        self.ax_time.clear()
        self.ax_time.plot(np.cumsum(self.nn.training_time))
        
        plt.draw()

    def plot_decision_boundary(self):
        self.ax_boundary.clear()
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), 
                           np.linspace(y_min, y_max, 100))
        Z = self.nn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        self.ax_boundary.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
        self.ax_boundary.scatter(self.X[:, 0], self.X[:, 1], c=self.Y.ravel(), 
                               cmap="RdBu", edgecolors='k', s=100)

# === Entry Point ===
def main():
    InteractiveNNGUI()
    plt.show(block=True)

if __name__ == "__main__":
    main()
