import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
import time

# Neural Network
class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.reset()

    def reset(self):
        sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        rng = np.random.default_rng()
        self.weights = [rng.standard_normal((sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.biases = [rng.standard_normal((1, size)) for size in sizes[1:]]
        self.loss_history = []
        self.accuracy_history = []
        self.training_time = []

    def sigmaMoment(self, x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def forwardPropagation(self, inputs):
        layer_outputs = [inputs]
        for weight, bias in zip(self.weights, self.biases):
            inputs = self.sigmaMoment(np.dot(inputs, weight) + bias)
            layer_outputs.append(inputs)
        return layer_outputs

    def predict(self, inputs):
        return self.forwardPropagation(inputs)[-1]

    def backwardPropagation(self, inputs, targets, layer_outputs, learning_rate):
        output = layer_outputs[-1]
        errors = targets - output
        delta = errors * self.sigmaMoment(output, derivative=True)
        for i in range(len(self.weights) - 1, -1, -1):
            inputs_T = layer_outputs[i].T if i > 0 else inputs.T
            self.weights[i] += learning_rate * inputs_T.dot(delta)
            self.biases[i] += learning_rate * np.sum(delta, axis=0, keepdims=True)
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.sigmaMoment(layer_outputs[i], derivative=True)

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

# GUI layer
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
        self.build_gui()

    def build_gui(self):
        plt.ion()
        self.fig = plt.figure(figsize=(11, 9))
        gs = gridspec.GridSpec(4, 2, height_ratios=[4, 0.5, 0.5, 1])

        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_boundary = self.fig.add_subplot(gs[0, 1])
        self.ax_lr_slider = self.fig.add_subplot(gs[1, :])
        self.ax_epoch_slider = self.fig.add_subplot(gs[2, :])
        self.ax_time_complexity = self.fig.add_subplot(gs[3, :])
        self.ax_button = plt.axes([0.45, 0.01, 0.1, 0.05], facecolor='#d4af37')

        self.button = Button(self.ax_button, 'Train', color='#f5deb3', hovercolor='#c4a000')
        self.button.label.set_fontsize(10)
        self.button.label.set_color('#2b2b2b')

        self.slider_lr = Slider(self.ax_lr_slider, 'Learning Rate', 0.01, 1.0, valinit=self.learning_rate, valstep=0.01)
        self.slider_epochs = Slider(self.ax_epoch_slider, 'Epochs', 100, 10000, valinit=self.epochs, valstep=100)

        self.slider_lr.on_changed(self.update_params)
        self.slider_epochs.on_changed(self.update_params)
        self.button.on_clicked(self.train_and_plot)

        # Set initial titles so they're not empty
        self.ax_loss.set_title("Loss & Accuracy")
        self.ax_boundary.set_title("Decision Boundary")
        self.ax_time_complexity.set_title("Training Time Complexity")

    def update_params(self, val):
        self.learning_rate = self.slider_lr.val
        self.epochs = int(self.slider_epochs.val)

    def train_and_plot(self, event):
        self.nn.reset()
        self.nn.train(self.X, self.Y, self.epochs, self.learning_rate)
        self.plot_loss_accuracy()
        self.plot_decision_boundary()
        self.plot_time_complexity()
        self.fig.canvas.draw_idle()
        time.sleep(0.05)  

    def plot_loss_accuracy(self):
        self.ax_loss.cla()
        self.ax_loss.plot(self.nn.loss_history, label='Loss')
        self.ax_loss.plot(self.nn.accuracy_history, label='Accuracy')
        self.ax_loss.set_title("Loss & Accuracy")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Value")
        self.ax_loss.grid(True)
        self.ax_loss.legend()

    def plot_decision_boundary(self):
        self.ax_boundary.cla()
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
        input_grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.nn.predict(input_grid)
        Z = predictions.reshape(xx.shape)

        self.ax_boundary.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.8)
        self.ax_boundary.scatter(self.X[:, 0], self.X[:, 1], c=self.Y.ravel(), edgecolors='k', cmap="RdBu", s=100)
        self.ax_boundary.set_title("Decision Boundary")
        self.ax_boundary.set_xlabel("Input 1")
        self.ax_boundary.set_ylabel("Input 2")
        self.ax_boundary.grid(True)

    def plot_time_complexity(self):
        self.ax_time_complexity.cla()
        total_time = np.cumsum(self.nn.training_time)
        self.ax_time_complexity.plot(total_time, label='Cumulative Time')
        self.ax_time_complexity.set_title("Training Time Complexity")
        self.ax_time_complexity.set_xlabel("Epoch")
        self.ax_time_complexity.set_ylabel("Time (s)")
        self.ax_time_complexity.grid(True)
        self.ax_time_complexity.legend()

# Entry Point
def main():
    InteractiveNNGUI()
    plt.show(block=True)

if __name__ == "__main__":
    main()
