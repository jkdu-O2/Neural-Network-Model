import numpy as np

class NeuralNetwork:
    def __init__(self, inputLayerSize, hiddenLayerSize, outputLayerSize):
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        #
        sizes = [inputLayerSize] + hiddenLayerSize + [outputLayerSize]
        rng = np.random.default_rng()
        # Improved Xavier Initialization
        self.weights = [rng.standard_normal((sizes[i], sizes[i+1])) / np.sqrt(sizes[i]) for i in range(len(sizes) - 1)]
        self.biases = [rng.standard_normal((1, size)) for size in sizes[1:]]
    #
    @staticmethod
    def preprocess(data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
    #
    def sigma(self, x, derivative=False):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig) if derivative else sig
    #
    def forwardPropagation(self, inputs):
        layerOutputs = [inputs]
        for weight, bias in zip(self.weights, self.biases):
            inputs = self.sigma(np.dot(inputs, weight) + bias)
            layerOutputs.append(inputs)
        return layerOutputs 
    #
    def backwardPropagation(self, inputs, targets, layerOutputs, learningRate):
        output = layerOutputs[-1]
        errors = targets - output
        delta = errors * self.sigma(output, derivative=True)
        #
        for i in range(len(self.weights) - 1, -1, -1):
            if i > 0:
                inputs_T = layerOutputs[i-1].T
                z = np.dot(layerOutputs[i-1], self.weights[i]) + self.biases[i]
            else:
                inputs_T = inputs.T
                z = np.dot(inputs, self.weights[i]) + self.biases[i]
            #
            self.weights[i] -= learningRate * inputs_T.dot(delta)
            self.biases[i] -= learningRate * np.sum(delta, axis=0, keepdims=True)
            #
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.sigma(z, derivative=True)
    #
    def train(self, inputs, targets, epochs, learningRate):
        for epoch in range(epochs): 
            layerOutputs = self.forwardPropagation(inputs)
            self.backwardPropagation(inputs, targets, layerOutputs, learningRate)
            if epoch % 1000 == 0:  # Reduced print frequency
                loss = np.mean((targets - layerOutputs[-1]) ** 2)
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
#
def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    #
    X_preprocess = NeuralNetwork.preprocess(X)
    #
    nn = NeuralNetwork(2, [4], 1)
    nn.train(X_preprocess, Y, epochs=10000, learningRate=0.1)

if __name__ == "__main__":
    main()
