import numpy as np


class Neuron(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        self.input = inputs
        total_net_input = self.__calculate_total_net_input()
        self.neuron_output = self.__sigmoid(total_net_input)
        return self.neuron_output

    def backward(self, error):
        self.node_delta = self.__node_delta(error)
        return self.node_delta

    def update_weights(self, learning_rate=0.1):
        self.weights += np.dot(self.node_delta, self.input) * learning_rate
        self.bias += self.node_delta * learning_rate
        return self.weights, self.bias

    def __calculate_total_net_input(self):
        return np.dot(self.input, self.weights) + self.bias

    def __sigmoid(self, total_net_input):
        return 1 / (1 + np.exp(-total_net_input))

    def __node_delta(self, error):
        return error * self.neuron_output * (1 - self.neuron_output)


if __name__ == "__main__":
    pass
