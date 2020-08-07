import numpy as np
import PyNeuron


class NeuronLayer(object):
    def __init__(self, weights_arr, bias_vec):
        self.weights = weights_arr
        self.bias = bias_vec
        self.neurons = list()
        for i in range(len(self.weights)):
            self.neurons.append(PyNeuron.Neuron(self.weights[i], self.bias[i]))

    def inspect(self):
        print("\tNeurons:", len(self.weights))
        for i in range(len(self.weights)):
            print("\t\tNeuron", i)
            for j in range(len(self.weights[i])):
                print("\t\t\tWeight:", self.weights[i][j])
            print("\t\t\tbias:", self.bias[i])

    def feed_forward(self, inputs):
        outputs = list()
        for i in range(len(self.weights)):
            outputs.append(self.neurons[i].forward(inputs))
        return outputs

    def feed_backward(self, errors):
        layer_node_delta = list()
        for i in range(len(self.weights)):
            layer_node_delta.append(self.neurons[i].backward(errors[i]))
        return np.dot(layer_node_delta, self.weights)

    def update_weights(self, learning_rate=0.1):
        for i in range(len(self.weights)):
            self.weights[i], self.bias[i] = self.neurons[i].update_weights(learning_rate)
        return self.weights, self.bias


if __name__ == "__main__":
    pass
