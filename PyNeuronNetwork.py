import numpy as np
import PyNeuronLayer
import PyOptimizer
import time
import random


class NeuronNetwork(object):
    def __init__(self, weights_arr, bias_vec):
        self.weights = weights_arr
        self.bias = bias_vec
        self.neuron_layers = list()
        for i in range(len(self.weights)):
            self.neuron_layers.append(PyNeuronLayer.NeuronLayer(self.weights[i], self.bias[i]))

    def inspect(self):
        print("Layers:", len(self.weights))
        for i in range(len(self.weights)):
            print("\tNeuronLayer", i)
            self.neuron_layers[i].inspect()

    def feed_forward(self, inputs):
        outputs = inputs
        for i in range(len(self.weights)):
            outputs = self.neuron_layers[i].feed_forward(outputs)
        return outputs

    def compute_loss(self, training_inputs, training_outputs):
        predicted_outputs = self.feed_forward(training_inputs)
        return np.linalg.norm(np.subtract(predicted_outputs, training_outputs))

    def train(self, training_inputs, training_outputs, learning_rate=0.1):
        predicted_outputs = self.feed_forward(training_inputs)
        errors = np.subtract(training_outputs, predicted_outputs)
        for i in reversed(range(len(self.weights))):
            errors = self.neuron_layers[i].feed_backward(errors)
        self.__update_weights(learning_rate)

    def __update_weights(self, learning_rate=0.1):
        for i in range(len(self.weights)):
            self.weights[i], self.bias[i] = self.neuron_layers[i].update_weights(learning_rate)


if __name__ == "__main__":
    # These parameters are used for testing.

    # weights = [[[0.15, 0.20], [0.25, 0.30]], [[0.40, 0.45], [0.50, 0.55]]]
    # bias = [[0.35, 0.35], [0.60, 0.60]]
    # inputs = [0.05, 0.10]
    # outputs = [0.01, 0.99]
    # training_sets = [[inputs, outputs]]
    # epoch = 10000
    # learning_rate = 0.5
    # testClass = NeuronNetwork(weights, bias)
    # testClass.inspect()
    # initial_output = list()
    # for i in range(len(training_sets)):
    #     initial_output.append(testClass.feed_forward(training_sets[i][0]))
    # print("Initial hidden weights:", *weights[0])
    # print("Initial hidden biases: ", bias[0])
    # print("Initial output weights:", *weights[1])
    # print("Initial output biases: ", bias[1])
    # print("Initial error:         ", np.linalg.norm(np.subtract(outputs, initial_output)))
    # print("Initial output:        ", *initial_output)
    # print("Target output:         ", *outputs, "\n")

    # start_ = time.time()
    # for i in range(epoch):
    #     training_inputs, training_outputs = random.choice(training_sets)
    #     testClass.train(training_inputs, training_outputs, learning_rate)
    # end_ = time.time()

    # final_output = list()
    # for i in range(len(training_sets)):
    #     final_output.append(testClass.feed_forward(training_sets[i][0]))
    # print("Final hidden weights:  ", *testClass.weights[0])
    # print("Final hidden bias:     ", testClass.bias[0])
    # print("Final output weights:  ", *testClass.weights[1])
    # print("Final output bias:     ", testClass.bias[1])
    # print("Final error:           ", np.linalg.norm(np.subtract(outputs, final_output)))
    # print("Final output:          ", *final_output)
    # print("Target output:         ", *outputs, "\n")

    # print("time: ", end_ - start_)
    # print()

    # Training a 2-input XOR 2-layer neural network

    # WEIGHTS = list()
    # for i in range(9):
    #     # WEIGHTS.append(i / 10 + 0.1)
    #     WEIGHTS.append(random.random())
    # weights = [[[WEIGHTS[0], WEIGHTS[1]], [WEIGHTS[3], WEIGHTS[4]]],
    #            [[WEIGHTS[6], WEIGHTS[7]]]]
    # bias = [[WEIGHTS[2], WEIGHTS[5]],
    #         [WEIGHTS[8]]]
    # inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # outputs = [[0], [1], [1], [0]]
    # training_sets = list()
    # for i in range(len(outputs)):
    #     training_sets.append([inputs[i], outputs[i]])

    # epoch = 10000
    # learning_rate = 0.5
    # testClass = NeuronNetwork(weights, bias)
    # # testClass.inspect()
    # initial_output = list()
    # for i in range(len(training_sets)):
    #     initial_output.append(testClass.feed_forward(training_sets[i][0]))
    # print("Initial hidden weights:", *weights[0])
    # print("Initial hidden biases: ", bias[0])
    # print("Initial output weights:", *weights[1])
    # print("Initial output biases: ", bias[1])
    # print("Initial error:         ", np.linalg.norm(np.subtract(outputs, initial_output)))
    # print("Initial output:        ", *initial_output)
    # print("Target output:         ", *outputs, "\n")

    # start_ = time.time()
    # for i in range(epoch):
    #     training_inputs, training_outputs = random.choice(training_sets)
    #     testClass.train(training_inputs, training_outputs, learning_rate)
    # end_ = time.time()

    # final_output = list()
    # for i in range(len(training_sets)):
    #     final_output.append(testClass.feed_forward(training_sets[i][0]))
    # print("Final hidden weights:  ", *testClass.weights[0])
    # print("Final hidden bias:     ", testClass.bias[0])
    # print("Final output weights:  ", *testClass.weights[1])
    # print("Final output bias:     ", testClass.bias[1])
    # print("Final error:           ", np.linalg.norm(np.subtract(outputs, final_output)))
    # print("Final output:          ", *final_output)
    # print("Target output:         ", *outputs, "\n")

    # print("time: ", end_ - start_)
    # print()

    # Training a 3-input XOR 2-layer neural network

    WEIGHTS = list()
    for i in range(16):
        # WEIGHTS.append(i / 10 + 0.1)
        WEIGHTS.append(random.random())
    weights = [[[WEIGHTS[0], WEIGHTS[1], WEIGHTS[2]], [WEIGHTS[4], WEIGHTS[5], WEIGHTS[6]], [WEIGHTS[8], WEIGHTS[9], WEIGHTS[10]]],
               [[WEIGHTS[12], WEIGHTS[13], WEIGHTS[14]]]]
    bias = [[WEIGHTS[3], WEIGHTS[7], WEIGHTS[11]],
            [WEIGHTS[15]]]
    inputs = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    outputs = [[0], [1], [1], [0], [1], [0], [0], [1]]
    training_sets = list()
    for i in range(len(outputs)):
        training_sets.append([inputs[i], outputs[i]])

    epoch = 10000
    learning_rate = 5
    testClass = NeuronNetwork(weights, bias)
    # testClass.inspect()
    initial_output = list()
    for i in range(len(training_sets)):
        initial_output.append(testClass.feed_forward(training_sets[i][0]))
    print("Initial hidden weights:", *testClass.weights[0])
    print("Initial hidden biases: ", testClass.bias[0])
    print("Initial output weights:", *testClass.weights[1])
    print("Initial output biases: ", testClass.bias[1])
    print("Initial error:         ", np.linalg.norm(np.subtract(outputs, initial_output)))
    print("Initial output:        ", *initial_output[0:4])
    print("                       ", *initial_output[4:8])
    print("Target output:         ", *outputs, "\n")

    start_ = time.time()
    for i in range(epoch):
        training_inputs, training_outputs = random.choice(training_sets)
        testClass.train(training_inputs, training_outputs, learning_rate)
    end_ = time.time()

    final_output = list()
    for i in range(len(training_sets)):
        final_output.append(testClass.feed_forward(training_sets[i][0]))
    print("Final hidden weights:  ", *testClass.weights[0])
    print("Final hidden bias:     ", testClass.bias[0])
    print("Final output weights:  ", *testClass.weights[1])
    print("Final output bias:     ", testClass.bias[1])
    print("Final error:           ", np.linalg.norm(np.subtract(outputs, final_output)))
    print("Final output:          ", *final_output[0:4])
    print("                       ", *final_output[4:8])
    print("Target output:         ", *outputs, "\n")

    print("time: ", end_ - start_)
    print()
