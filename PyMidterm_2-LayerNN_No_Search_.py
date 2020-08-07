import numpy as np
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets

# inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
#                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# expected_output = np.array([[0], [1], [1], [0], [1], [0], [0], [1]])
# inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 3, 2, 1

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

X0 = list()
for i in range(1, 10):
    X0.append(i / 10)

hidden_weights = np.array([[X0[0], X0[1]], [X0[3], X0[4]]])
hidden_bias = np.array([[X0[2], X0[5]]])
output_weights = np.array([[X0[6]], [X0[7]]])
output_bias = np.array([[X0[8]]])

# # Random weights and bias initialization
# hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
# hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
# output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
# output_bias = np.random.uniform(size=(1, outputLayerNeurons))


# inputs = np.array([[0.05, 0.10]])
# expected_output = np.array([[0.01, 0.99]])
# hidden_weights = np.array([[0.15, 0.25], [0.20, 0.30]])
# hidden_bias = np.array([[0.35, 0.35]])
# output_weights = np.array([[0.40, 0.50], [0.45, 0.55]])
# output_bias = np.array([[0.60, 0.60]])

print("Initial hidden weights:", *hidden_weights)
print("Initial hidden biases: ", *hidden_bias)
print("Initial output weights:", *output_weights)
print("Initial output biases: ", *output_bias)

epochs = 10000
lr = 0.1
errors = list()
outputs = list()

start_ = time.time()
# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    print(hidden_layer_activation)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    # print(inputs)
    # print(hidden_weights)
    print(hidden_layer_activation)
    print(hidden_layer_output)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

    outputs.append(predicted_output)
    errors.append(error)

end_ = time.time()

print("Initial error:         ", np.linalg.norm(errors[0]))
print("Initial output:        ", *outputs[0])
print("Target output:         ", *expected_output, "\n")
print("Final hidden weights:  ", *hidden_weights)
print("Final hidden bias:     ", *hidden_bias)
print("Final output weights:  ", *output_weights)
print("Final output bias:     ", *output_bias)
print("Final error:           ", np.linalg.norm(errors[-1]))
print("Final output:          ", *predicted_output)
print("Target output:         ", *expected_output, "\n")

print("time: ", end_ - start_)
print()
