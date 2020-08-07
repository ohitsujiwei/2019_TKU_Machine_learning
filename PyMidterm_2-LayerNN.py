import numpy as np
import matplotlib.pyplot as plt
import time
import random
import PyOptimizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prediction_2_input(x, in1, in2):
    x1 = in1
    x2 = in2
    f1 = sigmoid(np.dot(x[0], x1) + np.dot(x[1], x2) + x[2])
    f2 = sigmoid(np.dot(x[3], x1) + np.dot(x[4], x2) + x[5])
    output = sigmoid(np.dot(x[6], f1) + np.dot(x[7], f2) + x[8])
    return output


def learning_2_input(x):

    target = [[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]]  # XOR

    # target = [[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]]  # XNOR

    f1 = sigmoid(np.dot(x[0], target[0]) + np.dot(x[1], target[1]) + x[2])
    f2 = sigmoid(np.dot(x[3], target[0]) + np.dot(x[4], target[1]) + x[5])
    er = target[2] - sigmoid(np.dot(x[6], f1) + np.dot(x[7], f2) + x[8])

    return np.linalg.norm(er)


def prediction_3_input(x, in1, in2, in3):
    x1 = in1
    x2 = in2
    x3 = in3
    f1 = sigmoid(np.dot(x[0], x1) + np.dot(x[1], x2) + np.dot(x[2], x3) + x[3])
    f2 = sigmoid(np.dot(x[4], x1) + np.dot(x[5], x2) + np.dot(x[6], x3) + x[7])
    f3 = sigmoid(np.dot(x[8], x1) + np.dot(x[9], x2) + np.dot(x[10], x3) + x[11])
    output = sigmoid(np.dot(x[12], f1) + np.dot(x[13], f2) + np.dot(x[14], f3) + x[15])
    return output


def learning_3_input(x):

    target = [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 0, 1]]  # XOR

    # target = [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]]  # XNOR

    f1 = sigmoid(np.dot(x[0], target[0]) + np.dot(x[1], target[1]) + np.dot(x[2], target[2]) + x[3])
    f2 = sigmoid(np.dot(x[4], target[0]) + np.dot(x[5], target[1]) + np.dot(x[6], target[2]) + x[7])
    f3 = sigmoid(np.dot(x[8], target[0]) + np.dot(x[9], target[1]) + np.dot(x[10], target[2]) + x[11])
    er = target[3] - sigmoid(np.dot(x[12], f1) + np.dot(x[13], f2) + np.dot(x[14], f3) + x[15])

    return np.linalg.norm(er)


if __name__ == "__main__":
    input_x, number_x = list(), 9
    for i in range(0, number_x):
        input_x.append(0.1 + i / 10)
        # input_x.append(random.uniform(1, 2))
    input_func = learning_2_input
    input_Diff = "Backward"
    input_LS = "FiS"
    input_MinNorm = 1e-3
    input_MaxIter = 1e+3
    test_CGradDecent = PyOptimizer.CGradDecent(input_func, input_x, number_x, input_Diff, input_LS, input_MinNorm, input_MaxIter)
    test_CGradDecent.LineSearch.set_delta(1e-6)
    test_CGradDecent.LineSearch.set_eps(1e-2)

    start_ = time.time()
    ouput_x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
    end_ = time.time()

    result_init = prediction_2_input(input_x, [0, 0, 1, 1], [0, 1, 0, 1])
    result_final = prediction_2_input(ouput_x, [0, 0, 1, 1], [0, 1, 0, 1])
    error_init = learning_2_input(input_x)
    error_final = learning_2_input(ouput_x)

    print("Initial weights = [", ', '.join('{:7.4f}'.format(f) for f in input_x), "]")
    print("Initial error   = ", error_init)
    print("Initial outputs = [", ', '.join('{:7.4f}'.format(f) for f in result_init), "]")
    print("Final weights   = [", ', '.join('{:7.4f}'.format(f) for f in ouput_x), "]")
    print("Final error     = ", error_final)
    print("Final outputs   = [", ', '.join('{:7.4f}'.format(f) for f in result_final), "]")
    print("Total time      = ", end_ - start_)
    print()

    input_x, number_x = list(), 16
    for i in range(0, number_x):
        input_x.append(0.1 + i / 10)
        # input_x.append(random.uniform(1, 2))

    test_CGradDecent.set_x0(input_x)
    test_CGradDecent.set_dim(number_x)
    test_CGradDecent.set_costfun(learning_3_input)

    start_ = time.time()
    ouput_x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
    end_ = time.time()

    result_init = prediction_3_input(input_x, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1])
    result_final = prediction_3_input(ouput_x, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1])
    error_init = learning_3_input(input_x)
    error_final = learning_3_input(ouput_x)

    print("Initial weights = [", ', '.join('{:7.4f}'.format(f) for f in input_x[0:8]))
    print("                   ", ', '.join('{:7.4f}'.format(f) for f in input_x[8:16]), "]")
    print("Initial error   = ", error_init)
    print("Initial outputs = [", ', '.join('{:7.4f}'.format(f) for f in result_init), "]")
    print("Final weights   = [", ', '.join('{:7.4f}'.format(f) for f in ouput_x[0:8]))
    print("                   ", ', '.join('{:7.4f}'.format(f) for f in ouput_x[8:16]), "]")
    print("Final error     = ", error_final)
    print("Final outputs   = [", ', '.join('{:7.4f}'.format(f) for f in result_final), "]")
    print("Total time      = ", end_ - start_, "\n")
    print()

    plt.figure(figsize=(8, 4))
    plt.plot(np.linspace(0, iter_K, len(errors)), errors)
    plt.xticks(np.linspace(0, iter_K, 11))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
