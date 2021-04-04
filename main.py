import numpy as np
from keras.utils.np_utils import to_categorical
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime


# 1 - Propagation Forward
# 1.a
def initialize_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layer_dims)):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2 / layer_dims[i])
        parameters['b' + str(i)] = np.zeros(shape=(layer_dims[i], 1))
    return parameters

# 1.b
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = {'A': A, 'W': W, 'b': b}
    return Z, cache

# 1.c
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    activation_cache = {'Z': Z}
    return A, activation_cache

# 1.d
def relu(Z):
    """
    Input:
    Z – the linear component of the activation function

    Output:
    A – the activations of the layer
    activation_cache – returns Z, which will be useful for the backpropagation
    """
    activation_cache = {'Z': Z}
    A = np.maximum(0, Z)
    return A, activation_cache

# 1.e
def linear_activation_forward(A_prev, W, B, activation, use_batchnorm=False):
    """
    Description:
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Input:
    A_prev – activations of the previous layer
    W – the weights matrix of the current layer
    B – the bias vector of the current layer
    Activation – the activation function to be used (a string, either “softmax” or “relu”)

    Output:
    A – the activations of the current layer
    cache – a joint dictionary containing both linear_cache and activation_cache
    """
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)
    cache = {**linear_cache, **activation_cache}
    return A, cache

# 1.f
def L_model_forward(X, parameters, use_batchnorm=False):
    """
    Description:
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation

    Input:
    X – the data, numpy array of shape (input size, number of examples)
    parameters – the initialized W and b parameters of each layer
    use_batchnorm - a boolean flag used to determine whether to apply batchnorm after the activation (note that this option needs to be set to “false” in Section 3 and “true” in Section 4).

    Output:
    AL – the last post-activation value
    caches – a list of all the cache objects generated by the linear_forward function
    """
    caches = list()
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Layeres 1:L activate by Relu function
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W%s' % l], parameters['b%s' % l], 'relu')
        if use_batchnorm == True:
            A = apply_batchnorm(A)
        caches.append(cache)

    # Layer L+1 activate by Softmax function
    AL, cache = linear_activation_forward(A, parameters['W%s' % str(l + 1)], parameters['b%s' % str(l + 1)], 'softmax')
    caches.append(cache)
    return AL, caches

# 1.g
def compute_cost(AL, Y):
    """
    Description:
    Implement the cost function defined by equation. The requested cost function is categorical cross-entropy loss.

    Input:
    AL – probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    Y – the labels vector (i.e. the ground truth)

    Output:
    cost – the cross-entropy cost
    """
    m = len(Y)
    ans = -(1 / m) * np.sum(np.log(AL[Y, np.arange(m)]))
    return ans

# 1.h
def apply_batchnorm(A):
    """
    Description:
    performs batchnorm on the received activation values of a given layer.

    Input:
    A - the activation values of a given layer

    output:
    NA - the normalized activation values, based on the formula learned in class
    """
    var = np.var(A)
    mu = np.mean(A)
    epsilon = 0.0000001
    A_tmp = (A - mu) / ((var + epsilon) ** 0.5)
    gamma = 1       # changeable
    beta = 0        # changeable
    normal_a = beta + gamma * A_tmp
    return normal_a

# 2.a
def Linear_backward(dZ, cache):
    """
    description:
    Implements the linear part of the backward propagation process for a single layer

    Input:
    dZ – the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache – tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Output:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    np.asarray(dZ)
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.transpose(), dZ)
    return dA_prev, dW, db

# 2.b
def linear_activation_backward(dA, cache, activation):
    """
    Description:
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies the linear_backward function.

    Input:
    dA – post activation gradient of the current layer
    cache – contains both the linear cache and the activations cache

    Output:
    dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW – Gradient of the cost with respect to W (current layer l), same shape as W
    db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache = {'A': cache['A'], 'W': cache['W'], 'b': cache['b']}
    activation_cache = {'Z': cache['Z']}
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = softmax_backward(dA, activation_cache)

    return Linear_backward(dZ, linear_cache)

# 2.c
def relu_backward(dA, activation_cache):
    """
    Description:
    Implements backward propagation for a ReLU unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache['Z']
    dZ = dA
    dZ[Z <= 0] = 0
    #dZ[Z > 0] = 1
    return dZ

# 2.d
def softmax_backward(dA, activation_cache):
    """
    Description:
    Implements backward propagation for a softmax unit

    Input:
    dA – the post-activation gradient
    activation_cache – contains Z (stored during the forward propagation)

    Output:
    dZ – gradient of the cost with respect to Z
    """
    dZ = dA
    return dZ

# 2.e
def L_model_backward(AL, Y, caches):
    """
    Description:
    Implement the backward propagation process for the entire network.

    Some comments:
    the backpropagation for the softmax function should be done only once as only the output layers uses it and the RELU should be done iteratively over all the remaining layers of the network.

    Input:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache

    Output:
    Grads - a dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers

    # derivitve of softmax as showen in lecture
    dA = AL - Y

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA, current_cache,'softmax')
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = dA_prev_temp, dW_temp, db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'relu')
        grads["dA" + str(l + 1)],grads["dW" + str(l + 1)],grads["db" + str(l + 1)]  = dA_prev_temp,dW_temp,db_temp

    return grads

# 2.f
def Update_parameters(parameters, grads, learning_rate):
    """
    Description:
    Updates parameters using gradient descent

    Input:
    parameters – a python dictionary containing the DNN architecture’s parameters
    grads – a python dictionary containing the gradients (generated by L_model_backward)
    learning_rate – the learning rate used to update the parameters (the “alpha”)

    Output:
    parameters – the updated values of the parameters object provided as input
    """
    layers = int(len(parameters) / 2)
    for l in range(layers):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

# 3.b
def predict(X, Y, paramters, use_batchnorm=False):
    """
    Description:
    The function receives an input data and the true labels and calculates the accuracy of the trained neural network on the data.

    Input:
    X – the input data, a numpy array of shape (height*width, number_of_examples)
    Y – the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    Parameters – a python dictionary containing the DNN architecture’s parameters

    Output:
    accuracy – the accuracy measure of the neural net on the provided data (i.e. the percentage of the samples for which the correct label receives the hughest confidence score). Use the softmax function to normalize the output values.

    """
    A, _ = L_model_forward(X, paramters, use_batchnorm=use_batchnorm)
    accuracy = sum(np.argmax(A, axis=0) == Y) / len(Y)
    return accuracy


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val = X_train.T, X_val.T

    Y_train_categorial = to_categorical(Y_train, num_classes=layers_dims[-1]).T
    Y_val_categorial = to_categorical(Y_val, num_classes=layers_dims[-1])

    parameters = initialize_parameters(layers_dims)
    costs = [[], []]

    m = X_train.shape[1]  # number of training examples

    iteration_number = 0
    epoch_number = 0
    no_improvement_count = 0
    min_val_cost = float('inf')
    batch_costs = []

    while True:
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X_train[:, permutation]
        shuffled_Y = Y_train[permutation]
        shuffled_Y_categorial = Y_train_categorial[:, permutation]

        k = 0

        while k < (m / batch_size):

            XBatches = shuffled_X[:, k * batch_size: (k + 1) * batch_size]
            # print(XBatches.shape)
            YBatches = shuffled_Y[k * batch_size: (k + 1) * batch_size]
            Y_cat_Batches = shuffled_Y_categorial[:, k * batch_size: (k + 1) * batch_size]

            AL, caches = L_model_forward(XBatches, parameters, use_batchnorm)
            batch_costs.append(compute_cost(AL, YBatches))
            grads = L_model_backward(AL, Y_cat_Batches, caches)
            parameters = Update_parameters(parameters, grads, learning_rate)

            AL, _ = L_model_forward(X_val, parameters, use_batchnorm=use_batchnorm)
            new_val_cost = compute_cost(AL, Y_val)

            if iteration_number % 100 == 0:
                print('Iteration No.', iteration_number, ' - ', "Train Cost: ", np.mean(batch_costs),
                      ", Validation Cost: ", new_val_cost)
                costs[0].append(np.mean(batch_costs))
                costs[1].append(new_val_cost)
                batch_costs = []

            if min_val_cost - new_val_cost >= IMPROVEMENT_THRESHOLD:
                min_val_cost = new_val_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= NO_IMPROVEMENT_STOP_TRESHOLD or iteration_number == num_iterations:

                train_acc = predict(X_train, Y_train, parameters, use_batchnorm=use_batchnorm)
                val_acc = predict(X_val, Y_val, parameters, use_batchnorm=use_batchnorm)

                print('Number of Epochs: ', epoch_number, ' Mumber of Iterations: ', iteration_number)
                print('Train Accuracy: ', train_acc)
                print('Validation Accuracy: ', val_acc)
                Train_cost = costs[0]
                Val_cost = costs[1]
                epochs = range(0, iteration_number // 100 + 1)
                plt.plot(epochs, Train_cost, 'g', label='Training cost')
                plt.plot(epochs, Val_cost, 'b', label='validation cost')
                if use_batchnorm == True:
                    plt.title('Training and Validation cost - with Batch Normalization')
                elif use_batchnorm == False:
                    plt.title('Training and Validation cost - without Batch Normalization')


                plt.xlabel('Iterations (per hundreds)')
                plt.ylabel('Cost')
                plt.legend()
                plt.show()

                return parameters, costs

            k = k + 1
            iteration_number += 1

        epoch_number += 1


# General parameters for running:
IMPROVEMENT_THRESHOLD = 10 ** -13
NO_IMPROVEMENT_STOP_TRESHOLD = 100
LearningRate = 0.009
LayerDims = [20, 7, 5, 10]
BatchSize = 32

# Get the data and normalize it for the NN
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
maxX = np.max(x_train)
lenImageFlat = x_train.shape[1] * x_train.shape[2]
LayerDims.insert(0, lenImageFlat)
NumberOfIterations = 50000
x_train = x_train.reshape(x_train.shape[0],lenImageFlat) / 255
x_test = x_test.reshape(x_test.shape[0], lenImageFlat).T / 255


# 1'st Run - with no Batch Normalization
batchnormFlag = False
start = datetime.now()
parameters, costs = L_layer_model(x_train, y_train, LayerDims, LearningRate, NumberOfIterations, BatchSize, batchnormFlag)
Test_acc = predict(x_test, y_test, parameters, batchnormFlag)
end = datetime.now()
# Printing the results:
print('Test Accuracy:   ', Test_acc)
print('Total Runing Time: ', end - start)



# 2'nd Run - with no Batch Normalization
use_batch_norm = True
start = datetime.now()
parameters, costs = L_layer_model(x_train, y_train, LayerDims, LearningRate, NumberOfIterations, BatchSize, batchnormFlag)
Test_acc = predict(x_test, y_test, parameters, batchnormFlag)
end = datetime.now()
# Printing the results:
print('Test Accuracy:   ', Test_acc)
print('Total Runing Time: ', end - start)
