import numpy as np
from keras.datasets import mnist
import tensorflow as tf

""""
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()
print("after downloading the data")
"""
# Load MNIST data

# 1.a - This function initialize the Weights matrices and bias vectors for each layer
def initialize_parameters(layer_dims):
    #output array
    init_dictionary = {}
    #number of layers in the network
    num_of_layers = len(layer_dims)

    #insert each parmeter to dictionary
    for i in range(1,num_of_layers):
        init_dictionary['W' + str(i)] = np.random.randn(layer_dims[i] , layer_dims[i-1]) * 0.01
        init_dictionary['b' + str(i)] = np.zeros((layer_dims[i],1))
    return init_dictionary


"""
# Input Arguments: A,W,b 
# Dimensions of input Arguemtns:
#   |A| = L_i x 1
#   |W| = L_i x L_i-1
#   |b| = L_i x 1
#    where i is the i-th layer in the neural network
# Output Argument: Z
# Dimension of output Argument:
# |Z| = L_i x 1
"""
# 1.b - This function performs the linear part of a layer's forward propagation
def linear_forward(A, W, b):
    #Z = np.expand_dims(np.dot(W, A), axis=1)+b
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)
    return Z,linear_cache
"""
#Testing linear_forward function
A = [0.2, 0.5, 0.8]             #|A| = 1x3
W = [[1, 5, 12],                #|W| = 4x3
    [-5, 9, 0],
    [-6, 11, 19],
    [2 , 4, 7]]
b = [0.5, 0.5 , 0.2 , 0.1]      #|b| = 4x1
print (linear_forward(A,W,b))
"""

#Softmax Activation function - sigmoid
"""
Input Argument: Z
Dimensions of input Arguemtns:
|Z| = L_i x 1
   where i is the i-th layer in the neural network
Output Argument: A
Dimension of output Argument:
|A| = L_i x 1
"""

# 1.c
def softmax(Z):
    sumExpZ = sum(np.exp(Z))
    A = np.exp(Z) / sumExpZ
    activation_cache = Z
    return A,activation_cache

# 1.d
def relu(Z):
    activation_cache = Z
    A = np.maximum(Z, 0)
    return A,activation_cache

# 1.e
def linear_activation_forward(A_prev, W, B, activation):
    Z,linear_cache = linear_forward(A_prev,W,B)
    if (activation == 'relu'):
        A,activation_cache = relu(Z)
    if (activation == 'softmax'):
        A,activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A,cache


def apply_batchnorm(A):
    eps = 1e-6
    meanA = np.mean(A)
    stdA = np.std(A)
    stdA_eps = np.sqrt(stdA^2 + eps)
    NA = A - meanA / stdA_eps
    return NA


# 1.f - this function opperates on all layers starting from Layer0 = X
def L_model_forward(X, parameters, use_batchnorm):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    #Layeres 1:L activate by Relu function
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W%s' % l], parameters['b%s' % l], 'relu')
        if use_batchnorm == True:
            A = apply_batchnorm(A)
        caches.append(cache)

    #Layer L+1 activate by Softmax function
    AL, cache = linear_activation_forward(A, parameters['W%s' % str(l + 1)], parameters['b%s' % str(l + 1)], 'softmax')
    caches.append(cache)
    return AL, caches


# 1.g
def compute_cost(AL,Y):
    """""
    calculate categorical cross-entropy loss using the formula
    Input:
    AL -- probability vector. shape:(1, #examples)
    Y -- correct lable vector (1 - true,0 - false). shape:(1, #exammples)
    Output:
    categorical cross-entropy loss
    """""
    m = Y.shape[1]
    logprobs = np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL), (1-Y))
    # logprobs = np.multiply(np.log(AL), Y) #Hodaya version, from the assignment,
    cost = (-1/m) * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


# 2.a
def Linear_backward(dZ,cache):
    """""
    backward propagation process for a single layer
    Input:
    dZ - the gradient of the cost with respect to the linear output of the current layer (layer l)
    cache - tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    Output:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """""
    A_prev, W, b = cache
    m = len(A_prev)
    dW = np.dot(dZ,A_prev.T) / m
    db = np.sum(dZ, axis=1,keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev,dW,db

# 2.b
def linear_activation_backward(dA, cache, activation):
#TODO: add doc
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)

    dA_prev, dW, db = Linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

# 2.c
def relu_backward(dA, activation_cache):
    """
    The backward propagation for a single RELU unit.
    Arguments:
    dA - post-activation gradient, of any shape
    activation_cache - 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ - Gradient of the cost with respect to Z
    """
    Z = activation_cache
    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    # When z < 0, we should set dz to 0 as well.
    dZ[Z < 0] = 0
    return dZ

# 2.d
def softmax_backward(dA, activation_cache):
    Z = activation_cache
    lenZ = len(Z)
    expZ = np.exp(Z)
    sumExpZ = sum(expZ)
    Softmax = expZ / sumExpZ
    dSoftmax = np.zeros((lenZ,lenZ))

    dZ = np.zeros(dA.shape)

    for iExample in range(Softmax.shape[1]):
        iExampleSoftmax = np.expand_dims(Softmax[:, iExample], axis=-1)
        dSoftmaxTemp = -1 * iExampleSoftmax * np.transpose(iExampleSoftmax)
        for i in range(lenZ):
            dSoftmaxTemp[i,i] -= iExampleSoftmax[i]
        dZ[:,iExample] = np.matmul(dSoftmaxTemp, dA[:,iExample])

    return dZ


# Hodaya's reference - unclear from where this pars is true: "np.multiply(dA, s, (1 - s))"
def softmax_backward2(dA, activation_cache):
    z = activation_cache
    s = (np.exp(z).T / np.array((np.sum(np.exp(z), axis=1).T))).T
    dZ = np.multiply(dA, s, (1 - s))    # wtf
    return dZ

# 2.e
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))         #not sure this is correct
    dAL = AL - Y                                                           #trying from group

    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  'softmax')

    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 2)], caches".
        # Outputs: "grads["dA" + str(l + 1)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, 'relu')
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads
# 2.f
def update_parameters(parameters, grads, learning_rate):
#TODO: add doc
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(1, L + 1, 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters

# 3.a
def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    parameters = initialize_parameters(layers_dims)
    use_batchnorm = False
    cost = []
    numEpochs = 1
    epsilon = 0.00001

    for n in range(numEpochs):
        # Training phase:
        for i in range(0,num_iterations):
            randIndices = np.random.choice(X.shape[0], batch_size)
            # orderedIndices = range(i * batch_size, (i+1) * batch_size)
            X_batch_train = X[randIndices].transpose()
            Y_batch_train = Y[randIndices].transpose()
            #foward propagation
            AL, cache = L_model_forward(X_batch_train, parameters, use_batchnorm)
            cost = compute_cost(AL, Y_batch_train)
            grads = L_model_backward(AL, Y_batch_train, cache)
            parameters = update_parameters(parameters, grads, learning_rate)

        # Validation phase:
        randIndices = np.random.choice(X.shape[0], X.shape[0] // 5) #Validation on 20% from the training set
        X_batch_valid = X[randIndices].transpose()
        Y_batch_valid = Y[randIndices].transpose()
        acc = Predict(X_batch_valid,Y_batch_valid,parameters)


    return parameters, cost

# 3.b
def Predict(X, Y, parameters):
    use_batchnorm = False
    AL, cache = L_model_forward(X,parameters,use_batchnorm)
    countHits = 0
    for i in range(Y.shape[1]):
        labelID = np.argmax(AL[:,i])
        if(labelID == argmax(Y[:,i])):
            countHits += 1
    accuracy = countHits * 100 / Y.shape[1]
    return accuracy
#from sklearn.preprocessing import label_binarize

def getMnistFlatData():
    # Load MNIST data
    (x_train, y_train), (x_test,y_test) = mnist.load_data()
    max_x_train = max(x_train.reshape((x_train.shape[0] * x_train.shape[1] * x_train.shape[2])))
    x_train = (x_train - max_x_train / 2) / max_x_train
    # x_train = x_train / max_x_train
    numOfClasses = len(np.unique(y_train))
    lenImageFlattened = x_train.shape[1] *  x_train.shape[2]
    x_train_reshape = x_train.reshape(x_train.shape[0],lenImageFlattened)
    y_train_reshape = np.zeros((len(y_train),numOfClasses))
    for i in range(len(y_train)):
        y_train_reshape[i][y_train[i]] = 1

    return x_train_reshape,y_train_reshape, numOfClasses,lenImageFlattened


#CACHE:
# linear_cache = A - input of the layer(in size of the current layer), W - Weights matrix, b - biases
# activation_cache  = Z values = A*W + b


x_train_reshape, y_train_reshape, numOfClasses, lenImageFlattened = getMnistFlatData()
use_batchnorm = False
learning_rate = 0.009      # Hard Coded Value is: 0.009
batch_size = 32
num_of_iterations = 100      #y_train_reshape.shape[0] // batch_size

dimArray = [20,7,5,10]
dimArray.append(numOfClasses)           #first - input layer
dimArray.insert(0,lenImageFlattened)    #last  - output layer

#L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size)

(parameters, cost) = L_layer_model(x_train_reshape,y_train_reshape,dimArray,learning_rate,num_of_iterations,batch_size)
p = Predict(np.expand_dims(x_train_reshape[0, :], axis=1), y_train_reshape[0, :], parameters)

#x_test_reshape = x_test.reshape(x_test.shape[0],784)
#AL, cache = L_model_forward(x_test_reshape[0], parameters,use_batchnorm )
print('Finished')