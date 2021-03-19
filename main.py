import numpy as np

# a - This function initialize the Weights matrices and bias vectors for each layer
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

#Lavi is my partner
#Testing 2
#this is new info lavi
#kishkos

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
# b - This function performs the linear part of a layer's forward propagation
def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
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
def softmax(Z):
    sumExpZ = sum(np.exp(Z))
    A = np.exp(Z) / sumExpZ
    activation_cache = Z
    return A,activation_cache

def relu(z):
    activation_cache = Z
    A = np.maximum(z, 0)
    return A,activation_cache

def linear_activation_forward(A_prev, W, B, activation):
    Z,linear_cache = linear_forward(A_prev,W,b)
    if (activation == 'relu'):
        A,activation_cache = relu(Z)
    if (activation == 'softmax'):
        A,activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A,cache


#This function is not clear to me yet
def L_model_forward(X, parameters, use_batchnorm):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        ### START CODE HERE ### (≈ 2 lines of code)
        A, cache = linear_activation_forward(A_prev, parameters['W%s' % l], parameters['b%s' % l], 'relu')
        caches.append(cache)
        ### END CODE HERE ###

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    AL, cache = linear_activation_forward(A, parameters['W%s' % str(l + 1)], parameters['b%s' % str(l + 1)], 'sigmoid')
    caches.append(caches)
    ### END CODE HERE ###
    assert (AL.shape == (1, X.shape[1]))
    return AL, cache
