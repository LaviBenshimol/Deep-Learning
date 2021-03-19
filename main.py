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

# c
def softmax(Z):
    sumExpZ = sum(np.exp(Z))
    A = np.exp(Z) / sumExpZ
    activation_cache = Z
    return A,activation_cache

# d
def relu(Z):
    activation_cache = Z
    A = np.maximum(Z, 0)
    return A,activation_cache

# e
def linear_activation_forward(A_prev, W, B, activation):
    Z,linear_cache = linear_forward(A_prev,W,B)
    if (activation == 'relu'):
        A,activation_cache = relu(Z)
    if (activation == 'softmax'):
        A,activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A,cache




# f - this function opperates on all layers starting from Layer0 = X
def L_model_forward(X, parameters, use_batchnorm):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    #Layeres 1:L activate by Relu function
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W%s' % l], parameters['b%s' % l], 'relu')
        caches.append(cache)

    #Layter L+1 activate by Softmax function
    AL, cache = linear_activation_forward(A, parameters['W%s' % str(l + 1)], parameters['b%s' % str(l + 1)], 'softmax')
    caches.append(caches)
    return AL, cache



X = [0.5 , 0.6 , 0.7]
dimArray = [3,4,3]
parameters = initialize_parameters(dimArray)
use_batchnorm = False
AL, cache = L_model_forward(X, parameters,use_batchnorm )
print('Finished')