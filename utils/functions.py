''' Helper functions for the model'''


import numpy as np


def sigmoid(x):
    '''Compute the sigmoid of the given input'''
    return 1 /  (1 + np.exp(-x))

def relu(x):
    '''Compute the relu function of the given input'''
    return np.maximum(0, x)

def tanh(x):
    '''Compute the tanh of the given input'''
    return np.tanh(x)

def lrelu(x):
    '''Compute the leaky relu function of the given input'''
    return np.maximum(0.1 * x, x)

def compute_cost(A, Y):
    '''Compute the cost with a given output'''
    m = Y.shape[1]  # Number of traiing examples
    cost = -(1/m) * np.sum(np.multiply(Y, np.log(A)) +
                  np.multiply((1 - Y), np.log(1 - A)))
    cost = np.squeeze(cost)
    return cost

def linear_backwards(dZ, cache):
    '''
    Used to calculate dcost/dW and dcost/db from dZ
    Z = WX + b (Linear part of the neuron)
    '''
    _, A_prev, W, b = cache  # Unpack the cache = (Z, A_prev, W, b)
    m = A_prev.shape[1]
    dW = np.multiply(dZ, np.transpose(A_prev))
    # db =

def activation_backwards(dA, cache, activation):
    if activation == 'sigmoid':
        pass
