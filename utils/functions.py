''' Helper functions for the model'''


import numpy as np


def sigmoid(x):
    '''Compute the sigmoid of the given input'''
    return 1 /  (1 + np.exp(-x))

def relu(x):
    '''Compute the relu function of the given input'''
    return max(0, x)

def tanh(x):
    '''Compute the tanh of the given input'''
    return np.tanh(x)

def lrelu(x):
    '''Compute the leaky relu function of the given input'''
    return max(0.1 * x, x)

def compute_cost(A, Y):
    '''Compute the cost with a given output'''
    m = Y.shape[0]  # Number of traiing examples
    cost = np.sum(np.multiply(Y, np.log(A)) +
                  np.multiply((1 - Y), np.log(1 - A)))
    cost = np.squeeze(cost)
    return cost
