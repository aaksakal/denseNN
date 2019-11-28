import numpy as np
import sys
sys.path.append('..')
from utils.initialize import init_parameters
from utils.functions import *
from meta import *


class NN(ModelMeta):

    def __init__(self, X, Y, dimensions=[2,4,1], activations=['relu', 'relu', 'sigmoid']):
        '''
            Input:
                X = Input features to be used in training
                Y = The actual outputs for the training
                dimensions = model's dimensions, i.e. the hidden layers.
            Output:
                None
            ...
        '''
        self.parameters = {}
        self.L = len(dimensions)  # Hiddent layer count
        self.X = X
        self.Y = Y
        self.activations = activations
        # Shape of X: (num_examples, num_of_features)
        # Shape of Y: (num_examples, outputs)
        self.m = X.shape[1]  # Number of training examples
        input_shape = X.shape[0]  # Number of features
        self.initialize(input_shape, dimensions)

    def initialize(self, input_shape, dimensions):
        ''' Initialize the weights and biases '''
        prev = input_shape
        for i in range(len(dimensions)):
            current = dimensions[i]
            W, b = init_parameters(prev, current)
            self.parameters["W" + str(i + 1)] = W
            self.parameters["b" + str(i + 1)] = b
            prev = current

    def forward(self):
        '''
            Forward pass of the neural network.
            If activation is given use it. Options: sigmoid, tanh, relu, lrelu
            If it is not given: Use relu as default activation function.
            Inputs: activation = activation function to be used.
            Output:
        '''
        A_prev = self.X
        cache = []
        for i in range(1, self.L + 1):
            W = self.parameters["W{}".format(i)]
            b = self.parameters["b{}".format(i)]
            activation = self.activations[i - 1]
            Z = np.dot(W, A_prev) + b
            if activation == 'relu':
                act_fun = relu
            elif activation == 'lrelu':
                act_fun = lrelu
            elif activation == 'sigmoid':
                act_fun = sigmoid
            elif activation == 'tanh':
                act_fun = tanh
            else:
                act_fun = relu
            A = act_fun(Z)
            cache.append((Z, A_prev, W, b))
            A_prev = A
        A_last = A
        return A_last, cache

    def backward(self, A_last, cache, activations):
        grads = {}
        Y = Y.reshape(A_last.shape)
        # Initialize the backward propagation dCost / dA_last
        dA_last = -(np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
        # for i in range


def test_init(model):
    W_shapes = [(2, 6), (4, 2), (1, 4)]
    b_shapes = [(2, 1), (4, 1), (1, 1)]
    for i in range(3):
        W_i = model.parameters["W{}".format(i + 1)]
        b_i = model.parameters["b{}".format(i + 1)]
        assert W_i.shape == W_shapes[i]
        assert b_i.shape == b_shapes[i]
        print("W{} = ".format(i + 1), W_i)
        print("b{} = ".format(i + 1), b_i)
    print("\nTEST SUCCESSFULLY COMPLETED!!")

def test_forward(model):
    AL, cache = model.forward()
    assert AL.shape == (1, 3)
    assert len(cache) == 3
    print("Shape of AL is {}".format(AL.shape))
    print("Len of the cache is {}".format(len(cache)))
    print("\nTEST SUCCESSFULLY COMPLETED!!")

if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(6, 3)
    Y = np.random.randn(1, 3)
    activations = []
    model = NN(X, Y)
    # test_init(model)
    test_forward(model)
