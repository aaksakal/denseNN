import numpy as np
import sys
sys.path.append('..')
from utils.initialize import init_parameters
from utils.functions import *
from meta import *


class NN(ModelMeta):

    def __init__(self, X, Y, dimensions=[2,4,1]):
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
        # Shape of X: (num_examples, num_of_features)
        # Shape of Y: (num_examples, outputs)
        self.m = X.shape[0]  # Number of training examples
        input_shape = X.shape[1]  # Number of features
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

    def forward(self, activations=None):
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
            activation = activations[i - 1]
            Z = np.matmul(np.transpose(W), A_prev)
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
        return A, cache

    def backward(self):
        pass


def test_init(model):
    for i in range(3):
        print("W{} = ".format(i + 1), model.parameters["W{}".format(i + 1)])
        print("b{} = ".format(i + 1), model.parameters["b{}".format(i + 1)])

if __name__ == "__main__":
    np.random.seed(1)
    X = np.random.randn(3, 6)
    Y = np.random.randn(3, 1)
    model = NN(X, Y)
    test_init(model)
