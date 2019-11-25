import numpy as np
import sys
sys.path.append('..')
from utils import *
from meta import *


class NN(ModelMeta):

    def __init__(self, input_shape, dimensions=[2,4]):
        '''
            Dimensions is the model's dimensions.
            dimensions[0] = input shape
            dimensions[1] = 1st hidden layer
            ...
        '''
        self.weights = []
        self.biases = []
        for i in range(1, len(dimensions)):
            prev = dimensions[i - 1]
            current = dimensions[i]
            W, b = init_parameters(prev, current)
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, activation="relu"):
        '''
            Forward pass of the neural network.
            If activation is given use it. Options: sigmoid, tanh, relu
            If it is not given: Use relu as default activation function.
        '''


    def backward(self):


    def cost_function(self):
