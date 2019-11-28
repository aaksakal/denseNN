"""
Meta Class for the Deep Neural Network Model and Other Modules
====================================
It provides Abstract class for the Knowledge graph models.
"""


from abc import ABCMeta


class ModelMeta:

    __metaclass__ = ABCMeta

    def __init__(self, X, Y, dimensions=[2,4,1], activations=['relu', 'relu', 'sigmoid']):
        '''Constructor of the model'''
        pass

    def initialize(self, input_shape, dimensions):
        '''Initializes the weights according to input shape and dimensions'''
        pass

    def forward(self):
        '''Makes a forward pass in the neural network'''
        pass

    def backward(self):
        '''Makes a backward propagation pass and updates the weights'''
        pass
