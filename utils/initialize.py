import numpy as np


def init_parameters(n_x, n_h):
    '''
        Initializes the weights for a given layer
        Input:  n_x: Input shape of the previous layer.
                n_h: Dimension of the current layer.
        Output:
                W, b: Weight matrix and bias matrix for the given layer.
    '''
    W = np.random.rand(n_h, n_x) * 0.01
    b = np.zeros((n_h, 1))
    return W, b

if __name__ == '__main__':
    W, b = init_parameters(12, 5)
    print("Weights shape = " , W.shape)
    print(W)
    print("Bias vector shape = ", b.shape)
    print(b)
