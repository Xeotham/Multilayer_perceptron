from typing import Callable

from utils import sigmoid

from numpy import ndarray, array, sum
from numpy.random import randn

class Layers:
    Weight: ndarray
    bias:   ndarray
    n_curr: int
    n_prev: int
    prev_layer = None
    next_layer = None

    Z: ndarray
    activation: ndarray

    dZ: ndarray
    dWeight: ndarray
    dBias: ndarray

    dZ_calculator: Callable

    def dZf(self, y: ndarray) -> ndarray:
        return self.activation - y

    def dZc(self, y: ndarray) -> ndarray:
        c = self.next_layer
        return c.Weight.T.dot(c.dZ) * (self.activation * (1 - self.activation))

    def dWc(self):
        return (1 / self.n_curr) * (self.dZ.dot(self.prev_layer.activation.T))

    def dBc(self):
        return (1 / self.n_curr) * sum(self.dZ, axis=1, keepdims=True)


    def __init__(self, n_curr, n_prev, prev_layer = None, next_layer = None):
        self.n_curr = n_curr
        self.n_prev = n_prev
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        self.Weight = randn(n_curr, n_prev)
        self.bias = randn(n_curr, 1)
        if next_layer is None:
            self.dZ_calculator = self.dZf
        else:
            self.dZ_calculator = self.dZc


    def forward(self, X) -> ndarray:
        self.Z = self.Weight.dot(X) + self.bias
        self.activation = sigmoid(self.Z)
        if self.next_layer is not None:
            return self.next_layer.forward(self.activation)
        else:
            return self.activation

    def backward(self, dY = None):
        self.dWeight = self.dWc()
        self.dBias = self.dBc()
        self.dZ = self.dZ_calculator(dY)
        if self.prev_layer is not None:
            return self.prev_layer.backward()
        else:
            return self.activation