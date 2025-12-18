from typing import Callable

from .utils import sigmoid

from numpy import ndarray, array, sum
from numpy.random import randn

class Layers:
    Weight: ndarray
    bias:   ndarray
    n_curr: int
    n_prev: int
    prev_layer = None
    next_layer = None

    X: ndarray
    Z: ndarray
    activation: ndarray

    dZ: ndarray
    dWeight: ndarray
    dBias: ndarray

    dZ_calculator: Callable = None

    def dZf(self, y: ndarray) -> ndarray:
        # print(f"dZf y: {y}")
        # print(f"dZf act: {self.activation}")
        return self.activation - y

    def dZc(self, y: ndarray) -> ndarray:
        c = self.next_layer
        return c.Weight.T.dot(c.dZ) * (self.activation * (1 - self.activation))

    def dWc(self):
        if self.prev_layer is not None:
            prev_activation = self.prev_layer.activation
        else:
            prev_activation = self.X
        return (self.dZ.dot(prev_activation.T)) / self.n_curr

    def dBc(self):
        return sum(self.dZ, axis=1, keepdims=True) / self.n_curr


    def __init__(self, n_curr, n_prev, prev_layer = None, next_layer = None):
        self.n_curr = n_curr
        self.n_prev = n_prev
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        self.init_weights()

    def init_weights(self):
        self.Weight = randn(self.n_curr, self.n_prev)
        self.bias = randn(self.n_curr, 1)


    def forward(self, X) -> ndarray:
        if not self.prev_layer:
            self.X = X.copy()
        # print(f"Shape of X: {X.shape} {type(X[0, 0])}")
        # print(f"Shape of Weight: {self.Weight.shape}")
        self.Z = self.Weight.dot(X) + self.bias
        self.activation = sigmoid(self.Z.astype(float))
        if self.next_layer is not None:
            return self.next_layer.forward(self.activation)
        else:
            return self.activation

    def backward(self, dY = None):
        if not self.dZ_calculator:
            if not self.next_layer:
                self.dZ_calculator = self.dZf
            else:
                self.dZ_calculator = self.dZc
        self.dZ = self.dZ_calculator(dY)
        self.dWeight = self.dWc()
        self.dBias = self.dBc()
        if self.prev_layer is not None:
            return self.prev_layer.backward()
        else:
            return self.activation

    def update(self, learning_rate):
        self.Weight = self.Weight - learning_rate * self.dWeight
        self.bias = self.bias - learning_rate * self.dBias
        if self.next_layer is not None:
            self.next_layer.update(learning_rate)

    def predict(self, X):
        return self.forward(X)

    def __copy__(self):
        new_layer = Layers(self.n_curr, self.n_prev)
        if self.prev_layer:
            new_layer.prev_layer = self.prev_layer.copy()
        if self.next_layer:
            new_layer.next_layer = self.next_layer.copy()