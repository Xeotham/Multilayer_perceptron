from typing import Callable

from .utils import sigmoid, relu, softmax

from MLP.Optimizer import Adam, GD, Momentum

from numpy import ndarray, array, sum, zeros, sqrt
from numpy.random import randn

activation_dict = {
    "sigmoid": sigmoid,
    "relu": relu,
    "softmax": softmax,
}

optimizer_dict = {
    "adam": Adam,
    "gd": GD,
    "momentum": Momentum,
}

class Layers:
    name: str

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
    activation_function: Callable = None
    optimizer = None

    def dZf(self, y: ndarray) -> ndarray:
        # print(f"dZf y: {y}")
        # print(f"dZf act: {self.activation}")
        return self.activation - y

    def dZc(self, y: ndarray) -> ndarray:
        c = self.next_layer
        return c.Weight.T.dot(c.dZ) * self.activation_function.derivative(self.activation)

    def dWc(self):
        if self.prev_layer is not None:
            prev_activation = self.prev_layer.activation
        else:
            prev_activation = self.X
        m = prev_activation.shape[1]  # Number of samples in batch
        return (self.dZ.dot(prev_activation.T)) / m

    def dBc(self):
        m = self.dZ.shape[1]
        return sum(self.dZ, axis=1, keepdims=True) / m


    def __init__(self, n_curr, n_prev = 0, prev_layer = None, next_layer = None, activation = "relu",optimizer = "adam", learning_rate = 0.01, name = None):
        self.name = name

        self.n_curr = n_curr
        self.n_prev = n_prev
        self.prev_layer = prev_layer
        self.next_layer = next_layer

        self.activation_function = activation_dict[activation]
        self.optimizer = optimizer_dict[optimizer](learning_rate = learning_rate)

        self.init_weights()

    def init_weights(self):
        if self.n_prev == 0:
            return
        self.Weight = randn(self.n_curr, self.n_prev) * sqrt(2. / self.n_prev)
        self.bias = zeros((self.n_curr, 1))


    def forward(self, X) -> ndarray:
        if not self.prev_layer:
            self.X = X.copy()
        # print(f"Shape of X: {X.shape} {type(X[0, 0])}")
        # print(f"Shape of Weight: {self.Weight.shape}")
        self.Z = self.Weight.dot(X) + self.bias
        self.activation = self.activation_function(self.Z.astype(float))
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

    def update(self):
        self.optimizer.update(self)
        if self.next_layer is not None:
            self.next_layer.update()

    def predict(self, X):
        return self.forward(X)

    def copy(self):
        new_layer = Layers(self.n_curr, self.n_prev)

        new_layer.name = self.name
        new_layer.Weight = self.Weight.copy()
        new_layer.bias = self.bias.copy()
        if not self.prev_layer:
            new_layer.X = self.X.copy()
        new_layer.Z = self.Z.copy()
        new_layer.activation = self.activation
        new_layer.dZ = self.dZ.copy()
        new_layer.dWeight = self.dWeight.copy()
        new_layer.dBias = self.dBias.copy()
        new_layer.dZ_calculator = self.dZ_calculator
        new_layer.activation_function = self.activation_function
        new_layer.optimizer = self.optimizer

        if self.next_layer:
            print(new_layer.name)
            new_layer.next_layer = self.next_layer.copy()
            new_layer.next_layer.prev_layer = new_layer
        return new_layer