from numpy import zeros_like

class Momentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        # We store the velocity for weights and biases
        self.vW = None
        self.vb = None

    def update(self, layer):
        if self.vW is None:
            self.vW = zeros_like(layer.Weight)
            self.vb = zeros_like(layer.bias)

        # Velocity update
        self.vW = self.beta * self.vW + (1 - self.beta) * layer.dWeight
        self.vb = self.beta * self.vb + (1 - self.beta) * layer.dBias

        layer.Weight -= self.lr * self.vW
        layer.bias -= self.lr * self.vb

    def copy(self):
        new_copy = Momentum()
        new_copy.lr = self.lr
        new_copy.beta = self.beta
        # We store the velocity for weights and biases
        new_copy.vW = None if self.vW is None else self.vW.copy()
        new_copy.vb = None if self.vb is None else self.vb.copy()
        return new_copy