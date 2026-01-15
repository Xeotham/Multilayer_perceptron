class GD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, layer):
        layer.Weight -= self.lr * layer.dWeight
        layer.bias -= self.lr * layer.dBias

    def copy(self):
        return GD(learning_rate=self.lr)