from numpy import zeros_like, sqrt

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        # First and second moments
        self.mW, self.mb = None, None
        self.vW, self.vb = None, None

    def update(self, layer):
        if self.mW is None:
            self.mW = zeros_like(layer.Weight)
            self.mb = zeros_like(layer.bias)
            self.vW = zeros_like(layer.Weight)
            self.vb = zeros_like(layer.bias)

        self.t += 1

        dWeight_reg = layer.dWeight + self.weight_decay * layer.Weight

        # Update first moment (Momentum)
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dWeight_reg
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * layer.dBias

        # Update second moment (RMSProp)
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dWeight_reg ** 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (layer.dBias ** 2)

        # Bias correction
        m_hat_W = self.mW / (1 - self.beta1 ** self.t)
        m_hat_b = self.mb / (1 - self.beta1 ** self.t)
        v_hat_W = self.vW / (1 - self.beta2 ** self.t)
        v_hat_b = self.vb / (1 - self.beta2 ** self.t)

        # Update Weights and Bias
        layer.Weight -= self.lr * m_hat_W / (sqrt(v_hat_W) + self.epsilon)
        layer.bias -= self.lr * m_hat_b / (sqrt(v_hat_b) + self.epsilon)

    def copy(self):
        new_copy = Adam()

        new_copy.lr = self.lr
        new_copy.weight_decay = self.weight_decay
        new_copy.beta1 = self.beta1
        new_copy.beta2 = self.beta2
        new_copy.epsilon = self.epsilon
        new_copy.t = self.t

        new_copy.mW = None if self.mW is None else self.mW.copy()
        new_copy.mb = None if self.mb is None else self.mb.copy()
        new_copy.vW = None if self.vW is None else self.vW.copy()
        new_copy.vb = None if self.vb is None else self.vb.copy()
        return new_copy
