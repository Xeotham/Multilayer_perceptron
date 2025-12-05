class MLP:
    weights = []
    accuracies = []
    losses = []
    learning_rates = 0.01
    n_epochs = 1000
    hidden_layer_sizes = (10, 10)
    bias = []

    def __init__(self, learning_rate=0.01, n_epochs=1000, hidden_layer_sizes=(10, 10)):
        self.weights = []
        self.bias = []
        self.accuracies = []
        self.losses = []
        self.learning_rates = learning_rate
        self.n_epochs = n_epochs
        self.hidden_layer_sizes = hidden_layer_sizes
