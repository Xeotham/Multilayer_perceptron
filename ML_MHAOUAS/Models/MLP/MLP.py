from numpy.random import permutation

from .Layers import Layers
from ML_MHAOUAS.preprocessing import train_test_split
from ML_MHAOUAS.utils import softmax
from numpy import log, ndarray, max, unique, array, clip, mean, log, arange, zeros, sum
from matplotlib.pyplot import plot, show, subplots, legend

class MLP:
    # Setup
    hidden_layers:  tuple
    learning_rate:  float
    epochs:         int
    layers:         list[Layers]
    input_layer:    Layers
    output_layer:   Layers

    batch_size: int

    # Datasets for training
    X_train:    ndarray
    y_train:    ndarray

    # Datasets for validation
    X_val:      ndarray
    y_val:      ndarray

    # Graphs
    cost_evolution: list
    val_cost_evolution: list

    def __init__(self, hidden_layers = (10, 10, 10), learning_rate = 1e-3, epochs = 1000, batch_size = 32, patience = 50):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.cost_evolution = []
        self.val_cost_evolution = []

        self.layers = []

        # Create the list of layers
        for i, layer in enumerate(hidden_layers):
            if i == 0:
                prev_layer = None
                prev_layer_size = 0
            elif isinstance(hidden_layers[i - 1], Layers):
                prev_layer = self.layers[i - 1]
                prev_layer_size = hidden_layers[i - 1].n_prev
            elif isinstance(hidden_layers[i - 1], int):
                prev_layer = self.layers[i - 1]
                prev_layer_size = hidden_layers[i - 1]
            else:
                raise TypeError("Invalid type for hidden_layers. Layers must be of type int or Layers")

            if isinstance(layer, Layers):
                new_layer = layer
                new_layer.prev_layer = prev_layer
                new_layer.n_prev = prev_layer_size
            elif isinstance(layer, int):

                if i == len(hidden_layers) - 1:
                    name = "Output Layer"
                    if layer > 1:
                        activation = "softmax"
                    else:
                        activation = "sigmoid"
                else:
                    activation = "relu"
                    name = f"Hidden Layer {i}"
                new_layer = Layers(layer, prev_layer_size, prev_layer = prev_layer, activation=activation, name = name)
            else:
                raise TypeError("Invalid type for hidden_layers. Layers must be of type int or Layers")
            self.layers.append(new_layer)

        # Clean the setup of each layers
        for i, layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                layer.next_layer = self.layers[i + 1]
            else:
                layer.next_layer = None
            if layer.name is None:
                layer.name = f"Hidden Layer {i}"
            if i == 0:
                continue
            layer.n_prev = self.layers[i - 1].n_curr
            layer.init_weights()
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def log_loss(self, y_true):
        epsilon = 1e-15
        y_pred = clip(self.output_layer.activation, epsilon, 1 - epsilon)
        if y_true.shape[0] > 1:  # Multi-class
            return -mean(sum(y_true * log(y_pred), axis=0))
        return -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    def show_plots(self):
        fig, axs = subplots(1, 2)
        axs[0].plot(array(self.cost_evolution).T)
        axs[0].set_title("Cost Function")
        axs[1].plot(array(self.val_cost_evolution).T)
        axs[1].set_title("Validation Cost Function")
        show()

    def fit(self, X, y):

        self.X_val, self.X_train, self.y_val, self.y_train = train_test_split(X, y)

        if self.y_train.shape[1] != self.output_layer.n_curr:
            raise RuntimeError("Output layer has wrong shape")

        self.input_layer.n_prev = self.X_train.T.shape[0]
        self.input_layer.init_weights()

        best_val_cost = float("inf")
        best_val_index = 0
        patience_counter = 0

        for i in range(self.epochs):

            self.input_layer.forward(self.X_val.T)
            val_cost = self.log_loss(self.y_val.T)

            self.input_layer.forward(self.X_train.T)
            train_cost = self.log_loss(self.y_train.T)

            print(f"Epoch {i} / {self.epochs} - train_cost: {train_cost:.3} val_cost: {val_cost:.3}")

            self.val_cost_evolution.append(val_cost)
            self.cost_evolution.append(train_cost)

            if val_cost < best_val_cost:
                best_val_cost = val_cost
                best_val_index = i
                self.input_layer.save_curr_state()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience or i == self.epochs - 1:
                self.input_layer.load_save_state()
                # print(f"Best val index: {best_val_index}")
                self.cost_evolution = self.cost_evolution[:best_val_index]
                self.val_cost_evolution = self.val_cost_evolution[:best_val_index]
                return self


            shuffled_indices = permutation(self.X_train.shape[0])

            X_shuffle = self.X_train[shuffled_indices]
            y_shuffle = self.y_train[shuffled_indices]
            X_train_batch = [X_shuffle[i * self.batch_size:(i + 1) * self.batch_size, :] for i in range(self.X_train.shape[0] // self.batch_size)]
            y_train_batch = [y_shuffle[i * self.batch_size:(i + 1) * self.batch_size, :] for i in range(self.y_train.shape[0] // self.batch_size)]

            for X_batch, y_batch in zip(X_train_batch, y_train_batch):
                self.input_layer.forward(X_batch.T)
                self.output_layer.backward(y_batch.T)
                self.input_layer.update()

        return self

    def predict(self, X):
        raw_pred = self.input_layer.predict(X.T)
        # Get index of highest probability
        idx = raw_pred.argmax(axis=0)
        # Create one-hot matrix
        out = zeros(raw_pred.shape)
        out[idx, arange(raw_pred.shape[1])] = 1
        return out.T.astype(int)

    def copy(self):
        mlp_copy = MLP()
        mlp_copy.hidden_layers = self.hidden_layers
        mlp_copy.learning_rate = self.learning_rate
        mlp_copy.epochs = self.epochs

        mlp_copy.input_layer = self.input_layer.copy()

        mlp_copy.layers = []
        curr_layer = mlp_copy.input_layer
        while curr_layer is not None:
            curr_layer = curr_layer.next_layer
            if curr_layer is not None:
                mlp_copy.layers.append(curr_layer)

        mlp_copy.output_layer = mlp_copy.layers[-1]

        mlp_copy.batch_size = self.batch_size

        mlp_copy.X_train = self.X_train.copy()
        mlp_copy.y_train = self.y_train.copy()
        mlp_copy.X_val = self.X_val.copy()
        mlp_copy.y_val = self.y_val.copy()

        mlp_copy.cost_evolution = self.cost_evolution.copy()
        mlp_copy.val_cost_evolution = self.val_cost_evolution.copy()
        return mlp_copy
