from .Layers import Layers
from .preprocessing import train_test_split
from .utils import softmax
from numpy import log, ndarray, max, unique, array
from tqdm import tqdm
from matplotlib.pyplot import plot, show

class MLP:
    # Setup
    hidden_layers:  tuple
    learning_rate:  float
    epochs:         int
    layers:         list[Layers]
    input_layer:    Layers
    output_layer:   Layers

    # Datasets for training
    X_train:    ndarray
    y_train:    ndarray
    ohe_ytrain: ndarray

    # Datasets for validation
    X_val:      ndarray
    y_val:      ndarray

    # Graphs
    cost_evolution: list

    def __init__(self, hidden_layers = (10, 10), learning_rate = 1e-3, epochs = 1000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_evolution = []

        self.layers = [Layers(hidden_layers[0], 0)]

        for i in range(1, len(hidden_layers)):
            new_layer = Layers(hidden_layers[i], hidden_layers[i - 1], prev_layer = self.layers[i - 1])
            self.layers[i - 1].next_layer = new_layer
            self.layers.append(new_layer)
        self.input_layer = self.layers[0]

    def log_loss(self, y) -> float:
        # ohe_y = one_hot_encoder(y)
        raw_pred = self.input_layer.predict(self.X_train)
        softmax_pred = softmax(raw_pred).T
        bigger_pred = softmax_pred.T[0, :]
        y_cpy = array(y == unique(self.y_train)[0], dtype=int)
        bigger_pred = array(bigger_pred)

        error = 1/len(y) * sum(-y_cpy * log(bigger_pred) - (1 - y_cpy) * log(1 - bigger_pred))
        return error

    def show_plots(self):
        print(array(self.cost_evolution).T)
        plot(array(self.cost_evolution).T)
        show()

    def fit(self, X, y):

        self.X_val, self.X_train, self.y_val, self.y_train = train_test_split(X, y)

        self.input_layer.n_prev = self.X_train.T.shape[0]
        self.input_layer.init_weights()

        # self.ohe_y_train = one_hot_encoder(y_train)

        penult_layer = self.layers[-1]
        self.output_layer = Layers(self.ohe_y_train.shape[1], penult_layer.n_curr, prev_layer = penult_layer)
        penult_layer.next_layer = self.output_layer

        for i in tqdm(range(self.epochs)):
            self.input_layer.forward(self.X_train)

            cost = self.log_loss(y_train)
            # if i % 10 == 0:
            self.cost_evolution.append(cost)

            self.output_layer.backward(self.ohe_y_train.T)
            self.input_layer.update(self.learning_rate)
        return self

    # def predict(self, X, one_hot = True):
        # norm_X = standardize(X.T.copy()).astype(float)
        # raw_pred = self.input_layer.predict(norm_X)
        # softmax_pred = softmax(raw_pred).T

        # for i, pred in enumerate(softmax_pred):
        #     bigger = max(pred)
        #     for j, value in enumerate(pred):
        #         if value == bigger:
        #             softmax_pred[i, j] = 1
        #         else:
        #             softmax_pred[i, j] = 0
        # if one_hot:
        #     return one_hot_decoder(softmax_pred.astype(int), unique(self.y_train))
        # else:
        #     return softmax_pred.astype(int)

    def __copy__(self):
        new_mlp = MLP(hidden_layers = self.hidden_layers, learning_rate = self.learning_rate, epochs = self.epochs)
