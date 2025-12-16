from .Layers import Layers
from .preprocessing import one_hot_encoder
from .utils import standardize, softmax
from numpy import float64, log, ndarray, max
from tqdm import tqdm


class MLP:
    hidden_layers:  tuple
    learning_rate:  float
    epochs:         int
    layers:         list[Layers]
    input_layer:    Layers
    output_layer:   Layers

    X_train: ndarray
    y_train: ndarray

    def __init__(self, hidden_layers = (10, 10), learning_rate = 1e-3, epochs = 1000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.layers = [Layers(hidden_layers[0], 0)]

        for i in range(1, len(hidden_layers)):
            new_layer = Layers(hidden_layers[i], hidden_layers[i - 1], prev_layer = self.layers[i - 1])
            self.layers[i - 1].next_layer = new_layer
            self.layers.append(new_layer)
        self.input_layer = self.layers[0]

    def cost_function(self, y) -> float:
        last_pred = self.output_layer.activation.flatten().copy()
        error = -1/len(y) * sum(y * log(last_pred) + (1 - y) * log(1 - last_pred))
        return error

    def fit(self, X, y):
        self.input_layer.n_prev = X.T.shape[0]
        self.input_layer.init_weights()

        self.X_train = X.T.copy()
        norm_X = standardize(self.X_train).astype(float)

        self.y_train = one_hot_encoder(y)
        penult_layer = self.layers[-1]
        self.output_layer = Layers(self.y_train.shape[1], penult_layer.n_curr, prev_layer = penult_layer)
        penult_layer.next_layer = self.output_layer

        for _ in tqdm(range(self.epochs)):
            self.input_layer.forward(norm_X)
            self.output_layer.backward(self.y_train.T)
            self.input_layer.update(self.learning_rate)

    def predict(self, X):
        norm_X = standardize(X.T.copy()).astype(float)
        raw_pred = self.input_layer.predict(norm_X)
        softmax_pred = softmax(raw_pred).T

        for i, pred in enumerate(softmax_pred):
            bigger = max(pred)
            for j, value in enumerate(pred):
                if value == bigger:
                    softmax_pred[i, j] = 1
                else:
                    softmax_pred[i, j] = 0
        # print(softmax_pred.astype(int))
        return softmax_pred.astype(int)