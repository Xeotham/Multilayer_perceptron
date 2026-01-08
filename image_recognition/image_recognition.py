from sys import path
path.append("..")
path.append(".")
from tensorflow.keras.datasets import mnist
from MLP import MLP
from MLP.Pipeline import make_pipeline
from MLP.Preprocessing import LabelBinarizer
from numpy import array, ndarray
import matplotlib.pyplot as plt


# Load the MNIST dataset
(X_train, y_train), (X_val, y_val) = mnist.load_data()

X_train: ndarray = X_train.astype(float) / 255.0
X_val: ndarray = X_val / 255.0

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1] * X_val.shape[2]))

mlp = make_pipeline(LabelBinarizer(), MLP(hidden_layers = (128, 64), epochs=50, learning_rate=0.001))
mlp.fit(X_train, y_train)
pred = mlp.predict(X_val)
acc = sum(pred == y_val) / len(y_val)
print(f"Accuracy: {acc}")