import numpy as np
from pandas import read_csv
from MLP import MLP, train_test_split, cross_validation, Layers
from MLP.Preprocessing import LabelBinarizer, StandardScaler
from MLP.FeatureSelection import VarianceThreshold, MutualInformation
from MLP.Pipeline import Pipeline, make_pipeline
from numpy import unique, mean


def main():
    # multilayer_perceptron = MLP(hidden_layers = (30, 30, 30), epochs=100000)

    df = read_csv('./data/train_set.csv', index_col=None, header=None)
    X = df.values[:, 2:].astype(float)
    y = df.values[:, 1]
    # print(X)

    multilayer_perceptron = MLP(hidden_layers = (
                30,
                30,
                Layers(2, activation="softmax", name="Output Layer")
            ),
            epochs=1000,
            learning_rate=0.001
        )

    X_train, X_val, y_train, y_val = train_test_split(X, y)
    pipe = make_pipeline(
        LabelBinarizer(),
        StandardScaler(),
        VarianceThreshold(),
        # MutualInformation(26),
        multilayer_perceptron
    )

    pipe.fit(X_train, y_train)
    multilayer_perceptron.show_plots()
    # print(pipe.predict(X_val))



    print(f"Accuracy: {cross_validation(X, y, pipe)}")



if __name__ == "__main__":
    main()

