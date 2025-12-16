import numpy as np
from pandas import read_csv
from MLP import MLP, train_test_split, one_hot_decoder, cross_validation
from numpy import unique, mean


def main():
    multilayer_perceptron = MLP(hidden_layers = (30, 30, 30), epochs=100000)

    df = read_csv('./data/train_set.csv', index_col=None, header=None)
    X = df.values[:, 2:3].astype(float)
    y = df.values[:, 1]
    X_train, X_val, y_train, y_val = train_test_split(X, y)

    multilayer_perceptron.fit(X_train, y_train.flatten())
    y_pred = multilayer_perceptron.predict(X_val)
    y_pred = one_hot_decoder(y_pred, unique(y_train))

    # tr_y_val_mask = y_val == 'M'
    # tr_y_val = y_val.copy()
    # tr_y_val[tr_y_val_mask] = 0
    # tr_y_val[~tr_y_val_mask] = 1
    #
    # tr_y_pred_mask = y_pred == 'M'
    # tr_y_pred = y_pred.copy()
    # tr_y_pred[tr_y_pred_mask] = 0
    # tr_y_pred[~tr_y_pred_mask] = 1

    for i, j in zip(y_val, y_pred):
        print(i, j)
    print(f"Accuracy: {cross_validation(X, y, multilayer_perceptron)}")
    # y_mask = y == 'M'
    # y[y_mask] = 0
    # y[~y_mask] = 1



if __name__ == "__main__":
    main()

