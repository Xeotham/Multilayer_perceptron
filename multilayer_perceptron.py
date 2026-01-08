import numpy as np
from pandas import read_csv
from MLP import MLP, train_test_split, cross_validation
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
    X_train, X_val, y_train, y_val = train_test_split(X, y)



    # mutinf = MutualInformation()

    # mutinf.fit(X_train, y_train)


    # pipe = Pipeline([('label_binarizer', LabelBinarizer()), ('scaler', StandardScaler()), ('mlp', MLP())])
    # pipe = make_pipeline(LabelBinarizer(), StandardScaler(), VarianceThreshold(), MLP(hidden_layers = (30, 30, 30), epochs=100000))
    pipe = make_pipeline(
        LabelBinarizer(),
        StandardScaler(),
        VarianceThreshold(),
        MutualInformation(4),
        MLP(hidden_layers = (10, 10, 10), epochs=1000)
    )

    # pipe.fit(X_train, y_train)
    # print(pipe.predict(X_val))
    # var_thre = VarianceThreshold(threshold = 1e-3)
    # print(var_thre.fit_transform(X_train).shape)

    # label_binarizer = LabelBinarizer()
    # label_binarizer.fit(y_train)
    # print(label_binarizer.classes_)
    # print(label_binarizer.transform(y_train))
    # print(y_train)
    # print(label_binarizer.inverse_transform(label_binarizer.transform(y_train)))

    # std_scaler = StandardScaler()
    # sk_std_scaler = SkStandardScaler()
    # std_scaler.fit(X_train)
    # sk_std_scaler.fit(X_train)
    # print(f"std_scaler mean: {len(std_scaler.mean_)}")
    # print(f"sk_std_scaler mean: {len(std_scaler.mean_)}")
    # print(X_train.shape)
    # print(f"std_scaler transform{std_scaler.transform(X_train)}")
    # print()
    # print(f"sk_std_scaler transform{sk_std_scaler.transform(X_train)}")
    # print(standardize(X_train))


    # multilayer_perceptron.fit(X_train, y_train.flatten())
    # y_pred = multilayer_perceptron.predict(X_val)
    # y_pred = one_hot_decoder(y_pred, unique(y_train))

    # tr_y_val_mask = y_val == 'M'
    # tr_y_val = y_val.copy()
    # tr_y_val[tr_y_val_mask] = 0
    # tr_y_val[~tr_y_val_mask] = 1
    #
    # tr_y_pred_mask = y_pred == 'M'
    # tr_y_pred = y_pred.copy()
    # tr_y_pred[tr_y_pred_mask] = 0
    # tr_y_pred[~tr_y_pred_mask] = 1

    print(f"Accuracy: {cross_validation(X, y, pipe)}")



if __name__ == "__main__":
    main()

