from numpy import array, ndarray, var, mean, sqrt

class StandardScaler:
    scale_: ndarray
    mean_:  ndarray
    var_:   ndarray

    def __init__(self):
        pass

    def fit(self, X: ndarray, y = None):
        self.var_ = var(X, axis=0)
        self.mean_ = mean(X, axis=0)
        self.scale_ = sqrt(self.var_)
        return self

    def _pipe_transform(self, values):
        X, y = values
        X = self.transform(X)
        values[0] = X
        return X

    def transform(self, X: ndarray):
        return (X.copy() - self.mean_) / self.scale_

    def inverse_transform(self, X: ndarray):
        return (X.copy() * self.scale_) + self.mean_

    def _pipe_fit_transform(self, values):
        X, y = values
        X = self.fit_transform(X)
        values[0] = X
        return X

    def fit_transform(self, X: ndarray, y = None):
        self.fit(X, y)
        return self.transform(X)