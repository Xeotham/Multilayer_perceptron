from numpy import ndarray, zeros


class VarianceThreshold:
    threshold: float = 0.0
    variance_: ndarray
    n_features_in_: ndarray


    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.variance_ = zeros((1, 1))

    def fit(self, X: ndarray, y: ndarray = None):
        if not X:
            raise ValueError("X is not provided")
        self.variance_ = X.var(axis=0)
        return self

    def transform(self, X: ndarray):
        return X[self.variance_ > self.threshold]

    def inverse_transform(self, X: ndarray):
        pass

    def fit_transform(self, df: ndarray):
        self.variance_ = df.var(axis=0)
        return df[:, self.variance_ > self.threshold]