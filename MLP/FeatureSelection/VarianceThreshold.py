from numpy import ndarray, zeros
from MLP.utils import PipeValues


class VarianceThreshold:
    threshold: float = 0.0
    variance_: ndarray
    n_features_in_: ndarray


    def __init__(self, threshold: float = 0.0):
        self.__name__ = "VarianceThreshold"
        self.threshold = threshold
        self.variance_ = zeros((1, 1))

    def fit(self, X: ndarray, y: ndarray = None):
        if not X:
            raise ValueError("X is not provided")
        self.variance_ = X.var(axis=0)
        return self

    def transform(self, X: ndarray):
        real_X = X
        if isinstance(X, PipeValues):
            real_X = X.X

        filtered_X = real_X[self.variance_ > self.threshold]
        if isinstance(X, PipeValues):
            X.X = filtered_X
        return filtered_X

    def inverse_transform(self, X: ndarray):
        pass

    def fit_transform(self, X: ndarray, y: ndarray = None):
        real_X, real_y = X, y
        if isinstance(X, PipeValues):
            real_X, real_y = X.X, X.y

        self.fit(real_X, real_y)
        new_X = self.transform(real_X)

        if isinstance(X, PipeValues):
            X.X = new_X
            X.y = real_y

        return new_X