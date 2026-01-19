from numpy import array, ndarray, var, mean, sqrt
from ML_MHAOUAS.utils import PipeValues

class StandardScaler:
    scale_: ndarray
    mean_:  ndarray
    var_:   ndarray

    def __init__(self):
        pass

    def fit(self, X: ndarray):
        self.var_ = var(X, axis=0)
        self.mean_ = mean(X, axis=0)
        self.scale_ = sqrt(self.var_)
        return self

    def transform(self, X: ndarray):
        real_X = X
        if isinstance(X, PipeValues):
            real_X = X.X

        norm_X = (real_X - self.mean_) / self.scale_

        if isinstance(X, PipeValues):
            X.X = norm_X
        return norm_X

    def inverse_transform(self, X: ndarray):
        real_X = X
        if isinstance(X, PipeValues):
            real_X = X.X
        unnorm_X = (real_X * self.scale_) + self.mean_

        if isinstance(X, PipeValues):
            X.X = unnorm_X
        return unnorm_X

    def fit_transform(self, X: ndarray| PipeValues):
        real_X = X
        if isinstance(X, PipeValues):
            real_X, real_y = X.X, X.y
        self.fit(real_X)

        transformed_X = self.transform(real_X)
        if isinstance(X, PipeValues):
            X.X = transformed_X
        return transformed_X

    def copy(self):
        new_scaler = StandardScaler()
        new_scaler.scale_ = self.scale_.copy()
        new_scaler.mean_ = self.mean_.copy()
        new_scaler.mean_ = self.mean_.copy()
        return new_scaler