from numpy import ndarray, unique, zeros, array
from MLP.utils import PipeValues

class LabelBinarizer:
    classes_: ndarray

    def __init__(self):
        pass

    def fit(self, y: ndarray):
        self.classes_ = unique(y)
        return self

    def transform(self, y: ndarray) -> ndarray:
        real_y = y

        if isinstance(y, PipeValues):
            real_y = y.y

        converted_values = (real_y[:, None] == self.classes_).astype(int)

        if isinstance(y, PipeValues):
            y.y = converted_values
        return converted_values

    def inverse_transform(self, y: ndarray | PipeValues) -> ndarray:
        real_y = y
        if isinstance(y, PipeValues):
            real_y = y.y
        decoded_list = []
        for i, v_list in enumerate(real_y):
            for label, values in zip(self.classes_, v_list):
                if values == 1:
                    decoded_list.append(label)
        inv_arr = array(decoded_list)
        if isinstance(y, PipeValues):
            y.y = inv_arr
        return inv_arr

    def fit_transform(self, y: ndarray | PipeValues, X = None):
        real_X, real_y = 0, y
        if isinstance(y, PipeValues):
            real_X, real_y = y.X, y.y
        if real_y is None:
            return real_y
        self.fit(real_y)
        transformed_y = self.transform(real_y)
        if isinstance(y, PipeValues):
            y.y = transformed_y
        return transformed_y

    def copy(self):
        new_copy = LabelBinarizer()
        new_copy.classes_ = self.classes_.copy()
        return new_copy