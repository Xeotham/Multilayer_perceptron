from numpy import ndarray, unique, zeros, array

class LabelBinarizer():
    classes_: ndarray

    def __init__(self):
        super().__init__()

    def fit(self, y: ndarray):
        self.classes_ = unique(y)
        return self

    def _pipe_transform(self, values):
        X, y = values
        y = self.transform(y)
        values[1] = y
        return y

    def transform(self, y: ndarray) -> ndarray:
        converted_values = zeros((len(y), len(self.classes_)), dtype=int)
        for i, v in enumerate(y):
            for j, u in enumerate(self.classes_):
                converted_values[i, j] = int(u == v)
        return converted_values

    def inverse_transform(self, y: ndarray) -> ndarray:
        decoded_list = []
        for i, v_list in enumerate(y):
            for label, values in zip(self.classes_, v_list):
                if values == 1:
                    decoded_list.append(label)
        return array(decoded_list)

    def _pipe_fit_transform(self, values):
        X, y = values
        y = self.fit_transform(y)
        values[1] = y
        return y

    def fit_transform(self, y: ndarray):
        self.fit(y)
        return self.transform(y)
