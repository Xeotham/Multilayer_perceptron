from numpy import ndarray, zeros, histogram, digitize, unique, log, argsort, array

from MLP.utils import PipeValues


class MutualInformation:

    mi_score = []
    number_of_features = 2
    prev_X = None
    prev_y = None

    def __init__(self, number_of_features = 2):
        self.number_of_features = number_of_features

    @staticmethod
    def _discretize(coll: ndarray) -> ndarray:
        hist_features, hist_bins = histogram(coll)
        return digitize(coll, hist_bins[1: -1], right=True)

    @staticmethod
    def _calculate_probability(col1: ndarray, col2: ndarray):
        """

        :param col1:
        :param col2:
        :return: probability table for col1 and col2, probability table for col1, probability table for col2
        """
        col1_classes = unique(col1)
        col2_classes = unique(col2)
        proba_table = zeros((len(col1_classes), len(col2_classes)))
        proba_c1 = zeros(col1_classes.shape)
        proba_c2 = zeros(col2_classes.shape)

        def _check_column(idx):
            for c1_idx, c1 in enumerate(col1_classes):
                if col1[idx] == c1:
                    proba_c1[c1_idx] += 1
                for c2_idx, c2 in enumerate(col2_classes):
                    if col1[idx] == c1 and col2[i] == c2:
                        proba_table[c1_idx, c2_idx] += 1
                        proba_c2[c2_idx] += 1
                        return

        for i in range(0, len(col1)):
            _check_column(i)

        return proba_table / len(col1), proba_c1 / len(col1), proba_c2 / len(col1)

    def _calculate_mutual_information(self, col1: ndarray, col2: ndarray) -> float:
        proba_table, proba_c1, proba_c2 = self._calculate_probability(col1, col2)
        mi = 0.0

        for i in range(proba_table.shape[0]):
            for j in range(proba_table.shape[1]):
                if proba_table[i, j] > 0:
                    mi += proba_table[i, j] * log(proba_table[i, j] / (proba_c1[i] * proba_c2[j] + 1e-10))
        return mi



    def fit(self, X: ndarray, y: ndarray = None):
        real_x, real_y = X, y
        self.mi_score = []

        if isinstance(X, PipeValues):
            real_x, real_y = X.X, X.y
        self.prev_X = real_x
        self.prev_y = real_y
        if real_y.shape[1] > 1:
            real_y = real_y[:, 0]
        X_copy = X.copy()
        for i in range(0, X_copy.shape[1]):
            if isinstance(X_copy[0, i], float):
                X_copy[:, i] = self._discretize(X_copy[:, i])
            self.mi_score.append(float(self._calculate_mutual_information(real_y, X_copy[:, i])))
        return self

    def transform(self, X: ndarray, y: ndarray = None):
        real_x, real_y = X, y
        if isinstance(X, PipeValues):
            real_x, real_y = X.X, X.y

        # print(self.mi_score)
        sorted_feature_indices = argsort(self.mi_score)[::-1]
        sorted_mask = array(range(0, len(sorted_feature_indices)))
        sorted_mask[sorted_feature_indices[:self.number_of_features]] = True
        sorted_mask[sorted_feature_indices[self.number_of_features:]] = False

        if isinstance(X, PipeValues):
            X.X = real_x[:, sorted_mask.astype(bool)]
        return real_x[:, sorted_mask.astype(bool)]


    def inverse_transform(self, X: ndarray):
        if isinstance(X, PipeValues):
            X.X = self.prev_X
        pass

    def fit_transform(self, X: ndarray, y: ndarray = None):
        real_x, real_y = X, y
        if isinstance(X, PipeValues):
            real_x, real_y = X.X, X.y
        self.fit(real_x, real_y)
        filtered_X = self.transform(real_x, real_y)

        if isinstance(X, PipeValues):
            X.X = filtered_X

        return filtered_X