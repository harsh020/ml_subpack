import numpy as np
from utils.utility import gaussian_model

class _BaseClass:
    def _group_by_class(self, X, y):
        data = {}
        # try to figure out a way to do this without loop
        for c in list(set(y)):
            x = []
            for i in range(len(y)):
                if y[i] == c:
                    x.append(list(X[i, :]))

            data[c] = np.array(x).reshape(len(x), X.shape[1])

        return data


    def _statistic(self, data_dict):
        stats = {}
        for key, values in data_dict.items():
            _means = np.mean(values, axis=0)
            _var = np.var(values, axis=0)
            stats[key] = (_means, _var)

        return stats


    def _fit(self, X, y):
        data_dict = self._group_by_class(X, y)
        stats = self._statistic(data_dict)

        return stats


class GaussianNB(_BaseClass):
    def __init__(self):
        pass

    def _gaussian_prob(self, X, stats):
        self.probability_ = {}
        for key, values in stats.items():
            self.probability_[key] = np.prod(
                                            gaussian_model(X,
                                            values[0],
                                            values[1]))

        return self.probability_


    def fit(self, X, y):
        self._stats = self._fit(X, y)

    def predict(self, X):
        predicted = []
        for i in range(X.shape[0]):
            p = self._gaussian_prob(X[i, :], self._stats)
            for c, pr in p.items():
                if pr == np.max(list(p.values())):
                    predicted.append(int(c))
                    break

        return predicted
