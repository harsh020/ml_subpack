import numpy as np
from utils.utility import gaussian_model, multinomial_model

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
        m = 0
        for v in data_dict.values():
            m += len(v)
        for key, values in data_dict.items():
            _means = np.mean(values, axis=0)
            if isinstance(self, GaussianNB):
                _var = np.var(values, axis=0)

            elif isinstance(self, MultinomialNB):
                _sigma = np.zeros((values.shape[1], values.shape[1]))
                for i in range(values.shape[0]):
                    _sigma += ((values[i, :] -
                                _means).reshape(len(values[i, :]), 1) @
                               (values[i, :] -
                                _means).reshape(1, len(values[i, :])))

                _var = (1/m) * _sigma

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


class MultinomialNB(_BaseClass):
    def __init__(self):
        pass

    def _multinomial_prob(self, X, stats):
        self.probability_ = {}
        for key, values in stats.items():
            self.probability_[key] = np.prod(
                                            multinomial_model(X,
                                            values[0],
                                            values[1]))

        return self.probability_


    def fit(self, X, y):
        self._stats = self._fit(X, y)

    def predict(self, X):
        predicted = []
        for i in range(X.shape[0]):
            p = self._multinomial_prob(X[i, :], self._stats)
            for c, pr in p.items():
                if pr == np.max(list(p.values())):
                    predicted.append(int(c))
                    break

        return predicted
