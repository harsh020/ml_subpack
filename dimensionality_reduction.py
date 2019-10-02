import random
import numpy as np
from numpy.linalg import eig

from utils.utility import mean_normalize

class PCA:
    def __init__(self, k_dim=2):
        self.k_dim = k_dim

        return

    def fit(self, X):
        m, n = X.shape
        X = mean_normalize(X)

        # cov_X = 1/m * np.cov(X)
        cov_X = 1/m * (X.T @ X)
        w, v = eig(cov_X)
        self._new_basis = v[:, :self.k_dim]

        return self

    def transform(self, X):
        # if not self._new_basis:
        #     raise ValueError('Model not fit. Fit the data first.')

        X = mean_normalize(X)
        Xt = []
        for i in range(X.shape[0]):
            xt = X[i, :] @ self._new_basis
            Xt.append(xt)

        Xt = np.array(Xt)

        return Xt

    def inverse_transform(self, X):
        pass
