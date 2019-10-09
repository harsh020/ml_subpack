import random
import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

from ..utils.utility import mean_normalize

class PCA:
    def __init__(self, k_dim=2):
        self.k_dim = k_dim

        return

    def fit(self, X):
        m, n = X.shape
        X, mu = mean_normalize(X)
        # cov_X = 1/m * np.cov(X)
        cov_X = 1/m * (X.T @ X)
        U, S, vh = svd(cov_X)
        self.components_ = U

        return self

    def transform(self, X):
        # if not self.components_:
        #     raise ValueError('Model not fit. Fit the data first.')

        X, mu = mean_normalize(X)
        _new_basis = self.components_[:, :self.k_dim]
        Xt = []
        for i in range(X.shape[0]):
            xt = X[i, :] @ _new_basis
            Xt.append(xt)

        Xt = np.array(Xt)

        return Xt

    def inverse_transform(self, Xt):
        # if not self.components_:
        #     raise ValueError('Model not fit. Fit the data first.')

        _new_basis = self.components_[:, :self.k_dim]
        X_approx = []
        for i in range(Xt.shape[0]):
            xt = _new_basis @ Xt[i, :].reshape(Xt.shape[1], 1)
            X_approx.append(xt)

        X_approx = np.array(X_approx)

        return X_approx
