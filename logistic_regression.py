import numpy as np
from math import log
from utils.optimize import gradient_desc
from utils.utility import sigmoid
from scipy.optimize import fmin_cg

class LogisticRegression:
    def __init__(self, polynomial=False, lambda_=0, alpha=0.01):
        self.ploynomial = polynomial
        self.lambda_ = lambda_
        self.alpha = alpha

    def _cost_function(self, theta, X, y, lambda_):
        m = X.shape[1]
        J = 0
        h = sigmoid(X@theta).reshape(X.shape[0], 1)
        J = 1/m * -((y.T @ np.log(h)) + ((1-y.T) @ np.log(1-h)))
        grad = sum(((h - y) * X)).T

        # J = self._cost_fmin(theta, X, y)
        # grad = self._grad_fmin(theta, X, y)

        return J, grad

    def _cost_fmin(self, theta, X, y):
        m = X.shape[1]
        J = 0
        h = sigmoid(X@theta).reshape(X.shape[0], 1)
        J = 1/m * -((y.T @ np.log(h)) + ((1-y.T) @ np.log(1-h)))

        return J

    def _grad_fmin(self, theta, X, y):
        h = sigmoid(X@theta).reshape(X.shape[0], 1)
        grad = sum(((h - y) * X)).T

        return grad


    def fit(self, X, y):
        X = np.array(np.c_[np.ones((X.shape[0], 1)), X])
        theta = np.zeros((X.shape[1]))

        self.optimal_theta = gradient_desc(self._cost_function, X, y, theta, 0, .003, 50000, True)

        # self.optimal_theta = fmin_cg(self._cost_fmin, theta, fprime=self._grad_fmin, args=(X, y), disp=0)

        return self.optimal_theta

    def predict(self, X):
        prdicted_y = sigmoid(X@optimal_theta)
        predicted_y = np.array([1 if x>=0.5 else 0 for x in (predicted_y.reshape(predicted_y.shape))])

        return predicted_y
