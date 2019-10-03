import numpy as np
from math import log
from utils.optimize import gradient_desc
from utils.utility import sigmoid
from scipy.optimize import fmin_cg

class LogisticRegression:
    def __init__(self, polynomial=False, lambda_=0,
                 alpha=0.01, multi_class=False):
        self.ploynomial = polynomial
        self.lambda_ = lambda_
        self.alpha = alpha
        self.multi_class = multi_class

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


    def _fit(self, X, y):
        theta = np.zeros((X.shape[1]))

        optimal_theta = gradient_desc(self._cost_function, X, y,
                                      theta, 0, .003, 500, False)

        return optimal_theta


    def fit(self, X, y):
        class_iter = len(set(y.ravel()))
        classes = list(set(y.ravel()))

        X = np.c_[np.ones((X.shape[0], 1)), X]

        if not self.multi_class:
            class_iter -= 1

        thetas = []
        for i in range(class_iter):
            y_ = np.array([1 if x == classes[i]
                             else 0 for x in y]).reshape(X.shape[0], 1)
            thetas.append(list(self._fit(X, y_)))

        self.thetas = np.array(thetas).reshape(X.shape[1], class_iter)
        # self.optimal_theta = fmin_cg(self._cost_fmin, theta, fprime=self._grad_fmin, args=(X, y), disp=0)

        return self.thetas

    def predict(self, X):
        prdicted_y = sigmoid(X@self.thetas)
        predicted = []
        for i in range(predicted_y.shape[0]):
            predicted.append(np.argmax(predicted_y[i, :]))

        return predicted
