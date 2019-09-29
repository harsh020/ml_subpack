import numpy as np
from utils.optimize import gradient_desc

class SVC:
    def __init__(self, C, kernel='linear'):
        self.C = C
        self.kernel = kernel

        return

    def _cost(self, h_x, x):
        cost = 0
        if x == 0:
            if h_x > -1:
                cost = h_x
        else:
            if h_x < 1:
                cost = h_x

        return cost

    ####
    def _grad(self, h_x, x):
        cost = 0
        if x == 0:
            if h_x > -1:
                cost = h_x
        else:
            if h_x < 1:
                cost = h_x

        return cost


    def _cost_function(self, theta, X, y, V):
        J = 0
        theta = theta.reshape((X.shape[1], 1))
        h_x = X @ theta
        for i in range(X.shape[1]):
            J += y[i]*self._cost(h_x[i], 1) + (1-y[i])*self._cost(h_x[i], 0)
        J *= self.C
        J += .5 * sum(theta[1:])

        h_x_ = y*X + (1-y)*X
        h_1 = h_x_.copy()
        h_2 = h_x_.copy()
        h_1[h_1 > 1] = 0
        h_2[h_2 < -1] = 0
        h_x_ = (h_1 + h_2) / 2
        grad = np.zeros(theta.shape)
        grad = sum(h_x_).T.reshape(theta.shape)

        return J, grad

    def _cost_fmin(self, theta, X, y):
        J, grad = self._cost_function(theta, X, y)

        return J

    def _grad_fmin(self, theta, X, y):
        J, grad = self._cost_function(theta, X, y)

        return grad


    def fit(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        y = y.reshape(X.shape[0], 1)
        initial_theta = np.zeros((X.shape[1], 1))

        self.optimal_theta = gradient_desc(self._cost_function, X, y, initial_theta, self.C, 0.03, 1000, True)
