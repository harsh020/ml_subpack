import time
import numpy as np
from utils.optimize import gradient_desc
from scipy.optimize import fmin_cg

class SVC:
    def __init__(self, C, kernel='linear'):
        self.C = C
        self.kernel = kernel
        return

    # def _cost(self, )


    def _cost_function(self, theta, X, y, C):
        J = 0
        h_x = (X @ theta).reshape(y.shape)
        for i in range(len(y)):
            J += self.C*max(0, (1-(y[i]*h_x[i]))[0])
            # print(str(h_x[i]) + "  " + str(y[i]) + "  " + str(J))
        # time.sleep(1)
        J += sum(theta[1:]**2) / 2
        # J = (self.C * sum(max(0, 1-y*(X @ theta)))) + sum(theta**2)/2

        grad = np.zeros(theta.shape)
        temp = (sum(-y*X).T).reshape(theta.shape)
        for i in range(theta.shape[0]):
            grad[i] = self.C*max(0, temp[i])
        grad[1:] += sum(theta[1:])

        return J, grad

    def cost_fmin(self, theta, X, y):
        J, grad = self._cost_function(theta, X, y, self.C)

        return J

    def grad_fmin(self, theta, X, y):
        J, grad = self._cost_function(theta, X, y, self.C)

        return grad


    def fit(self, X, y):
        X = np.array(np.c_[np.ones((X.shape[0], 1)), X])
        y = np.array([-1 if y_ == 0 else 1 for y_ in y]).reshape(y.shape[0], 1)

        initial_theta = np.sin(np.random.randn(X.shape[1], 1)).reshape(X.shape[1], 1)
        # initial_theta = np.zeros((X.shaspe[1], 1))

        self._optimal_theta = gradient_desc(
                                    self._cost_function, X, y,
                                    initial_theta, self.C, 0.3,
                                    10000, True)

        # self._optimal_theta = fmin_cg(self.cost_fmin, initial_theta, args=(X, y))

        return self._optimal_theta


    def predict(self, X):
        if not self._optimal_theta:
            raise ValueError("Classifier not fit. \
                                        Fit the classifier first.")

            return

        if self._optimal_theta.shape[0] != X.shape[1]:
            raise ValueError("Classifier is fitted on different data.")

            return

        predict = X @ self._optimal_theta
        predict = np.array([1 if h_x >= 0 else 0 for h_x in predict])

        return predict
