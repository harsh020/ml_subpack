import numpy as np
from scipy.optimize import fmin_cg
from ..utils.optimize import gradient_desc, computeNumericalGradient
from ..utils.utility import sigmoid, sigmoid_grad, ravel, unravel

import sys
np.set_printoptions(threshold=sys.maxsize)

np.seterr(divide = 'ignore')

class NeuralNetwork:
    def __init__(self, hidden_layers, units):
        self.hidden_layers = hidden_layers
        self.units = units

    def _forward_prop(self, X, y, theta, i):
        m, n = X.shape
        a_ = []
        z_ = []
        J = 0
        a = np.array(X[i, :].T).reshape(n, 1)
        a_.append(a)
        z_.append(np.concatenate((np.ones((1, 1)), a), axis=0))
        for j in range(self.hidden_layers+1):
            z = np.array(theta[j] @ a)
            a = sigmoid(z)
            a = np.concatenate((np.ones((1, 1)), a), axis=0)
            a_.append(a)
            z_.append(z)

        a_L = a[1:]
        J = ((((y[i, :])@np.log(a_L)) + ((1-y[i, :])@np.log(1-a_L)))/m)

        return J, np.array(z_), np.array(a_)

    def _back_prop(self, X, y, theta, z, a):
        y = np.array(y).reshape(10, 1)
        a_L = a[-1][1:, :]

        theta_grad = []
        delta = a_L - y
        theta_grad.append(np.array(delta * a[len(theta)-1].T))

        for j in range(len(theta)-1, 0, -1):
            delta = (theta[j][:, 1:].T @ delta) * sigmoid_grad(z[j])
            theta_grad.append(np.array(delta * a[j-1].T))

        return theta_grad[::-1]

    def _cost_function(self, theta, X, y, lambda_):
        m, n = X.shape
        J = 0

        theta = np.array(unravel(theta, self.units, self.hidden_layers, n-1))

        theta_grad = np.array([np.zeros(x.shape) for x in theta])

        for i in range(m):
            j, z, a = self._forward_prop(X, y, theta, i)
            J -= j

            theta_grad += np.array(self._back_prop(X, y[i, :].T, theta, z, a))

        reg_sum = 0
        for i in range(self.hidden_layers+1):
            reg_sum += sum(sum(np.array(theta[i])**2)[1:])
        J += (lambda_/(2*m))*reg_sum

        reg_sum = 0
        for i in range(self.hidden_layers+1):
            reg_sum += sum(sum(np.array(theta[i]))[1:])

        for i in range(len(theta)):
            theta_grad[i] = np.c_[(1/m)*theta_grad[i][:, 0], ((1/m)*theta_grad[i][:, 1:]+(lambda_/m)*theta[i][:, 1:])]

        theta_grad = ravel(theta_grad)

        return J, theta_grad

    def fit(self, X, y, lambda_=1, iter=500, alpha=1):
        """ Fit the Nural Network according to the given training data.

        Parameters
        ----------
        X : Numpy array, shape: (n_samples, n_features).
        Training Data. Consists of feature vectors with n_features.

        y : Numpy array (vector), shape: (n_samples, 1).
        Target Values or Labels. It is a vector with n_samples elements.

        lambda_ : Scalar, Real-Number.
        Regularization Parameter.

        alpha : Scalar, Real-Number.
        It defines the step-size to be taken during gradient descent.

        iter : Scalar, Positive Integer.
        It defines the number of times to run gradient descent.

        Return
        ------
        self : object

        """
        m, n = X.shape
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        theta = np.random.randn((n+1)*self.units + self.units*(self.units+1)*(self.hidden_layers-1) + 10*(self.units+1), 1) * 2

        theta = np.sin(theta)

        y_ = np.zeros((m, max(y)[0]+1))
        for i in range(m):
            y_[i, y[i][0]] = 1
        y = y_

        self.optimal_theta = gradient_desc(self._cost_function, X, y, theta, lambda_, iter, alpha, disp_curve)

        return self


    def predict(self, X):
        m, n = X.shape
        X = np.concatenate((np.ones((m, 1)), X), axis=1)
        m, n = X.shape
        y_predict = np.zeros((X.shape[0], 1))
        theta = unravel(self.optimal_theta, self.units, self.hidden_layers, n-1)

        for i in range(m):
            a = np.array(X[i, :].T).reshape(n, 1)
            for j in range(self.hidden_layers+1):
                z = np.array(theta[j] @ a)
                a = sigmoid(z)
                a = np.concatenate((np.ones((1, 1)), a), axis=0)
            a_L = a[1:]
            y_predict[i] = list(a_L).index(max(a_L))

        return y_predict
