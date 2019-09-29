import matplotlib.pyplot as plt
import numpy as np

def gradient_desc(func, X, y, theta, lambda_, alpha, iter, disp_curve):
    Y = []
    x = []
    for i in range(iter):
        grad = np.zeros(theta.shape)
        J, grad = func(theta, X, y, lambda_)
        theta -= alpha*grad
        print('Iteration: {} | Cost: {}' .format(i+1, round(float(J[0]), 4)), end='\r')
        Y.append(float(J[0]))
        x.append(i)
    print('')

    # if disp_curve:
    #     plt.plot(x, Y)
    #     plt.show()

    return theta

def computeNumericalGradient(J, theta, X, y, lambda_):
    numgrad = np.zeros(theta.shape);
    perturb = np.zeros(theta.shape);
    e = 1e-4
    for i in range(theta.shape[0]):
        print('Iteration: {}' .format(i+1), end='\r')
        perturb[i] = e
        cost_1, grad_1 = J(theta-perturb, X, y, lambda_)
        cost_2, grad_2 = J(theta+perturb, X, y, lambda_)

        numgrad[i] = (cost_2 - cost_1) / (2*e)
        perturb[i] = 0

    return numgrad
