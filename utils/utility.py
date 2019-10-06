import numpy as np
import random

def sigmoid(z):
    g = 1.0 / (1 + np.exp(-z))
    return g

def sigmoid_grad(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g

def ravel(thetagrad):
    f = list(map(lambda x : list(x.ravel()), thetagrad))
    ravel = []
    for l in f:
        ravel += l
    return np.array(ravel).reshape(len(ravel), 1)

def unravel(theta, units, hidden_layers, n):
    theta_ = []
    theta_.append(np.array(theta[0:(n+1)*units, 0]).reshape(units, n+1))

    j = (n+1)*units
    for i in range(hidden_layers-1):
        theta_.append(np.array(theta[j:j+((units+1)*units), 0]).reshape(units, units+1))
        j += units*(units+1)

    theta_.append(np.array(theta[j:, 0]).reshape(10, units+1))

    return theta_

def mean_normalize(X):
    m, n = X.shape
    means_ = np.array([np.sum(X[:, i]) for i in range(n)]).reshape(1, n) / m
    std_ = np.std(X, axis=0)
    normalized = (X - means_) / std_
    mu = np.array([np.sum(normalized[:, i]) for i in range(n)]).reshape(1, n) / m
    return normalized, mu

def euclidean_dist(obj_1, obj_2):
    return np.sqrt(sum(np.square(obj_1 - obj_2).T))

def gaussian_model(X, mu, sigma_2):
    exp = np.exp(-np.square(X-mu) / (2*sigma_2))

    return np.array(list((1/np.sqrt(2*np.pi*sigma_2)) * exp))

def multinomial_model(X, mu, sigma_2):
    n = len(mu)
    exp = np.exp(-0.5 * (X-mu)@np.linalg.pinv(sigma_2)@(X-mu).reshape(n, 1))
    coef = 1/(((2*np.pi)**(n/2))*(np.linalg.det(sigma_2)**0.5))
    # print(sigma_2)

    # print(exp.shape)
    return coef*np.array(list(exp))

def test_train_split(X, y, test_size=0.4, valid_size=0):
    rand_idx = list(range(X.shape[0]))
    random.shuffle(rand_idx)
    X = X[rand_idx, :]
    y = y[rand_idx]

    train_idx = int((1-test_size)*X.shape[0])
    # valid_idx = valid_size*X.size[0]
    # test_idx = int(test_size*X.shape[0])
    return X[:train_idx, :], y[:train_idx], X[train_idx:, :], y[train_idx:]

def accuracy_score(true_y, pred_y):
    correct = 0
    for i in range(len(true_y)):
        if true_y[i] == pred_y[i]:
            correct += 1

    return correct/len(true_y)
