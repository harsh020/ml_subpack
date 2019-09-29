from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression
from k_means import KMeans

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()


X_ = load_iris()
X = X_.data[:, [0, 1]]
y = X_.target

thetas = []

clf = KMeans(3)

# diff_y = list(set(y))
# for c in [0, 2]:
#     y_ = np.array([1 if i == diff_y[c] else 0 for i in y]).reshape(150, 1)
#     res = clf.fit(X, y_)
#     thetas.append(res)

res = clf.fit(X)
thetas.append(res)

# plt.scatter(X[:50, 0], X[:50, 1], color='r', marker='^')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='r', marker='o')
# # plt.scatter(X[100:, 0], X[100:, 1], color='g', marker='+')
#
# x_val = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.2)
#
# for c in thetas:
#     # y_val = - (c[0] + c[1]*x_val) / c[2]
#     #
#     # plt.plot(x_val, y_val)
#     plt.scatter(c[:, 0], c[:, 1], color='k')
#
# plt.show()
