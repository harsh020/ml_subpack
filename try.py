from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression
from k_means import KMeans
from dimensionality_reduction import PCA

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()


X_ = load_iris()
X = X_.data[:, :]
y = X_.target

thetas = []

pca = PCA(k_dim=3)
pca.fit(X)
Xt = pca.transform(X)

# diff_y = list(set(y))
# for c in [0, 2]:
#     y_ = np.array([1 if i == diff_y[c] else 0 for i in y]).reshape(150, 1)
#     res = clf.fit(X, y_)
#     thetas.append(res)

# # k-Means
# clf = KMeans(3)
# res = clf.fit(X)
# thetas.append(res)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Xt[:50, 0], Xt[:50, 1], Xt[:50, 2], color='r', marker='o')
ax.scatter(Xt[50:100, 0], Xt[50:100, 1], Xt[50:100, 2], color='b', marker='o')
ax.scatter(Xt[100:, 0], Xt[100:, 1], Xt[100:, 2], color='g', marker='o')
#
# x_val = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.2)
#
# for c in thetas:
#     # y_val = - (c[0] + c[1]*x_val) / c[2]
#     #
#     # plt.plot(x_val, y_val)
#     plt.scatter(c[:, 0], c[:, 1], color='k')
#
plt.show()
