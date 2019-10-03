from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression
from k_means import KMeans

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

import seaborn as sns
sns.set()


# X_ = load_iris()
# X = X_.data[:, [2, 0]]
# y = X_.target
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
# X = np.dot(X, transformation)

# diff_y = list(set(y))
# for c in [0, 2]:
#     y_ = np.array([1 if i == diff_y[c] else 0 for i in y]).reshape(150, 1)
#     res = clf.fit(X, y_)
#     thetas.append(res)
lr = LogisticRegression(multi_class=True)
thetas = lr.fit(X, y)

for x, c in zip(range(X.shape[0]), ['b', 'r', 'g']):
    idx = np.where(y == x)
    plt.scatter(X[idx, 0], X[idx, 1], c=c)
#
x_val = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.2)
#
for c in thetas:
    y_val = - (c[0] + c[1]*x_val) / c[2]

    plt.plot(x_val, y_val)
    # plt.scatter(c[:, 0], c[:, 1], color='k')
#
plt.show()
