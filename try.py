from sklearn.datasets import load_iris
from logistic_regression import LogisticRegression
from k_means import KMeans
from dimensionality_reduction import PCA
from k_means import KMeans

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()


X_ = load_iris()
X = X_.data[:, [0, 1, 2]]
y = X_.target

thetas = []

pca = PCA(k_dim=2)
pca.fit(X)
Xt = pca.transform(X)

print(pca.components_)

# k = KMeans(n_clusters=3, centroid_plot=False)
# k.fit(Xt)

# diff_y = list(set(y))
# for c in [0, 2]:
#     y_ = np.array([1 if i == diff_y[c] else 0 for i in y]).reshape(150, 1)
#     res = clf.fit(X, y_)
#     thetas.append(res)

# # k-Means
# clf = KMeans(3)
# res = clf.fit(X)
# thetas.append(res)
fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(221, projection='3d')
ax.scatter(X[:50, 0], X[:50, 1], X[:50, 2], color='r', marker='o')
ax.scatter(X[50:100, 0], X[50:100, 1], X[50:100, 2], color='b', marker='o')
ax.scatter(X[100:, 0], X[100:, 1], X[100:, 2], color='g', marker='o')


ax = fig.add_subplot(222)
ax.scatter(Xt[:50, 0], Xt[:50, 1], color='r', marker='o')
ax.scatter(Xt[50:100, 0], Xt[50:100, 1], color='b', marker='o')
ax.scatter(Xt[100:, 0], Xt[100:, 1], color='g', marker='o')

X_ = pca.inverse_transform(Xt)
print(X_.shape)
ax = fig.add_subplot(223, projection='3d')
ax.scatter(X_[:50, 0], X_[:50, 1], X_[:50, 2], color='r', marker='o')
ax.scatter(X_[50:100, 0], X_[50:100, 1], X_[50:100, 2], color='b', marker='o')
ax.scatter(X_[100:, 0], X_[100:, 1], X_[100:, 2], color='g', marker='o')



# ax.plot(pca.components_[0, 0], pca.components_[1, 0], color='k')
# ax.plot(pca.components_[0, 1], pca.components_[1, 1], color='k')
#
# for c in thetas:
#     # y_val = - (c[0] + c[1]*x_val) / c[2]
#     #
#     # plt.plot(x_val, y_val)
#     plt.scatter(c[:, 0], c[:, 1], color='k')
#
plt.show()
