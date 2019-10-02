import random
import numpy as np
import matplotlib.pyplot as plt

from utils.utility import euclidean_dist

class KMeans:
    def __init__(self, n_clusters=1, centroid_plot=True):
        self.centroid_plot = True
        # self.elbow_plot = True
        self.n_clusters = n_clusters

        return

    def _cost_function(self, X, c, mu):
        J = 0
        for i in range(X.shape[0]):
            J += np.square(
                        euclidean_dist(X[i, :],
                        mu[int(c[i]), :].reshape(1, X.shape[1])))
        J /= X.shape[0]

        return J


    def _assign_cluster(self, X, c, mu):
        for i in range(len(X)):
            dist = euclidean_dist(X[i], mu)
            idx = np.argmin(dist)
            c[i] = idx

        return c


    def _cluster_centroid(self, X, c, mu):
        mu = np.zeros((self.n_clusters, X.shape[1]))
        for c_ in set(c.ravel()):
            counter = 0
            for i in range(X.shape[0]):
                if c[i] == c_:
                    mu[int(c[i]), :] += X[i, :]
                    counter += 1
            mu[int(c_)] /= counter
        return mu


    def _plot_scatter(self, X, cluster_arr, optimal):
        X_1 = X[:, 0]
        X_2 = X[:, 1]
        c = cluster_arr.ravel()

        plt.scatter(X_1[c==0], X_2[c==0])
        plt.scatter(X_1[c==1], X_2[c==1])
        plt.scatter(X_1[c==2], X_2[c==2])
        plt.scatter(optimal[:, 0], optimal[:, 1], color='k')
        plt.show()


    def fit(self, X):
        m, n = X.shape
        prev_J = float('inf')
        optimal = None
        cluster_arr = np.zeros((m , 1))
        random.seed(1)
        for i in range(10):
            cluster_mean = []
            for j in range(self.n_clusters):
                mean = random.choice(X)
                cluster_mean.append(mean)

            cluster_mean = np.array(cluster_mean)

            for iter in range(100):
                cluster_arr = self._assign_cluster(X, cluster_arr, cluster_mean)
                cluster_mean = self._cluster_centroid(X, cluster_arr, cluster_mean)
                J = self._cost_function(X, cluster_arr, cluster_mean)
                print("Job : {} | Iteration : {} | "
                      "Cost : {}" .format(i+1, iter+1,
                                                round(float(J), 4)), end='\r')
            J = self._cost_function(X, cluster_arr, cluster_mean)
            if J  <= prev_J:
                optimal = cluster_mean
                prev_J = J

        if self.centroid_plot:
            self._plot_scatter(X, cluster_arr, optimal)

        self._optimal = optimal

        return self


    def predict(self, X):
        c = np.zeros((X.shape[0], 1))
        for i in range(len(X)):
            dist = euclidean_dist(X[i], self._optimal)
            idx = np.argmin(dist)
            c[i] = idx

        return c
