import numpy as np
import random
import matplotlib.pyplot as plt
from utils.utility import euclidean_dist

class KMeans:
    def __init__(self, n_clusters=1):
        self.n_clusters = n_clusters

        return

    def _cost_function(self, X, c, mu):
        J = 0
        for i in range(X.shape[0]):
            J += np.square(euclidean_dist(X[i, :], mu[int(c[i]), :]))
        J /= X.shape[0]

        return J

    def _assign_cluster(self, X, c, mu):
        # nuw_mu = np.zeros(mu.shape)
        # for i in range(X.shape[0]):
        #     new_mu[i] = min(euclidean_dist(X[i], mu))
        for i in range(len(X)):
            dist = list(euclidean_dist(X[i], mu))
            idx = dist.index(min(dist))
            c[i] = idx

        return c

    def _cluster_centroid(self, X, c, mu):
        for c_ in set(c.ravel()):
            counter = 0
            for i in range(X.shape[0]):
                if c[i] == c_:
                    mu[int(c_), :] += X[i, :]
                    counter += 1
            mu[int(c_)] /= counter

        return mu

    def _plot_scatter(self, X, cluster_arr, optimal):
        X_1 = X[:, 0]
        X_2 = X[:, 1]
        c = cluster_arr.ravel()

        print(c)

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
                                                round(J, 4)), end='\r')

                if J  <= prev_J:
                    optimal = cluster_mean

        self._plot_scatter(X, cluster_arr, optimal)

        return optimal
