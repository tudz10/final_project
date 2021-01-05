import numpy as np
import scipy

from sklearn.cluster import KMeans
import helpers
import cv2

          
class SpectralClustering:
    def __init__(self, n_clusters, affinity_type, gamma = 1.0):
        self.n_clusters = n_clusters
        self.affinity_type = affinity_type
        self.gamma = gamma

    def train(self, x_train):
        self.affinity_matrix_ = self._get_affinity_matrix(x_train)
        embedding_features = self._get_embedding()
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(embedding_features)
        self.labels_ = kmeans.labels_

    def _get_affinity_matrix(self, x_train):
        if self.affinity_type == 0:
          return helpers.chi_square_affinity(x_train)
        elif self.affinity_type == 1:
          return helpers.chi_square_affinity(x_train, False)

    def _get_embedding(self, norm_laplacian=True):
        n = self.affinity_matrix_.shape[0]
        # compute the unnormalized Laplacian
        D = np.sum(self.affinity_matrix_, axis=0)
        L =  np.diag(D) - self.affinity_matrix_
        if norm_laplacian:
            m = np.array(self.affinity_matrix_)
            np.fill_diagonal(m, 0)
            D = np.sum(m, axis=0)
            L = np.eye(D.shape[0]) - np.diag(1.0 / D) @ m
        values, vectors = np.linalg.eig(L)
        Ls = [[i, np.real(values[i])] for i in range(n)]
        Ls.sort(key=lambda x:x[1])
        k = self.n_clusters
        selected_array = [Ls[i][0] for i in range(k)]
        return np.real(vectors[:, selected_array])

    def fit(self, x_train):
        # alias for train
        self.train(x_train)
