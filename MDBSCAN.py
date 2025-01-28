import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

from data_read.data_arff import create_compound
import visualization.scatter_plot as sp




class MDBSCAN:
    def __init__(self, k=5, t=0.7, eps=1.3, min_pts=3):
        self.k = k
        self.t = t
        self.eps = eps
        self.min_pts = min_pts


    def fit_predict(self, X):
        # figure3_in_paper(X, k)
        rds = self.compute_relative_densities(X)

        mask_low_density = rds < self.t
        data_low_density = X[mask_low_density]

        # figure4_in_paper(X, y, mask_low_density)

        knns = self.knn_of_data(X)

        knns_low_density = knns[mask_low_density]
        F = self.snnc(data_low_density, knns_low_density)
        # print(F)

        f_flattened = self.flatten(F)

        maskF_inverse = np.ones(len(X), dtype=bool)
        maskF_inverse[f_flattened] = False

        data_high_density = X[maskF_inverse]

        db = DBSCAN(eps=self.eps, min_samples=self.min_pts).fit(data_high_density)
        labels = db.labels_

        labels = self.assign_outliers(data_high_density, labels)

        final_labels = -1 * np.ones(len(X), dtype=int)
        final_labels[maskF_inverse] = labels

        max_label = np.amax(np.unique(labels))
        for count, cluster_ids in enumerate(F):
            maskF = np.zeros(len(X), dtype=bool)
            maskF[cluster_ids] = True
            final_labels[maskF] = max_label + (count+1)

        return final_labels


    def snnc(self, data, knns):
        """
        Share Nearest Neighbors-based Clustering method (SNNC).

        Parameters:
            data: list of low-density data points.
            k_nearest_neighbors: function or dictionary that returns the k-nearest neighbors for a given data point.

        Returns:
            F: list of low-density natural clusters.
        """
        # Initialize variables
        num = len(data)  # The number of low-density points
        clusters = {i: {i} for i in range(num)}  # Initial clusters, one for each data point

        # Step 1: Merge based on shared nearest neighbors
        for i in range(num):
            for j in range(i + 1, num):
                if len(np.intersect1d(knns[i], knns[j])) >= 1:  # Intersection of neighbors
                    clusters[i].update(clusters[j])

        # Step 2: Iteratively merge clusters
        for t in range(100):
            merged = False
            for i in range(num):
                for j in range(i + 1, num):
                    if len(clusters[i].intersection(clusters[j])) >= 1:  # Shared members between clusters
                        clusters[i].update(clusters[j])
                        clusters[j] = set()  # Empty cluster j
                        merged = True
            if not merged:
                break  # Exit loop if no clusters were merged

        # Calculate mean cluster size for non-empty clusters
        cluster_sizes = [len(c) for c in clusters.values() if len(c) > 0]
        mean_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

        # Add large clusters to final result F
        F = []
        for c in clusters.values():
            if len(c) >= mean_size:
                F.append(list(c))

        return F

    def knn_of_point(self, point, data):
        distances = np.linalg.norm(data - point, axis=1)

        nearest_indices = np.argsort(distances)[1:self.k + 1]  # exclude self from knn
        distances = np.sort(distances)[1:self.k + 1]
        return nearest_indices, distances

    def knn_of_data(self, data):
        knns = []
        for data_point in data:
            idx, _ = self.knn_of_point(data_point, data)
            knns.append(idx)
        knns = np.array(knns)
        return knns

    def compute_relative_density_of_a_point(self, point, data):
        _, dists = self.knn_of_point(point, data)
        rd = self.k / (np.sum(dists))

        return rd

    def compute_relative_densities(self, data):
        rds = np.array([self.compute_relative_density_of_a_point(data[i], data) for i in range(len(data))])
        return rds

    def figure3_in_paper(self, X):
        rds = self.compute_relative_densities(X)
        col1 = np.arange(0, len(X), 1)
        col2 = np.array(rds)
        test = np.vstack((col1, col2)).T
        print(test.shape)

        numbers = np.arange(1, len(X) + 1, 1)

        plt.figure()

        # Annotate each point with the corresponding number
        for i, num in enumerate(numbers):
            plt.text(X[i, 0], X[i, 1], str(num), fontsize=8, ha='center', va='center', color='k')
        # Set axis limits for better visibility
        plt.xlim(min(X[:, 0]) - 1, max(X[:, 0]) + 1)
        plt.ylim(min(X[:, 1]) - 1, max(X[:, 1]) + 1)

        sp.plot("test", test)

    def figure4_in_paper(self, X, y, mask_low_density):
        sp.plot("data", X, y)
        sp.plot("data", X[mask_low_density], y[mask_low_density])

    def flatten(self, xss):
        return [x for xs in xss for x in xs]

    def assign_outliers(self, data, labels, metric='euclidean'):
        # Identify unique cluster labels (excluding outliers)
        unique_labels = np.unique(labels[labels != -1])

        if len(unique_labels) == 0:
            raise ValueError("No clusters in data.")

        # Calculate centroids for each cluster
        centroids = np.array([
            data[labels == cluster_label].mean(axis=0) for cluster_label in unique_labels
        ])

        # Identify the indices of outliers (labeled as -1)
        outlier_indices = np.where(labels == -1)[0]

        if len(outlier_indices) == 0:
            # No outliers to process
            return labels

        # Get the outliers' data points
        outliers = data[outlier_indices]

        # Compute distances between outliers and centroids
        distances = cdist(outliers, centroids, metric=metric)

        # Find the closest cluster for each outlier
        closest_clusters = np.argmin(distances, axis=1)

        # Assign the closest cluster label to each outlier
        labels[outlier_indices] = unique_labels[closest_clusters]

        return labels


if __name__ == "__main__":
    X, y = create_compound()
    model = MDBSCAN()
    labels = model.fit_predict(X)

    sp.plot("ground truth", X, y)
    sp.plot("mdbscan", X, labels)
    plt.show()
