import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

training_data = np.loadtxt("clustering_train.txt")

print(training_data.shape)

K = 3 

kmeans = KMeans(K, init='random').fit(training_data)
cluster_assignments = kmeans.predict(training_data)
centroid_locations = kmeans.cluster_centers_

def plot_k_means(X, cluster_assignments, centroid_locations):
    plt.figure(figsize=(6, 6))
    plt.viridis() # Set colour map
    plt.scatter(X[:, 0], X[:, 1], s=20, c=cluster_assignments, alpha=0.8) # plot data points
    plt.scatter(centroid_locations[:, 0], centroid_locations[:, 1], s=200, marker='X', c=range(K), edgecolors='k') # plot centroids
    plt.show()

plot_k_means(training_data, cluster_assignments, centroid_locations)