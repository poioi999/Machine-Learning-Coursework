import numpy as np

training_data = np.loadtxt("clustering_train.txt")

# ignore data and put dummy value for cluster centres
kmeans_cluster_centres = np.zeros((3,3))

print('k-means cluster centres:')
print(kmeans_cluster_centres)

# ignore data and put dummy value for Gaussian means
gmm_means = np.zeros((3,3))
print('Gaussian means:')
print(gmm_means)

