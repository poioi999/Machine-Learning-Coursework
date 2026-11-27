import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

training_data = np.loadtxt("clustering_train.txt")

print(training_data.shape)

'''
K = 3 

kmeans = KMeans(K).fit(training_data)

cluster_assignments = kmeans.predict(training_data)
centroid_locations = kmeans.cluster_centers_
print("K means cluster centres", centroid_locations)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(training_data[:, 0], training_data[:, 1], training_data[:, 2], c=cluster_assignments, cmap='viridis', s=20, alpha=0.8)
ax.scatter(centroid_locations[:, 0], centroid_locations[:, 1], centroid_locations[:, 2], c=range(K), cmap='viridis', s=200, marker='X', edgecolors='k')
plt.title(f'K-Means (K={K})')
plt.show()
'''


K = 3
gmm = GaussianMixture(n_components=K, max_iter=100, warm_start=True, init_params='kmeans', tol=1e-8, n_init=10, random_state=42).fit(training_data)

print("Converged:", gmm.converged_)
print("Iterations:", gmm.n_iter_)
print("Means:\n", gmm.means_)

assignments = gmm.predict(training_data)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(training_data[:, 0], training_data[:, 1], training_data[:, 2], c=assignments, cmap='viridis', s=20, alpha=0.8)
ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], gmm.means_[:, 2], c=range(K), cmap='viridis', s=200, marker='X', edgecolors='k')
plt.title(f'Converged GMM (K={K})')
plt.show()
