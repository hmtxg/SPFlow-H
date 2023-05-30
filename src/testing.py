import numpy as np
from sklearn.cluster import KMeans

# Create a dataset with identical instances
X = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])

# Create a KMeans object with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit the model to the data
clusters = kmeans.fit_predict(X)

# Obtain the cluster labels for each data point
labels = kmeans.labels_

# Obtain the cluster centroids
centroids = kmeans.cluster_centers_

print(labels)     # Output: [0 0 0 0]
print(centroids)  # Output: [[1. 2.], [1. 2.]]
print(clusters)


