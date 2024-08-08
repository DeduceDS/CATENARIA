import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

#### FUNCTIONS FOR CLUSTERING ####
def initialize_centroids(points, n_clusters, coord):
    """
    Initialize centroids for clustering based on the minimum, maximum, and optionally mean of the x-coordinates.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    n_clusters (int): The number of clusters.

    Returns:
    numpy.ndarray: The initialized centroids.
    """
    logger.trace(f"Initializing for {n_clusters} clusters")
    centroids = np.zeros(n_clusters)
    
    centroids[0] = np.min(points[coord, :])
    centroids[1] = np.max(points[coord, :])
    
    if n_clusters > 2:
        step = (centroids[1] - centroids[0]) / (n_clusters - 1)
        for i in range(2, n_clusters):
            centroids[i] = centroids[0] + step * (i - 1)
    
    return centroids

def assign_clusters(points, centroids, coord):
    """
    Assign each point to the nearest centroid based on the specified coordinate.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    centroids (numpy.ndarray): The centroids.
    coord (int): The coordinate axis to use for distance calculation.

    Returns:
    numpy.ndarray: The index of the nearest centroid for each point.
    """
    distances = np.abs(points[coord, :, None] - centroids)
    return np.argmin(distances, axis=1)

def update_centroids(points, labels, n_clusters, coord):
    """
    Update the centroids based on the mean of the points assigned to each cluster.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    labels (numpy.ndarray): The cluster labels for each point.
    n_clusters (int): The number of clusters.
    coord (int): The coordinate axis to use for updating centroids.

    Returns:
    numpy.ndarray: The updated centroids.
    """
    new_centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_points = points[coord, labels == i]
        if cluster_points.size > 0:
            new_centroids[i] = np.mean(cluster_points)
    return new_centroids

def kmeans_clustering(points, n_clusters, max_iterations, coord):
    """
    Perform KMeans clustering on the specified coordinate of the points.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    n_clusters (int): The number of clusters.
    max_iterations (int): The maximum number of iterations for the KMeans algorithm.
    coord (int): The coordinate axis to use for clustering.

    Returns:
    tuple: Cluster labels and centroids.
    """
    logger.debug(f"Starting KMeans clustering for coord: {coord}")
    centroids = initialize_centroids(points, n_clusters, coord)
    for iteration in range(max_iterations):
        labels = assign_clusters(points, centroids, coord)
        new_centroids = update_centroids(points, labels, n_clusters, coord)
        if np.allclose(new_centroids, centroids):
            logger.trace(f"Convergence reached at iteration {iteration}")
            break
        centroids = new_centroids
    return labels, centroids

def plot_clusters(points, labels, centroids, coord):
    """
    Plot the clusters and centroids.

    Parameters:
    points (numpy.ndarray): The x, y, and z coordinates of the points.
    labels (numpy.ndarray): The cluster labels for each point.
    centroids (numpy.ndarray): The centroids.
    coord (int): The coordinate axis used for clustering.
    """
    if coord == 0:
        coord1, coord2 = 1, 2
    elif coord == 1:
        coord1, coord2 = 0, 2
    else:
        coord1, coord2 = 0, 1

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.scatter(points[coord, :], points[coord1, :], c=labels, cmap='viridis', s=1)
    plt.vlines(centroids, ymin=np.min(points[coord1, :]), ymax=np.max(points[coord1, :]), color='red', label='Centroids')
    plt.subplot(1,2,2)
    plt.scatter(points[coord, :], points[coord2, :], c=labels, cmap='viridis', s=1)
    plt.vlines(centroids, ymin=np.min(points[coord2, :]), ymax=np.max(points[coord2, :]), color='red', label='Centroids')
    plt.title(f'Clusters along {["X", "Y", "Z"][coord]}-axis vs {["X", "Y", "Z"][coord1]}-axis')
    plt.xlabel(f'{["X", "Y", "Z"][coord]} Coordinate')
    plt.ylabel(f'{["X", "Y", "Z"][coord1]} Coordinate')
    plt.legend()
    plt.show()
    
def group_dbscan_3(k,X_scaled):
    """
    Groups data points into clusters using the DBSCAN algorithm, with the optimal epsilon value determined from k-nearest neighbors distances.

    Parameters:
    k (int): The number of nearest neighbors to consider for determining the epsilon value.
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.
        - labels (numpy.ndarray): An array of labels assigned to each data point, indicating the cluster it belongs to. 
        Noise points are labeled as -1.

    The function performs the following steps:
    1. Fits a k-nearest neighbors model to the data to determine distances to the k-th nearest neighbors.
    2. Sorts these distances and computes the second derivative to find the inflection point, which indicates the optimal epsilon value.
    3. Applies the DBSCAN algorithm with the determined epsilon value and the specified min_samples.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.

    Note:
    - The function determines the optimal epsilon value dynamically based on the distances to the k-th nearest neighbors.
    - The DBSCAN algorithm groups data points into clusters based on density, identifying regions of high density and labeling points in low-density regions as noise.
    """

    neighbors = NearestNeighbors(n_neighbors=int(k))
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, int(k)-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=int(k), algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = X_scaled[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]

    # centroids=flatten_sublist(centroids)
    return centroids, labels


def dbscan_find_clusters_3(X_scaled):
    """
    Finds clusters in the scaled data using the DBSCAN algorithm with different epsilon values,
    and selects the best clustering based on the silhouette score.

    Parameters:
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (numpy.ndarray): The centroids of the clusters found by DBSCAN.
        - labels (numpy.ndarray): The labels assigned to each data point indicating the cluster it belongs to.

    The function performs the following steps:
    1. Iterates over a predefined list of epsilon values.
    2. Applies the `group_dbscan_3` function to cluster the data with each epsilon value.
    3. Computes the silhouette score for each clustering result.
    4. Selects the clustering result with the highest silhouette score.
    5. Returns the centroids and labels of the best clustering result.

    The silhouette score measures how similar an object is to its own cluster compared to other clusters, providing an indication of the quality of the clustering.
    """

    ar=[5,10,20]
    for k in ar:
        
        centroids,labels=group_dbscan_3(k,X_scaled.T)
        if len(np.unique(labels))==1:
            score=-1
        else:
            score=metrics.silhouette_score(X_scaled.T,labels)

        if k==np.array(ar).min():
            best_score=score
            best_k=k
        elif score>best_score:
            best_score=score
            best_k=k

    centroids,labels=group_dbscan_3(best_k,X_scaled.T)

    return centroids,labels

############################ NOT IN RELEASE #########################################

def spectral_clustering(points, n_clusters, n_init):
    """
    Performs spectral clustering on the given data points with scikit-learn. 
    The affinity matrix is constructed using the k-nearest neighbors graph.

    Parameters:
    points (numpy.ndarray): A 2D array where each column represents a data point.
    n_clusters (int): The number of clusters to form.
    n_init (int): The number of times the algorithm will run with different initializations.

    Returns:
    tuple: A tuple containing:
        - labels (numpy.ndarray): An array of cluster labels assigned to each data point.
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.

    The function performs the following steps:
    1. Determines the number of neighbors for the k-nearest neighbors graph.
    2. Constructs the affinity matrix using the k-nearest neighbors graph.
    3. Applies the Spectral Clustering algorithm with the specified number of clusters and initializations.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.
    5. Returns the cluster labels and centroids.
    """

    n_samples = points.shape[1]
    n_neighbors = max(2, n_samples // 2)
    n_neighbors = min(n_neighbors, n_samples)
    affinity_matrix = kneighbors_graph(points.T, n_neighbors=n_neighbors, include_self=True)

    # model = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=n_init, random_state=0)
    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)
    labels = model.fit_predict(points.T)
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = points.T[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]
    return labels, centroids

def invert_linear_model(y_val, slope, intercept):
    """
    Inverts a linear model to compute the x-values from given y-values.

    Parameters:
    y_val (float or numpy.ndarray), slope (float), intercept (float)
    
    Returns:
    float or numpy.ndarray: The computed x-values corresponding to the given y-values.
    """
    
    return (y_val - intercept) / slope


def group_dbscan(k,X_scaled):
    """
    Perform DBSCAN clustering on scaled data, determining the optimal epsilon using the k-nearest neighbors method.

    This function uses the DBSCAN algorithm to cluster the given scaled data. It first fits the k-nearest neighbors 
    to determine the distances and uses the second derivative of the sorted distances to find the inflection point, 
    which is used as the epsilon value for DBSCAN. The function then performs DBSCAN clustering and returns the cluster labels.

    Parameters:
    k (int): The number of neighbors to use for determining the optimal epsilon.
    X_scaled (numpy.ndarray): The scaled x, y, and z coordinates of the data points.

    Returns:
    numpy.ndarray: The cluster labels assigned to each data point.
    """

    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, k-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=k, algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    return labels

def group_dbscan_4(k,X_scaled):
    """
    Groups data points into clusters using the DBSCAN algorithm, with the optimal epsilon value determined from k-nearest neighbors distances.

    Parameters:
    k (int): The number of nearest neighbors to consider for determining the epsilon value.
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (list): A list of centroids for each cluster, where each centroid is a list of coordinates.
        - labels (numpy.ndarray): An array of labels assigned to each data point, indicating the cluster it belongs to. 
        Noise points are labeled as -1.

    The function performs the following steps:
    1. Fits a k-nearest neighbors model to the data to determine distances to the k-th nearest neighbors.
    2. Sorts these distances and computes the second derivative to find the inflection point, which indicates the optimal epsilon value.
    3. Applies the DBSCAN algorithm with the determined epsilon value and the specified min_samples.
    4. Computes the centroids for each cluster by averaging the coordinates of points within the cluster.

    Note:
    - The function determines the optimal epsilon value dynamically based on the distances to the k-th nearest neighbors.
    - The DBSCAN algorithm groups data points into clusters based on density, identifying regions of high density and labeling points in low-density regions as noise.
    """

    neighbors = NearestNeighbors(n_neighbors=int(k))
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)

    distances = np.sort(distances[:, int(k)-1], axis=0)
    second_derivative = np.diff(distances, n=5)
    inflection_point = np.argmax(second_derivative) + 1

    dbscan = DBSCAN(eps=distances[inflection_point], min_samples=int(k), algorithm = "auto")  # Ajusta eps y min_samples según tus datos
    labels = dbscan.fit_predict(X_scaled)

    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:
            cluster_points = X_scaled[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
    centroids=[point.tolist() for point in centroids]
    return centroids, labels

def dbscan_find_clusters_4(X_scaled):
    """
    Finds clusters in the scaled data using the DBSCAN algorithm with different epsilon values,
    and selects the best clustering based on the silhouette score.

    Parameters:
    X_scaled (numpy.ndarray): A 2D array of scaled data points to be clustered.

    Returns:
    tuple: A tuple containing:
        - centroids (numpy.ndarray): The centroids of the clusters found by DBSCAN.
        - labels (numpy.ndarray): The labels assigned to each data point indicating the cluster it belongs to.

    The function performs the following steps:
    1. Iterates over a predefined list of epsilon values.
    2. Applies the `group_dbscan_3` function to cluster the data with each epsilon value.
    3. Computes the silhouette score for each clustering result.
    4. Selects the clustering result with the highest silhouette score.
    5. Returns the centroids and labels of the best clustering result.

    The silhouette score measures how similar an object is to its own cluster compared to other clusters, providing an indication of the quality of the clustering.
    """

    ar=[3,5,7,10,15,20,35,50]
    matching_clust=[]
    for k in ar:
        
        centroids,labels=group_dbscan_4(k,X_scaled.T)
        if len(np.unique(labels))!=3:
            score=-1

        elif len(np.unique(labels))==3:
            score=metrics.silhouette_score(X_scaled.T,labels)

        if k==np.array(ar).min():
            best_score=score
            best_k=k

        elif score>best_score:
            best_score=score
            best_k=k

    centroids,labels=group_dbscan_3(best_k,X_scaled.T)

    return centroids,labels

