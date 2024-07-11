import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def determine_optimal_clusters(embeddings, max_clusters=10, distance_threshold=0.4):
    n_samples = len(embeddings)
    if n_samples < 3:
        return 1
    scaled_embeddings = StandardScaler().fit_transform(embeddings)
    silhouette_scores = []
    max_score = -1
    optimal_n_clusters = 1
    for k in range(2, min(max_clusters, n_samples) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_embeddings)
        score = silhouette_score(scaled_embeddings, labels)
        silhouette_scores.append(score)
        if score > max_score:
            max_score = score
            optimal_n_clusters = k
        centers = kmeans.cluster_centers_
        min_dist = cdist(centers, centers).min()
        if min_dist < distance_threshold:
            break
    return optimal_n_clusters

def cluster_embeddings(embeddings):
    if len(embeddings) < 2:
        print("Not enough embeddings for clustering. Assigning all to one cluster.")
        return np.zeros(len(embeddings), dtype=int)
    n_clusters = determine_optimal_clusters(embeddings)
    print(f"Optimal number of clusters determined: {n_clusters}")
    if n_clusters == 1:
        return np.zeros(len(embeddings), dtype=int)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings_scaled)
    return clusters