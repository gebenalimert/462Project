import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Assume you have a DataFrame `df` with `genre` and feature columns
# Example features: danceability, energy, instrumentalness, etc.
FEATURES = ["danceability", "energy",  "acousticness", "loudness"]

# Function to apply k-means clustering and evaluate results
def test_clustering(df, features, n_clusters):
    X = df[features].values
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # Calculate silhouette score to evaluate clustering quality
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"Silhouette score for features {features}: {silhouette_avg}")
    
    # Visualize clusters using PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette="viridis", s=60)
    plt.title(f"K-means Clustering with features {features}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

    # Return clustering labels and silhouette score for further analysis
    return cluster_labels, silhouette_avg

# Test clustering with different feature combinations
n_clusters = len(df['genre'].unique())  # Number of genres as number of clusters
results = {}

for r in range(1, len(FEATURES) + 1):
    for feature_combination in itertools.combinations(FEATURES, r):
        # Run clustering and save results
        labels, silhouette_avg = test_clustering(df, list(feature_combination), n_clusters)
        results[feature_combination] = silhouette_avg

# Print the best feature combination for clustering
best_features = max(results, key=results.get)
print(f"Best feature combination for clustering: {best_features}")
print(f"Highest silhouette score: {results[best_features]}")
