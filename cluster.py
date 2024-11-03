import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Load and prepare the data
data = pd.read_csv("song_data.csv")
feats = data[['dance', 'energy', 'loudness',"acousticness","instrumentalness","tempo","valence","key"]].copy()

# Define the number of clusters
k = 3

# Scale the features
def scale(column):
    feats[column] = (feats[column] - feats[column].min()) / (feats[column].max() - feats[column].min()) * 10

for column in feats.columns:
    scale(column)

# Initialize centroids
centroids = feats.sample(n=k).values
print("Initial centroids:", centroids)

# Function to assign clusters
def assign_clusters(centroids):
    assignments = []
    for _, row in feats.iterrows():
        distances = [np.sqrt(np.sum(np.square(centroid - row.values))) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        assignments.append(closest_centroid)
    return np.array(assignments)

# Function to recalculate centroids
def recalculate_centroids(assignments, k):
    new_centroids = []
    for i in range(k):
        cluster_points = feats[np.array(assignments) == i]
        if len(cluster_points) > 0:
            new_centroid = cluster_points.mean(axis=0)
            new_centroids.append(new_centroid.values)
        else:
            # If a cluster has no points, reinitialize to a random point
            new_centroids.append(feats.sample(n=1).values[0])
    return np.array(new_centroids)

# Perform the k-means algorithm
num_iterations = 10
for _ in range(num_iterations):
    assignments = assign_clusters(centroids)
    centroids = recalculate_centroids(assignments, k)

# Adding cluster assignments to the dataframe
feats['cluster'] = assignments

# Visualize the clusters in 3D (choose three features for plotting)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feats['dance'], feats['energy'], feats['valence'], c=feats['cluster'], cmap='viridis')
ax.set_xlabel('Dance')
ax.set_ylabel('Energy')
ax.set_zlabel('Valence')
plt.show()
error = 0
for x in range(100):
    if (1-assignments[x]) ** 2 !=0:
        error += 1
for x in range(100,200):
    if (2 - assignments[x]) ** 2 != 0:
        error += 1
for x in range(200,300):
    if (0 - assignments[x]) ** 2 != 0:
        error += 1
print(np.sum(np.abs(error)))
print(assignments)


