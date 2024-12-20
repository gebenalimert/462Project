import os
import itertools
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

# Initialize Spotify API client
CLIENT_ID = "9d298c3e447746c2b2f7a11ce6e0dbef"
CLIENT_SECRET = "146718025284413eb81105596cb933eb"
CLIENT_CREDENTIALS_MANAGER = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=CLIENT_CREDENTIALS_MANAGER)

# Available features
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "valence", "tempo", "liveness", "key", "mode"]

def initialize_csv(file_path="song_data.csv"):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genre"] + FEATURES)

def search_genre(genre):
    track_ids = []
    genre_data = []
    
    for offset in range(0, 100, 20):
        results = sp.search(q=f'genre:{genre}', type='track', limit=20, offset=offset, market="US")
        track_ids.extend([t['id'] for t in results['tracks']['items']])
    
    if track_ids:
        features = sp.audio_features(track_ids)
        for feature in features:
            if feature:
                genre_data.append([
                    genre,
                    feature["danceability"],
                    feature["energy"],
                    feature["loudness"],
                    feature["speechiness"],
                    feature["acousticness"],
                    feature["instrumentalness"],
                    feature["valence"],
                    feature["tempo"],
                    feature["liveness"],
                    feature["key"],
                    feature["mode"]
                ])
    return genre_data

def save_to_csv(data, file_path="song_data.csv"):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

# Compute similarity function
def compute_similarity(vector1, vector2, metric="euclidean"):
    if metric == "cosine":
        return cosine_similarity([vector1], [vector2])[0][0]
    elif metric == "euclidean":
        return distance.euclidean(vector1, vector2)
    else:
        raise ValueError("Unsupported metric")

def main():
    initialize_csv()
    
    genres = ["pop", "classical", "metal", "rock", "jazz", "hip-hop"]
    genre_data = {}
    for genre in genres:
        data = search_genre(genre)
        genre_data[genre] = data
        save_to_csv(data)

    best_features = None
    lowest_inter_similarity = float("inf")  # We want this as low as possible
    highest_intra_similarity = -1  # We want this as high as possible

    # Try all non-empty combinations of features
    for r in range(1, len(FEATURES) + 1):
        for feature_combination in itertools.combinations(FEATURES, r):
            print(f"Testing feature combination: {feature_combination}")
            
            # Calculate inter-class similarity
            inter_similarities = []
            for genre1, genre2 in itertools.combinations(genres, 2):
                for x in genre_data[genre1]:
                    for y in genre_data[genre2]:
                        vec1 = [x[FEATURES.index(f) + 1] for f in feature_combination]  # +1 to skip genre column
                        vec2 = [y[FEATURES.index(f) + 1] for f in feature_combination]
                        similarity = compute_similarity(vec1, vec2, metric="euclidean")
                        inter_similarities.append(similarity)
            avg_inter_similarity = np.mean(inter_similarities)
            print(f"Avg inter-class similarity for {feature_combination}: {avg_inter_similarity}")

            # Calculate intra-class similarity
            intra_similarities = []
            for genre in genres:
                for i in range(len(genre_data[genre])):
                    for j in range(i + 1, len(genre_data[genre])):
                        vec1 = [genre_data[genre][i][FEATURES.index(f) + 1] for f in feature_combination]
                        vec2 = [genre_data[genre][j][FEATURES.index(f) + 1] for f in feature_combination]
                        similarity = compute_similarity(vec1, vec2, metric="euclidean")
                        intra_similarities.append(similarity)
            avg_intra_similarity = np.mean(intra_similarities)
            print(f"Avg intra-class similarity for {feature_combination}: {avg_intra_similarity}")

            # Determine if this is the best combination based on low inter-class similarity and high intra-class similarity
            if avg_inter_similarity < lowest_inter_similarity and avg_intra_similarity > highest_intra_similarity:
                lowest_inter_similarity = avg_inter_similarity
                highest_intra_similarity = avg_intra_similarity
                best_features = feature_combination

    print("\nBest feature combination:")
    print(best_features)
    print("Lowest average inter-class similarity:", lowest_inter_similarity)
    print("Highest average intra-class similarity:", highest_intra_similarity)

if __name__ == "__main__":
    main()
