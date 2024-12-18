import os
import math
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv

# Setup environment variables for security
CLIENT_ID = "9d298c3e447746c2b2f7a11ce6e0dbef"
CLIENT_SECRET = "146718025284413eb81105596cb933eb"

# Initialize Spotify API client
CLIENT_CREDENTIALS_MANAGER = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=CLIENT_CREDENTIALS_MANAGER)

# Function to write CSV headers
def initialize_csv(file_path="song_data.csv"):
    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genre", "dance", "energy", "loudness", "speechiness", "acousticness",
                         "instrumentalness", "valence", "tempo", "liveness", "key", "mode"])

# Function to fetch audio features of a genre
def search_genre(genre):
    track_ids = []
    genre_data = []
    
    # Fetch tracks of the specified genre
    for offset in range(0, 100, 20):  # Adjust limit as needed
        results = sp.search(q=f'genre:{genre}', type='track', limit=20, offset=offset, market="US")
        track_ids.extend([t['id'] for t in results['tracks']['items']])
    
    # Get audio features for the tracks
    if track_ids:
        features = sp.audio_features(track_ids)
        for feature in features:
            if feature:  # Check if feature data is not None
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

# Function to save data to CSV
def save_to_csv(data, file_path="song_data.csv"):
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

# Distance calculation function
def calculate_distances(data1, data2):
    distance = 0
    for item1 in data1:
        for item2 in data2:
            # Use only selected features (danceability, speechiness, instrumentalness) for distance
            a = item1[1] - item2[1]  # danceability
            b = item1[4] - item2[4]  # speechiness
            c = item1[6] - item2[6]  # instrumentalness
            distance += math.sqrt(a**2 + b**2 + c**2)
    return distance

# Main clustering function
def main():
    initialize_csv()
    
    # Define genres to compare
    genres = ["pop", "classical", "metal", "rock", "jazz", "hip-hop"]
    genre_data = {}

    # Fetch and save data for each genre
    for genre in genres:
        data = search_genre(genre)
        genre_data[genre] = data
        save_to_csv(data)

    # Calculate and display inter-genre and intra-genre distances
    for genre1 in genres:
        intra_distance = calculate_distances(genre_data[genre1], genre_data[genre1])
        print(f"Intra-distance for {genre1}: {intra_distance}")
        
        for genre2 in genres:
            if genre1 != genre2:
                inter_distance = calculate_distances(genre_data[genre1], genre_data[genre2])
                print(f"Inter-distance between {genre1} and {genre2}: {inter_distance}")

if __name__ == "__main__":
    main()
