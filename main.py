import math

import spotipy
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import csv

base = "https://api.spotify.com."
CLIENT_SECRET = "146718025284413eb81105596cb933eb"
CLIENT_ID = "9d298c3e447746c2b2f7a11ce6e0dbef"
REDIRECT_URI = "https://localhost"

CLIENT_CREDENTIALS_MANAGER = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
sp= spotipy.Spotify(client_credentials_manager=CLIENT_CREDENTIALS_MANAGER)
file = open("data.txt","w")
artist_name = []
track_name = []
popularity = []
track_id = []
sp.audio_features("11dFghVXANMlKmJXsNCbNl")
with open("song_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genre","dance","energy","loudness","speechiness","acousticness","instrumentalness","valence","tempo","liveness","key","mode"])
def search_genre(genre,data):
    track_id = []
    for i in range(0,100,20):
        query = 'genre:{genre}'.format(genre=genre)
        track_results = sp.search(q=query, market="GB", type='track', limit=20,offset=i)

        for i, t in enumerate(track_results['tracks']['items']):
            artist_name.append(t['artists'][0]['name'])
            track_name.append(t['name'])
            track_id.append(t['id'])
            popularity.append(t['popularity'])
    track = sp.audio_features(track_id)
    for x in range(len(track)):
        #info = sp.track(x)
        #artist = info['artists'][0]["id"]
        #name =info['artists'][0]["name"]
        #k = sp.artist(artist)
        #genre = k['genres']
            dance = track[x]["danceability"]
            energy = track[x]["energy"]
            loudness = track[x]["loudness"]
            speechiness = track[x]["speechiness"]
            acousticness = track[x]["acousticness"]
            instrumentalness = track[x]["instrumentalness"]
            valence = track[x]["valence"]
            tempo = track[x]["tempo"]
            liveness = track[x]["liveness"]
            key = track[x]["key"]
            mode = track[x]["mode"]
            data.append([genre,dance,energy,loudness,speechiness,acousticness,instrumentalness,valence,tempo,liveness,key,mode])
            with open("song_data.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([genre,dance,energy,loudness,speechiness,acousticness,instrumentalness,valence,tempo,liveness,key,mode])
pop = []
classical = []
rock = []
interdistance = 0
intradistance = 0
search_genre("pop",pop)
search_genre("classical",classical)
search_genre("metal",rock)
for x in pop:
    for y in classical:
        a = float(x[1]) - float(y[1])
        b = float(x[4]) - float(y[4])
        c = float(x[6]) - float(y[6])
        eac = math.sqrt((a*a) + (b*b) + (c*c))
        interdistance +=eac
for x in pop:
    for y in pop:
        a = float(x[1]) - float(y[1])
        b = float(x[4]) - float(y[4])
        c = float(x[6]) - float(y[6])
        eac = math.sqrt((a*a) + (b*b) + (c*c))
        intradistance +=eac

print("inter pop classic",interdistance)
print("intra pop", intradistance)

for x in classical:
    for y in classical:
        a = float(x[1]) - float(y[1])
        b = float(x[4]) - float(y[4])
        c = float(x[6]) - float(y[6])
        eac = math.sqrt((a*a) + (b*b) + (c*c))
        interdistance +=eac
print("intra classic", intradistance)
