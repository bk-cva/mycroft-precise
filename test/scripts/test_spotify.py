import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import urllib.request
import  vlc

os.environ['SPOTIPY_CLIENT_ID'] = 'e95d3002cf2e408ea7d4f5f34ea63b3c'
os.environ['SPOTIPY_CLIENT_SECRET'] = '5c213b772ebc4c01803f20d2a225fc28'
os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888'

lz_uri = 'spotify:artist:5dfZ5uSmzR7VQK0udbAVpf'
track_uri = 'spotify:track:0f5yQttJS5nNxRAleF4kZO'

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
# print(spotify.devices())
result = spotify.search('Hãy trao cho anh', limit=1)
print(result)
if not result['tracks']['items'] or not result['tracks']['items'][0]['preview_url']:
    print("Not found preview url")
    exit()

preview_url = result['tracks']['items'][0]['preview_url']
print(preview_url)

player = vlc.MediaPlayer(preview_url)
player.play()

good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
while str(player.get_state()) in good_states:
    pass
print('Stream is not working. Current state = {}'.format(player.get_state()))
player.stop()

# for track in results['tracks'][:1]:
#     print('track    : ' + track['name'])
#     print('audio    : ' + track['preview_url'])
#     print('cover art: ' + track['album']['images'][0]['url'])
#     print()
