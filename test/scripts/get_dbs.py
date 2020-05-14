
import sys

from pydub import AudioSegment


sound = AudioSegment.from_file(sys.argv[1])
loudness = sound.dBFS
print(loudness)
