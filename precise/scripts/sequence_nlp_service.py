import os
import sys
import numpy as np
import pyaudio
import io
import re
from queue import Queue

import vlc
import simpleaudio as sa
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from google.cloud import speech_v1
from google.cloud.speech_v1 import enums

from .nlp_service import ConversationProccessor


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
RECORD_SECONDS = 3
CHUNK_COUNT = RECORD_SECONDS*RATE/CHUNK
SILENT_MESSAGE = 'Silent audio'
TIMEOUT = 9
MAX_SILENCES = TIMEOUT / RECORD_SECONDS

queue = Queue()


def audio_callback(in_data, frame_count, time_info, status):
    # data0 = np.frombuffer(in_data, dtype=np.int16) / np.iinfo(np.int16).max
    queue.put(in_data)
    return (in_data, pyaudio.paContinue)


def sample_recognize(content):
    """
    Transcribe a short audio file using synchronous speech recognition
    Args:
      local_file_path Path to local audio file, e.g. /path/audio.wav
    """

    client = speech_v1.SpeechClient()
    # local_file_path = 'resources/brooklyn_bridge.raw'
    # The language of the supplied audio
    language_code = "vi-VN"
    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000
    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
        'use_enhanced': True
    }
    audio = {"content": content}
    response = client.recognize(config, audio)
    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        return alternative.transcript

    return SILENT_MESSAGE


def nlp_task(stream):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './precise/scripts/Smart-Speaker-e00f3b0e6efb.json'
    # pa = pyaudio.PyAudio()
    # stream = pa.open(format=pyaudio.paInt16,
    #                  channels=1,
    #                  rate=RATE,
    #                  input=True,
    #                  output=False,
    #                  frames_per_buffer=CHUNK)

    silent_count = 0
    while True:
        stream.start_stream()
        print("* recording")
        data = []
        for _ in range(int(CHUNK_COUNT)):
            data.append(stream.read(CHUNK))
        stream.stop_stream()
        samples = b''.join(data)
        print('done get data')

        # Speech to text
        transcript = sample_recognize(samples)
        if re.search(r'\b(dừng|thoát|cám ơn|cảm ơn|ok)\b', transcript, re.I):
            print('Exiting ...')
            break
        elif transcript == SILENT_MESSAGE:
            if silent_count > MAX_SILENCES:
                break
            silent_count += 1
            continue
        else:
            conversator_task = ConversationProccessor(transcript)
            conversator_task.run()

    stream.stop_stream()
    stream.close()
    pa.terminate()


if __name__ == "__main__":
    nlp_task()
