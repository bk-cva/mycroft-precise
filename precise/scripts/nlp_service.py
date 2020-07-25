#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the dependencies `pyaudio` and `termcolor`.
To install using pip:

    pip install pyaudio
    pip install termcolor

Example usage:
    python transcribe_streaming_infinite.py
"""

# [START speech_transcribe_infinite_streaming]

import os
import time
import re
import sys
import requests
import json
import base64
from time import sleep
from enum import Enum
import threading

# import RPi.GPIO as GPIO
import pyaudio
import spotipy
import urllib.request
import vlc
import simpleaudio as sa
from google.cloud import speech
from six.moves import queue
from spotipy.oauth2 import SpotifyClientCredentials

from .text_to_speech import text_to_bytes

# Audio recording parameters
STREAMING_LIMIT = 14000  # 240000  # 4 minutes
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'

#  NLP Endpoint server

NLP_URL = 'http://118.69.144.206:5050/cva'


class DeviceGPIO(Enum):
    Aircon = 8
    LeftWindow = 10
    LeftDoor = 12
    Radio = 36
    RightWindow = 38
    RightDoor = 40


# GPIO.setwarnings(False)    # Ignore warning for now
# GPIO.setmode(GPIO.BOARD)   # Use physical pin numbering


def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            # print('reset data')
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                        * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break
            print(len(data))
            yield b''.join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    transcript = ''
    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream.result_end_time = int((result_seconds * 1000)
                                     + (result_nanos / 1000000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')

            # # Pause stream
            # print("Pausing stream")
            # stream._audio_stream.stop_stream()

            conversator_task = ConversationProccessor(transcript, stream)
            conversator_task.start()


            # data = {'topic': "request_cva",
            #         'user_id': '1',
            #         'utterance': transcript}
            # nlp_response = requests.post(NLP_URL, json=data)
            # response_json = json.loads(nlp_response.text)
            # print(response_json)

            # # Text to speech
            # audio_data = text_to_bytes(response_json['response'][1])
            # play_obj = sa.play_buffer(audio_data, 1, 2, 16000)
            # play_obj.wait_done()

            # stream.is_final_end_time = stream.result_end_time
            # stream.last_transcript_was_final = True

            # # Play music
            # conversator = ConversationProccessor(response_json)
            # has_song = conversator._play_music()
            # if not has_song:
            #     notify_str = 'Không tìm thấy bài hát trong danh sách'
            #     notify_audio = text_to_bytes(notify_str)
            #     play_obj = sa.play_buffer(notify_audio, 1, 2, 16000)
            #     play_obj.wait_done()

            print('done conversator')
            # conversator._control_car()

            # # Start stream again
            # print('Starting stream again')
            # stream._audio_stream.start_stream()

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(dừng|thoát|cám ơn|cảm ơn|ok)\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write('Exiting...\n')
                stream.closed = True
                break

        else:
            sys.stdout.write(RED)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

            stream.last_transcript_was_final = False

    return transcript


def nlp_task():
    """start bidirectional streaming from microphone input to speech API"""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './precise/scripts/Smart-Speaker-e00f3b0e6efb.json'
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='vi-VN',
        max_alternatives=1,
        use_enhanced=True)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
        single_utterance=False)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(
                STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')

            stream.audio_input = []
            audio_generator = stream.generator()

            request = (speech.types.StreamingRecognizeRequest(
                audio_content=content)for content in audio_generator)

            print('begin get response')
            responses = client.streaming_recognize(streaming_config,
                                                   request)
            print('response done')
            # Now, put the transcription responses to use.
            transcript = listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True


class ConversationProccessor():
    def __init__(self, transcript):
        # threading.Thread.__init__(self)
        self.transcript = transcript
        # self.stream = stream

    def _nlp_request(self):
        data = {'topic': "request_cva",
                'user_id': '1',
                'utterance': self.transcript}
        nlp_response = requests.post(NLP_URL, json=data)
        self.response_json = json.loads(nlp_response.text)
        print(self.response_json)
        self.state = self.response_json.get('response')[0]
        self.intent = self.response_json.get('intent')
        if self.intent == 'music' and self.response_json['action'] == 'respond_music':
            self.music_info = self.response_json['metadata'].get(
                'song_name') or self.response_json['metadata'].get('music_genre')
            print(self.music_info)
        # self.devices_car = DeviceGPIO
        # for d in self.devices_car:
        #     GPIO.setup(d.value, GPIO.OUT)

    def _play_music(self):
        if self.intent == 'music' and self.response_json['action'] == 'respond_music':
            os.environ['SPOTIPY_CLIENT_ID'] = 'e95d3002cf2e408ea7d4f5f34ea63b3c'
            os.environ['SPOTIPY_CLIENT_SECRET'] = '5c213b772ebc4c01803f20d2a225fc28'
            os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888'

            spotify = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials())
            result = spotify.search(self.music_info, limit=1)
            print(result)
            if not (result['tracks']['items'] and result['tracks']['items'][0]['preview_url']):
                self._text_to_speech('Không tìm thấy bài hát trong danh sách')
            preview_url = result['tracks']['items'][0]['preview_url']

            player = vlc.MediaPlayer(preview_url)
            player.play()
            good_states = ["State.Playing",
                           "State.NothingSpecial", "State.Opening"]
            while str(player.get_state()) in good_states:
                pass
            print('Stream is not working. Current state = {}'.format(
                player.get_state()))
            player.stop()

    def _text_to_speech(self, text):
        audio_data = text_to_bytes(text)
        play_obj = sa.play_buffer(audio_data, 1, 2, 16000)
        play_obj.wait_done()

    def _response_to_speech(self):
        audio_data = text_to_bytes(self.response_json['response'][1])
        play_obj = sa.play_buffer(audio_data, 1, 2, 16000)
        play_obj.wait_done()

    def _control_car(self):
        if self.intent == 'control_aircon':
            GPIO.output(self.devices_car['Aircon'].value,
                        self.response['metadata']['action_type'])
        elif self.intent == 'control_radio':
            GPIO.output(self.devices_car['Radio'].value,
                        self.response['metadata']['action_type'])
        elif self.intent == 'control_door':
            if self.response['metadata']['side'] == -1:
                GPIO.output([self.devices_car['LeftDoor'].value, self.devices_car['RightDoor'].value],
                            self.response['metadata']['action_type'])
            else:
                device_pin = self.devices_car['RightDoor'].value if self.response[
                    'metadata']['side'] else self.devices_car['LeftDoor'].value
                GPIO.output(device_pin,
                            self.response['metadata']['action_type'])
        elif self.intent == 'control_window':
            if self.response['metadata']['side'] == -1:
                GPIO.output([self.devices_car['LeftWindow'].value, self.devices_car['RightWindow'].value],
                            self.response['metadata']['action_type'])
            else:
                device_pin = self.devices_car['RightWindow'].value if self.response[
                    'metadata']['side'] else self.devices_car['LeftWindow'].value
                GPIO.output(device_pin,
                            self.response['metadata']['action_type'])

    def run(self):
        # self.stream._audio_stream.stop_stream()
        print('begin nlp service')
        self._nlp_request()
        self._response_to_speech()
        self._play_music()
        # self.stream._audio_stream.start_stream()

if __name__ == '__main__':
    nlp_task()
    # utte = sys.argv[1]
    # data = {'topic': "request_cva",
    #         'user_id': '1',
    #         'utterance': utte
    #         }
    # nlp_response = json.loads(requests.post(NLP_URL, json=data).text)
    # audio_data = text_to_bytes(nlp_response['response'][1])
    # play_obj = sa.play_buffer(audio_data, 1, 2, 16000)
    # play_obj.wait_done()
    # print(nlp_response)
    # conversator_task = ConversationProccessor(utte)
    # conversator_task.start()

# [END speech_transcribe_infinite_streaming]
