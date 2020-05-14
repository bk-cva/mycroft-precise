import os
import argparse
import numpy as np
import wavio, wave

from keras.layers.core import Dense
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
from keras.layers import Conv1D, Conv2D, MaxPooling1D, \
                        MaxPooling2D, Reshape, TimeDistributed, \
                        Dropout, LeakyReLU, Flatten
from sonopy import mfcc_spec
from prettyparse import Usage

from precise.scripts.base_script import BaseScript
from precise.functions import load_keras


class SimpleFlow():
    def __init__(self, args):
        self.args = args
        if self.args.model and os.path.isfile(self.args.model):
            print('Loading from ' + self.args.model + '...')
            self.model = load_keras().models.load_model(self.args.model)
        else:
            self.model = Sequential()
            self.model.add(Reshape((74, 13, 1), input_shape=(74, 13)))
            self.model.add(Conv2D(48, kernel_size=(10, 4), strides=2, padding='valid', activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(MaxPooling2D(2, 2, 'same'))
            self.model.add(Reshape((17, 3*48)))
            self.model.add(GRU(32, activation='relu', return_sequences=True))
            self.model.add(Dropout(0.2))
            self.model.add(GRU(32,activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, activation='sigmoid'))

    def get_summary(self):
            self.model.summary()


    def get_raw_audio(self):
        try:
            wav = wavio.read(self.args.audio_file)
            data = np.squeeze(wav.data)
            return data
        except (EOFError, wave.Error) as e:
            print(e)

    def get_vectorize_audio(self):
        sample_rate = 16000
        window_t = 0.04
        hop_t = 0.02  # == stride
        n_filt=20
        n_fft=512
        n_mfcc=13
        window_sample = int(sample_rate * window_t + 0.5)
        hop_sample = int(sample_rate * hop_t + 0.5)
        data = self.get_raw_audio()
        vector = mfcc_spec(data, sample_rate,
                        (window_sample, hop_sample),
                        num_filt=n_filt, fft_size=n_fft, num_coeffs=n_mfcc)
        return vector

    def get_predict(self, vector_input):
        import tensorflow as tf
        from tensorflow.python.keras.backend import set_session		# ISSUE 88
        sess = tf.Session()
        set_session(sess)
        graph = tf.get_default_graph()
        with graph.as_default():
            set_session(sess)		# ISSUE 88
            return self.model.predict(vector_input)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get information of model')
    parser.add_argument(
        '--model',
        type=str,
        default='./model/hey-sunshine-CRNN/hey_sunshine.net',
        help='Pretrained model'
    )
    parser.add_argument(
        '--audio_file',
        type=str,
        default='./data/test-data/test_audio.wav',
        help='File audio to get information'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        default=False,
        help='get model summary'
    )
    parser.add_argument(
        '--raw_audio',
        action='store_true',
        default=False,
        help='get raw audio'
    )
    parser.add_argument(
        '--vector_audio',
        action='store_true',
        default=False,
        help='get vectorized audio'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        default=False,
        help='get probabitiy of predict'
    )


    args = parser.parse_args()
    info = SimpleFlow(args)

    if args.summary:
        info.get_summary()

    if args.raw_audio:
        raw_data = info.get_raw_audio()
        print("Raw data: \n", raw_data)

    if args.vector_audio:
        vector = info.get_vectorize_audio()
        print("MFCC Vectorize: \n", vector)
        print(vector.shape)

    if args.predict:
        vector = info.get_vectorize_audio()
        vector_input = np.expand_dims(vector, axis=0)
        prob = info.get_predict(vector_input)
        print("Predict probability: ', prob)
