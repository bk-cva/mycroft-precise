import numpy as np

from prettyparse import Usage

from precise_runner import PreciseEngine, PreciseRunner
from precise_runner.runner import ListenerEngine
from precise.network_runner import Listener
from precise.scripts.base_script import BaseScript
from precise.util import buffer_to_audio, load_audio
from precise.vectorization import vectorize
from precise.params import pr, inject_params, ListenerParams


model_file = './models/hey-sunshine-GRU/hey_sunshine.net'
test_file = './data/test/background_noise.wav-00.wav'
lable_file = './data/test/background_noise.wav-00.text'
CHUNK_SIZE = 2048
BUFFER_SAMPLES = 24000

audio_buffer = np.zeros(BUFFER_SAMPLES, dtype=float)

def get_prediction(chunk):
    audio = buffer_to_audio(chunk)
    self.audio_buffer = np.concatenate((audio_buffer[len(audio):], audio))
    return listener.update(chunk)



data_test = load_audio(test_file)
buffer_sample = pr.buffer_samples
hop_sample = pr.hop_samples
features = vectorize(data_test[:buffer_sample - 100])
print(features.shape)

listener = Listener.find_runner(model_file)(model_file)
output = listener.predict(features.reshape(1,-1,13))
print(output)


class EvaluateModel:
    usage = Usage('''
        Evaluate a model by the long wav file

        :model str
            Either Keras (.net) or TensorFlow (.pb) model to test
        
        :testFile str
            A long file to evaluate
        
        ...    
    ''')

    def __init__(self, model: str, test_file: str, lable_file: str, params: ListenerParams):
        self.listener = Listener.find_runner(model)(model)
        self.raw_data = load_audio(test_file)
        self.params = params
        self.buffer_sample = params.buffer_samples
        self.hop_sample = params.hop_samples
        self.window_samples = params.window_samples
        self.num_predicts = (self.raw_data.shape[0] - self.window_samples) // self.hop_sample + 1
        self.counter = 0
        print(self.num_predicts)

    def predict(self):
        step = 0
        for i in range(self.num_predicts):
            data = self.raw_data[i*self.hop_sample : i*self.hop_sample+self.buffer_sample]
            feature = vectorize(data)
            predict = listener.predict(np.expand_dims(feature, axis=0))
            prob = np.squeeze(predict)
            if prob > 1 - self.params.threshold_center and i - step == 1:
                print(f"{self.get_time(i):.2f}: {prob:.2f}")
                # check_point = i
                # if check_point == i or i - check_point < 5:
                #     print(self.get_time(i), ': ', prob)
                # elif toggle > 5:
                #     toggle = 0
                # else:
                #     toggle += 1
            elif prob < 1 - self.params.threshold_center and i - step > 10:
                step = i

    def get_time(self, counter):
        return counter * self.hop_sample / self.params.sample_rate


if __name__ == '__main__':
    evaluator = EvaluateModel(model_file, test_file, lable_file, pr)
    evaluator.predict()