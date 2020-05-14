import numpy as np
import os

import wave
from prettyparse import Usage

from precise_runner import PreciseEngine, PreciseRunner
from precise_runner.runner import ListenerEngine
from precise.network_runner import Listener
from precise.scripts.base_script import BaseScript
from precise.util import buffer_to_audio, load_audio, save_audio
from precise.vectorization import vectorize
from precise.params import pr, inject_params, ListenerParams


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class EvaluateModel:
    usage = Usage('''
        Evaluate a model by the long wav file

        :model str
            Either Keras (.net) or TensorFlow (.pb) model to test

        :testFile str
            A long file to evaluate

        ...
    ''')

    def __init__(self, model: str, data_path: str, test_file: str, lable_file: str, params: ListenerParams):
        self.listener = Listener.find_runner(model)(model)
        self.raw_data = load_audio(test_file)
        self.params = params
        self.buffer_sample = params.buffer_samples
        self.hop_sample = params.hop_samples
        self.window_samples = params.window_samples
        self.num_predicts = (self.raw_data.shape[0] - self.window_samples) // self.hop_sample + 1
        self.lenght_audio = self.raw_data.shape[0] / 16000 / 60
        self.counter = 0
        self.label_file = label_file
        self.data_path = data_path

    def predict(self, output_file: str):
        output = []
        for i in range(self.num_predicts):
            data = self.raw_data[i*self.hop_sample : i*self.hop_sample+self.buffer_sample]
            feature = vectorize(data)
            predict = self.listener.predict(np.expand_dims(feature, axis=0))
            prob = np.squeeze(predict)
            if prob > 1 - self.params.threshold_center:
                output.append(self.get_time(i))
        np.savetxt(output_file, np.array(output), fmt='%6.4f')

    def process_output(self, output_file: str, processed_output_file: str, offset: float):
        output = np.loadtxt(output_file)
        processed_output = [output[0]]
        for i in range(1, output.shape[0]):
            if output[i] - output[i-1] > offset:
                processed_output.append(output[i])
        self.process_output_file = processed_output_file
        np.savetxt(processed_output_file, np.array(processed_output), fmt='%6.4f')

    def visuallize(self, predict_file: str, label_file: str, offset: float):
        predict = np.loadtxt(predict_file)
        label = np.loadtxt(label_file)
        TP, FA, FR = self._TP_FA_FR_cases(predict, label, offset)


    def save_predict_cases(self, lenght_audio: float, offset: float):
        predict = np.loadtxt(self.process_output_file)
        label = np.loadtxt(self.label_file)
        TP, FA, FR = self._TP_FA_FR_cases(predict, label, offset)
        prepend = lenght_audio - self.params.buffer_t
        clean_folder(os.path.join(self.data_path, 'true-positive'))
        clean_folder(os.path.join(self.data_path, 'false-alarm'))
        clean_folder(os.path.join(self.data_path, 'false-reject'))
        for i in range(TP.shape[0]):
            t = TP[i]
            index = int((t-prepend)*self.params.sample_rate)
            data = self.raw_data[index : index + self.params.buffer_samples]
            file_path = os.path.join(self.data_path, 'true-positive', f"TP_{t//60:.0f}_{t%60:.0f}.wav")
            save_audio(file_path, data)

        for i in range(FA.shape[0]):
            t = FA[i]
            index = int((t-prepend)*self.params.sample_rate)
            data = self.raw_data[index : index + self.params.buffer_samples]
            file_path = os.path.join(self.data_path, 'false-alarm', f"FA_{t//60:.0f}_{t%60:.0f}.wav")
            save_audio(file_path, data)

        for i in range(FR.shape[0]):
            t = FR[i]
            index = int((t-prepend)*self.params.sample_rate)
            data = self.raw_data[index : index + self.params.buffer_samples]
            file_path = os.path.join(self.data_path, 'false-reject', f"FR_{t//60:.0f}_{t%60:.0f}.wav")
            save_audio(file_path, data)

    def _TP_FA_FR_cases(self, predict: np.array, label: np.array, offset: float):
        TP = []                # True Positive
        FA = predict.copy()    # False Alarm
        FR = label.copy()      # False Reject
        for p in predict:
            for l in label:
                if abs(p - l) < offset:
                    TP.append(p)
                    FA = np.delete(FA, np.argwhere(FA == p))
                    FR = np.delete(FR, np.argwhere(FR == l))
                    continue
        return np.array(TP), FA, FR

    def get_time(self, counter):
        return counter * self.hop_sample / self.params.sample_rate


if __name__ == '__main__':
    data_path = './data/test/test-case-00'
    model_file = './models/hey-sunshine-CRNN/hey_sunshine.net'
    test_file = os.path.join(data_path, 'test_case-00.wav')
    label_file = os.path.join(data_path, 'test_case-00.txt')
    output_file = os.path.join(data_path, 'output.npy')
    processed_output_file = os.path.join(data_path, 'processed_output.npy')

    tp_folder = os.path.join(data_path, 'true-positive')
    fa_folder = os.path.join(data_path, 'false-alarm')
    fr_folder = os.path.join(data_path, 'false-reject')
    os.makedirs(tp_folder, exist_ok=True)
    os.makedirs(fa_folder, exist_ok=True)
    os.makedirs(fr_folder, exist_ok=True)

    evaluator = EvaluateModel(model_file, data_path, test_file, label_file, pr)
    evaluator.predict(output_file)
    evaluator.process_output(output_file, processed_output_file, 2)
    evaluator.save_predict_cases(2, 2)
    output = np.loadtxt(processed_output_file)
    label = np.loadtxt(label_file)
    print('predict cases: ', output.shape)
    print('label cases: ', label.shape)

    TP, FA, FR = evaluator._TP_FA_FR_cases(output, label, 1.5)
    # print(np.array([f"{t//60:.0f}:{t%60:.0f}" for t in FA]))
    print('False reject rate: ', (FR.shape[0] / (FR.shape[0] + TP.shape[0])))
    print('False alarm rate: ', (FA.shape[0] / (evaluator.lenght_audio/60)))
