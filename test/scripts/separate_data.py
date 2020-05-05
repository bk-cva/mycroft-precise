import os
import shutil
import glob

from sklearn.model_selection import train_test_split

list_data_folder = ['wwRandDb-randPos-noise_low_SNR', \
                    'wwRandDb-center-noise_high_SNR', \
                    'wwRandDb-randPos', \
                    'wwRandDb-center']

list_data_path = [os.path.join('./data/enhenced', f) for f in list_data_folder]

train_path = './data/hey-sunshine-enhenced/wake-word'
test_path = './data/hey-sunshine-enhenced/test/wake-word'

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


clean_folder(train_path)
clean_folder(test_path)

def separate_data(list_data_path, test_rate=0.2):
    for folder in list_data_path:
        list_file = glob.glob(os.path.join(folder, '*.wav'))
        train, test = train_test_split(list_file, test_size=test_rate)
        for t in train:
            shutil.copy(t, train_path)
        for t in test:
            shutil.copy(t, test_path)

separate_data(list_data_path)
