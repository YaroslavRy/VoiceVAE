import numpy as np
import librosa
import os
import pickle

AUDIO_PATH = '/Users/nemo/Downloads/DS_10283_1942/noisy_trainset_wav/'
PATH_TO_SAVE = 'data/data_wavs_compiled.pkl'


def load_dataset(audio_path):
    data = []
    for i, file_name in enumerate(os.listdir(audio_path)):
        print(i)
        wav, sr = librosa.load(os.path.join(audio_path, file_name))
        data.append(wav)
    return data


data_wavs = load_dataset(AUDIO_PATH)
pickle.dump(data_wavs, open(PATH_TO_SAVE, 'wb'))
