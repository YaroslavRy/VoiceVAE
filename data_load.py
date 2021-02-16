import numpy as np
import librosa
import os
import pickle
from config import AUDIO_PATH, PATH_TO_SAVE, SAMPLE_RATE
from tqdm import tqdm


def load_dataset(audio_path):
    data = []
    for file_name in tqdm(os.listdir(audio_path)):
        wav, sr = librosa.load(os.path.join(audio_path, file_name), sr=SAMPLE_RATE)
        data.append(wav)
    return data


data_wavs = load_dataset(AUDIO_PATH)
pickle.dump(data_wavs, open(PATH_TO_SAVE, 'wb'))
