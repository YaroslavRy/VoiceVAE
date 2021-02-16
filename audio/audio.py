import numpy as np
import librosa
from config import AUDIO_PATH
import os
import sounddevice as sd
import matplotlib.pyplot as plt
import librosa
import librosa.display


def load_wav(filename, sr):
    wav, sr = librosa.load(filename, sr=sr)
    return wav, sr


def play_wav_file(wav, fs):
    sd.play(wav, fs)
    status = sd.wait()


def plot_spectrogram(wav, sample_rate):
    D = librosa.stft(wav, n_fft=512)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title='Now with labeled axes!')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


def plot_spectrogram_mel(wav, sample_rate):
    fig, ax = plt.subplots()
    M = librosa.feature.melspectrogram(y=wav, sr=sample_rate)
    M = librosa.power_to_db(M, ref=np.max)
    img = librosa.display.specshow(M, y_axis='mel', x_axis='time', ax=ax)
    ax.set(title='Mel spectrogram display')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()


# print(os.listdir(AUDIO_PATH))
#
# wav, sr = load_wav(AUDIO_PATH + 'p231_014.wav', sr=16000)
#
# play_wav_file(wav, sr)
