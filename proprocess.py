import numpy as np
import librosa

# pipeline

# 1st option
# wav -> spectrogram -> NeuralNet -> spectrogram -> Loss -> wav

# 2nd option
# wav -> mel -> NeuralNet -> mel -> Loss -> wav

# 2nd option
# wav -> spectrogram -> power_to_db -> Neural Net -> spectrogram -> S_db -> Loss ->


