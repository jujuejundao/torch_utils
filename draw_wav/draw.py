from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import librosa, librosa.display

def audio_norm(audio):
    return np.divide(audio, np.max(np.abs(audio)))

sr = 2000

# Load the data and calculate the time of each sample
data0, sr = librosa.load('demo/0.wav', sr = sr)
data0 = audio_norm(data0)
data0 = data0[2000:8000]

data1, sr = librosa.load('demo/1.wav', sr = sr)
data1 = audio_norm(data1)
data1 = data1[2000:8000]

plt.figure()
# librosa.display.waveplot(data0, sr=sr, color='gray')

librosa.display.waveplot(data1, sr=sr, color='g')

plt.axis('off')
plt.savefig('/home/panzexu/Download/wav.png')



# tmm drawing
# data0, sr = librosa.load('demo/0.wav', sr = sr)
# data0 = audio_norm(data0)
# data0 = data0[2000:8000]

# data1, sr = librosa.load('demo/1.wav', sr = sr)
# data1 = audio_norm(data1)
# noise2 = data1[5000:8000]
# data1 = data1[:12000]

# data1[:]=0

# data1[3000:9000]=data0

# noise = np.random.rand(12000)

# data1 = data1+noise*0.02
# data1[9000:] +=noise2*0.2