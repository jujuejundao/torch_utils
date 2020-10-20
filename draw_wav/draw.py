from scipy.io import wavfile
from matplotlib import pyplot as plt
import numpy as np
import librosa, librosa.display

def audio_norm(audio):
    return np.divide(audio, np.max(np.abs(audio)))

sr = 2000

# Load the data and calculate the time of each sample
data0, sr = librosa.load('egs/0.wav', sr = sr)
data0 = audio_norm(data0)
data0 = data0[2000:8000]

data1, sr = librosa.load('egs/1.wav', sr = sr)
data1 = audio_norm(data1)
data1 = data1[2000:8000]

plt.figure()
# librosa.display.waveplot(data0, sr=sr, color='b')

librosa.display.waveplot(data1, sr=sr, color='r')

# times = np.arange(len(data))/float(samplerate)
# # print(times)

# # Make the plot
# # You can tweak the figsize (width, height) in inches
# # plt.figure(figsize=(30, 4))
# plt.plot(data)
# # plt.plot(data)
# # plt.xlim(times[0], times[-1])
# # plt.xlabel('time (s)')
# # plt.ylabel('amplitude')
# # You can set the format by changing the extension
# # like .pdf, .svg, .eps
plt.axis('off')
plt.savefig('1.png')
