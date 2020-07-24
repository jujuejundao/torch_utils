import os
import numpy as np
import tqdm
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

# path_data = '/data07/zexu/datasets/Orig/LRS2/'
# folders = ['Pretrain/', 'Test/', 'Validation/', 'Train/']
# test = ['Test/', 'Validation/']

# # names = []
# lengths = []
# for folder in folders:
# 	path_audio = path_data + folder +'audio/'
# 	print(path_audio)
# 	for path, dirs, files in os.walk(path_audio):
# 		for filename in tqdm.tqdm(files):
# 			# names.append(filename)
# 			sr,audio = wavfile.read(path_audio+filename)
# 			length = audio.shape[0]/sr
# 			if length < 30:
# 				lengths.append(length)

# plt.hist(lengths,  bins=200)
# plt.ylabel('No. of audios')
# plt.xlabel('Length (s)')
# plt.title("Histogram of LRS2 audios")
# # plt.show()
# plt.savefig('foo.pdf')

# print(len(lengths))

# np.save("lengths.npy", lengths)
# print(len(names))

lengths = np.load("lengths.npy")
plt.hist(lengths,  bins=100)
plt.ylabel('No. of audios')
plt.xlabel('Length (s)')
plt.title("Histogram of LRS2 audios")
# plt.show()
plt.savefig('lrs2.png')
