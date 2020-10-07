import os
import numpy as np 
from tqdm import tqdm

# scp faces
# path_visual = '/home/panzexu/datasets/avspeech/Test/faces/'
# path_to = '/home/panzexu/datasets/AVSpeech/Test/orig/faces/'
# path_audio  = '/home/panzexu/datasets/AVSpeech/Test/audio_temp/'


# names = []
# for path, dirs, files in os.walk(path_audio):
# 	for filename in files:
# 		names.append(int(os.path.splitext(filename)[0]))

# # names.sort()
# names = np.array(names)
# names = np.sort(names)
# names = list(names)
# # print(names)
# # print(len(names))

# print("Process finished")

# for i in tqdm(names):
# 	command = "";
# 	for j in range(1,10):
# 		command += 'scp ' + path_visual + 'frame_%s_0%s.jpg'%(i, j) +' ' + path_to + ';'
# 		os.system(command)
# 	for j in range(10,76):
# 		command += 'scp ' + path_visual + 'frame_%s_%s.jpg'%(i, j) +' ' + path_to+ ';'
# 	os.system(command)


# scp video
# path_audio  = '/home/panzexu/datasets/AVSpeech/Test/audio_temp/'
# path_video = '/home/panzexu/datasets/avspeech/Test/abandon/video/'
# path_to = '/home/panzexu/datasets/AVSpeech/Test/orig/video_temp/'


# names = []
# for path, dirs, files in os.walk(path_audio):
# 	for filename in files:
# 		names.append(int(os.path.splitext(filename)[0]))

# # names.sort()
# names = np.array(names)
# names = np.sort(names)
# names = list(names)
# # print(names)
# # print(len(names))

# # print("Process finished")

# for i in tqdm(names):
# 	command = 'scp ' + path_video + '%s.mp4'%(i) +' ' + path_to+ ';'
# 	os.system(command)

# scp faces
path_visual = '/home/panzexu/datasets/AVSpeech/Test/faces/'
path_to = '/home/panzexu/datasets/AVSpeech/Test/faces2/'
path_audio  = '/home/panzexu/datasets/AVSpeech/Test/audio_mix_two/'

# path_audio  = '/home/panzexu/workspace/av_speechseparation/data/datasets/audio_mix_two/'
# path_visual = '/home/panzexu/workspace/av_speechseparation/data/datasets/faces/'
# path_to = '/home/panzexu/workspace/av_speechseparation/data/datasets/faces1/'

names = []
for path, dirs, files in os.walk(path_audio):
	for filename in files:
		name = os.path.splitext(filename)[0]
		idx = name.split("_")
		names.append(int(idx[2]))

# names.sort()
# names = np.array(names)
# names = np.sort(names)
# names = list(names)
print(names)
print(len(names))

print("Process finished")

for i in tqdm(names):
	command = "";
	for j in range(1,76):
		command += 'mv ' + path_visual + 'frame_%s_%02d.jpg'%(i, j) +' ' + path_to + ';'
	os.system(command)