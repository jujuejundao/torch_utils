import numpy as np
import math
import scipy.io.wavfile as wavfile
import tqdm
import random

random.seed(0)

def audio_norm(audio):
    return np.divide(audio, np.max(np.abs(audio)))

audio_direc = '/home/panzexu/datasets/voxceleb2/audio_sync/huge/'

mix_lst=open('/home/panzexu/datasets/voxceleb2/audio_sync/huge/sync_list.csv').read().splitlines()

# random.shuffle(mix_lst)

print(len(mix_lst))

#################################
### filter list out
#################################

mix_lst = [ x for x in mix_lst if "id08137,Pvrmbe76RkU/00196" not in x ]
print(len(mix_lst))

#################################
### read single audio all
#################################

# for line in tqdm.tqdm(mix_lst):
# 	audio_path=audio_direc+line.split(',')[0]+'/'+line.split(',')[1]+'/'+ line.replace(',','_').replace('/','_')+'.wav'
# 	_, audio = wavfile.read(audio_path)
# 	if np.all((audio == 0)):
# 		print('1')
# 		print(line)
# 	if np.max(np.abs(audio)) ==0:
# 		print('2')
# 		print(line)
# 		print(np.max(np.abs(audio)))
# 	audio = np.divide(audio, np.max(np.abs(audio)))


#################################
### read batch audio all
#################################

# sorted_mix_lst = sorted(mix_lst, key=lambda data: (int(data.split(',')[6]) - int(data.split(',')[5]) ) , reverse=True)
# start = 0
# batch_size = 32
# minibatch = []
# while True:
#     end = min(len(sorted_mix_lst), start + batch_size)
#     minibatch.append(sorted_mix_lst[start:end])
#     if end == len(sorted_mix_lst):
#         break
#     start = end
# for batch_lst in minibatch:
# 	min_length = int(batch_lst[-1].split(',')[6]) - int(batch_lst[-1].split(',')[5])
# 	for line in batch_lst:
# 		# read audio
# 		audio_path=audio_direc+line.split(',')[0]+'/'+line.split(',')[1]+'/'+ line.replace(',','_').replace('/','_')+'.wav'
# 		_, audio = wavfile.read(audio_path)
# 		if np.all((audio == 0)):
# 			print('1')
# 			print(line)
# 		if np.max(np.abs(audio)) ==0:
# 			print('2')
# 			print(line)
# 			print(np.max(np.abs(audio)))
# 		audio = np.divide(audio, np.max(np.abs(audio)))



#################################
### read single audio
#################################

# line = 'train,sync_m,train,id01099,0E7TxYeHunU/00009,5760,49280,0,train,id00694,AzggAlJacpI/00062,12463,55983,9.42651817873218'
# audio_path='/home/panzexu/datasets/voxceleb2/audio_sync/huge/'+line.split(',')[0]+'/'+line.split(',')[1]+'/'+ line.replace(',','_').replace('/','_')+'.wav'
# _, audio = wavfile.read(audio_path)
# print(audio)
# audio = audio_norm(audio)

