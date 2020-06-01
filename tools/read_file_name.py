#This file read file names in the directory and save into csv file
import numpy as np
import os
import csv

path = '/home/panzexu/datasets/avspeech/Test/video'
names = []
f=open("Test_names.csv",'w+')
w=csv.writer(f)
for path, dirs, files in os.walk(path):
	for filename in files:
		names.append(int(os.path.splitext(filename)[0]))

# names.sort()
names = np.array(names)
names = np.sort(names)
print(names.shape[0])

for i in range(names.shape[0]):
	w.writerow([names[i]])

print("Process finished")

