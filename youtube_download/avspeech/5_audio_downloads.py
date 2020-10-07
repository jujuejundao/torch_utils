import sys
import pandas as pd
import os
import time
import argparse

def download(loc,name,link,sr=16000,type='audio'): #download audio
    if type == 'audio':
        command = 'cd %s;' % loc
        command += 'youtube-dl -x --audio-format wav -o o' + name + '.wav ' + link + ';'
        command += 'ffmpeg -i o%s.wav -ar %d -ac 1 f%s.wav;' % (name,sr,name)
        command += 'rm o%s.wav' % name
        os.system(command)


def cut(loc,name,start_time,length):
    command = 'cd %s;' % loc
    command += 'sox f%s.wav %s.wav trim %s %s;' % (name,name,start_time,length)
    command += 'rm f%s.wav' % name
    os.system(command)


def make_audio(location, d_csv, start_idx, end_idx):
    for i in range(start_idx,end_idx):
        f_name = str(d_csv.loc[i][0])
        link = "https://www.youtube.com/watch?v="+d_csv.loc[i][1]
        start_time = d_csv.loc[i][2]
        start_time = time.strftime("%H:%M:%S.0",time.gmtime(start_time))
        download(location,f_name,link)
        cut(location,f_name,start_time,3.0)
        print("\r Process audio... ".format(i) + str(i), end="")
    print("\r Finish !!", end="")

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('--start', default = 0, type=int)
parser.add_argument('--stop', default = 100, type=int)
args = parser.parse_args()

cat_train = pd.read_csv('Testset.csv',header=None)

location = '/data07/zexu/datasets/avspeech/Train/audio/'

make_audio(location,cat_train,args.start,args.stop)

# print(cat_train.shape)



