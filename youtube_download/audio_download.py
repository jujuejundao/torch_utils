import pandas as pd
import os
import time
import tqdm

def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")

def download(loc,name,link,sr=16000,type='audio'): #download audio
    if type == 'audio':
        command = 'cd %s;' % loc
        command += 'youtube-dl -x --audio-format wav -o o' + name + '.wav ' + link + ';'
        command += 'ffmpeg -i o%s.wav -ar %d -ac 1 oo%s.wav;' % (name,sr,name)
        command += 'rm o%s.wav' % name
        os.system(command)


def cut(loc,name,start_time,end_time):
    length = end_time - start_time
    print(length)
    # start_time = time.strftime("%H:%M:%S.0",time.gmtime(start_time))
    command = 'cd %s;' % loc

    command += 'sox oo%s.wav trim_%s.wav trim %s %s;' % (name,name,start_time,length)
    command += 'rm oo%s.wav;' % name
    # command += 'mv trim_%s.wav %s.wav;' %(name, name)
    os.system(command)


def make_audio(location, d_csv, start_idx, end_idx):
    for i in tqdm.trange(start_idx,end_idx):
        line = d_csv[i].split(',')
        f_name = line[0]
        link = "https://www.youtube.com/watch?v="+f_name
        start_time = float(line[1])
        end_time = float(line[2])
        exi=location+'/trim_'+f_name+'.wav'
        if os.path.exists(exi):
            continue
        download(location,f_name,link)
        cut(location,f_name,start_time,end_time)
        print("\r Process audio... ".format(i) + str(i), end="")
    print("\r Finish !!", end="")


if __name__ == '__main__':
    partition='balanced_train_segments'
    cat_train = open(partition+'.csv').read().splitlines()
    # print(len(cat_train[3:]))
    mkdir(partition)
    make_audio(partition,cat_train[3:],10000,22160)





