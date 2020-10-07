from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import datetime

import utils
import pandas as pd
import time

path = "/home/panzexu/datasets/avspeech/Train/"

def download_video_frames(loc,d_csv,start_idx,end_idx,rm_video):
    # Download each video and convert to frames immediately, can choose to remove video file
    # loc        | the location for downloaded file
    # cat        | the catalog with audio link and time
    # start_idx  | the starting index of the video to download
    # end_idx    | the ending index of the video to download
    # rm_video   | boolean value for delete video and only keep the frames

    for i in range(start_idx, end_idx):
        command = 'cd %s;' % path
        command += 'cd %s;' % loc
        f_name = str(i)
        link = "https://www.youtube.com/watch?v="+d_csv.loc[i][0]
        start_time = d_csv.loc[i][1]
        
        start_time = time.strftime("%H:%M:%S.0",time.gmtime(start_time))
        command += 'youtube-dl --prefer-ffmpeg -f "mp4" -o o' + f_name + '.mp4 ' + link + ';'
        command += 'ffmpeg -i o'+f_name+'.mp4'+' -c:v h264 -c:a copy -ss '+str(start_time)+' -t '+"3 "+f_name+'.mp4;'
        command += 'rm o%s.mp4;' % f_name

        #converts to frames
        #command += 'ffmpeg -i %s.mp4 -y -f image2  -vframes 75 ../frames/%s-%%02d.jpg;' % (f_name, f_name)
        command += 'ffmpeg -i %s.mp4 -vf fps=25 ../frames/%s-%%02d.jpg;' % (f_name, f_name)
        #command += 'ffmpeg -i %s.mp4 ../frames/%sfr_%%02d.jpg;' % ('clip_' + f_name, f_name)

        if rm_video:
            command += 'rm %s.mp4;' % f_name
        os.system(command)
        print("\r Process video... ".format(i) + str(i), end="")
    print("\r Finish !!", end="")


cat_data = pd.read_csv('~/datasets/avspeech/avspeech_train.csv',header=None)

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('--start', default = 200, type=int)
    parser.add_argument('--stop', default = 1000, type=int)
    args = parser.parse_args()

# download each video and convert to frames immediately
download_video_frames(loc='video',d_csv=cat_data,start_idx=args.start,end_idx=args.stop,rm_video=False)