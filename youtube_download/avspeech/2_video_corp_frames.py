from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import tqdm
import argparse

if __name__ == "__main__":
    path = '/data07/zexu/datasets/avspeech/Train/'
    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('--start', type=int)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()

    video_path = path + 'video/'
    frame_pth = path + 'frame/'

    for i in tqdm.trange(args.start, args.stop):
        f_name = str(i)
        if (not os.path.exists('%s%s.mp4' % (video_path, f_name))):
            # print('cannot find input: ' + '%s.mp4' % (f_name))
            continue

        if not os.path.exists('%s%s' % (frame_pth, i)):
            os.makedirs(frame_pth + '%s' % (i))
            for j in range(1, 76):
                filename = '%d-%02d.jpg' % (i, j)
                if (not os.path.exists('%s/%s/%s' % (frame_pth, i, filename))):
                    command = 'ffmpeg -i %s%s.mp4 -vf fps=25 %s/%s/%s-%%02d.jpg;' % (video_path, f_name, frame_pth, i, f_name)
                    os.system(command)
                    break
