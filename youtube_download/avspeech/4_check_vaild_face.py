import os, glob
from tqdm import trange
import csv
import pandas as pd
import argparse

def check_face_valid(index, part, check_pth):
    path = check_pth + '/%d/face_%d_%02d.jpg' % (index,index, part)
    if (not os.path.exists(path)):
        return False
    else:
        return True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
    parser.add_argument('--start', default = 200, type=int)
    parser.add_argument('--stop', default = 1000, type=int)
    args = parser.parse_args()

    test_path = '/home/zexu/workspace/av_speechseparation/data/avspeech/avspeech_train.csv'
    test_csv = pd.read_csv(test_path, header=None)

    f=open("Testset.csv",'a')
    w=csv.writer(f)

    check_pth = '/data07/zexu/datasets/avspeech/Train/face'

    indexs = []
    for i in trange(args.start, args.stop):
        valid = True
        # print('Processing video %s' % i)
        for j in range(1, 76):
            if (check_face_valid(i, j, check_pth) == False):
                # path = check_pth + '/frame_%d_*.jpg' % i
                # for file in glob.glob(path):
                #     pass
                #     os.remove(file)
                valid = False
                # print('Video %s is not valid' % i)
                break

        if valid == True:
            w.writerow([i, test_csv.loc[i][0], test_csv.loc[i][1]])

