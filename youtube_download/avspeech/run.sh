#!/bin/bash 
#SBATCH -p cpu1
#SBATCH --output=log.txt
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=ttnusa7

# python 2_video_corp_frames.py --start 0 --stop 2622000 > log_frame.txt 2>&1

CUDA_VISIBLE_DEVICES=-1 KERAS_BACKEND=tensorflow python -W ignore 3_face_detect.py --start 2100000 --stop 2110000 > log_face1.txt 2>&1

# 1-700000



#train: 2,621,845 rows x 5 columns
#test: 183,273 rows x 5 columns