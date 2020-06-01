#!/bin/bash
CUDA_VISIBLE_DEVICES=13,14,15 python -m torch.distributed.launch --nproc_per_node=2 egs.py