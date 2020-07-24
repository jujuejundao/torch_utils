#!/bin/bash

CUDA_VISIBLE_DEVICES=11,13 python -m torch.distributed.launch --nproc_per_node=2 trainer.py