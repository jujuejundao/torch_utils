#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6 python -m torch.distributed.launch --nproc_per_node=2 trainer.py