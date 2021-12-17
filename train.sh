#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 main_resnet50_scratch.py --epochs 160 --num_tasks 4 --batch 128 --learning_rate 0.025 > train-${CURRENTTIME}.log
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=8 main_resnet50_scratch.py --epochs 160 --num_tasks 8 --batch 64 --learning_rate 0.02 > train-${CURRENTTIME}-512-step30-SGD-0.02.log
