#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 main_resnet50_scratch.py --epochs 160 --num_tasks 4 --batch 128 > train-${CURRENTTIME}.log
