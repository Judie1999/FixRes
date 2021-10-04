#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 main_resnet50_scratch.py --epochs 120 --num_tasks 2 --batch 128 > train-${CURRENTTIME}.log
