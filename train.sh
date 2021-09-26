#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

python -m torch.distributed.launch --nproc_per_node=1 main_resnet50_scratch.py --epochs 1 --num_tasks 1 --batch 16 > train-${CURRENTTIME}.log
