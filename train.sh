#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

python -m torch.distributed.launch --nproc_per_node=8 main_resnet50_scratch.py --epochs 120 --num_tasks 8 --batch 64 --learning_rate 0.016 > train-${CURRENTTIME}-512-step30-SGD-0.016.log
