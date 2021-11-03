#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_finetune.py --input-size 384 --architecture 'ResNet50' --epochs 60 --num-tasks 1 --batch 256 --learning-rate 1e-3 --resnet-weight-path '/data2/herunyu/fixres_cache/20211022-155016/checkpoint_119.pth' > finetune-${CURRENTTIME}.log
