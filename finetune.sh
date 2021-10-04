#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py --input-size 384 --architecture 'ResNet50' --epochs 60 --num-tasks 2 --batch 128 --learning-rate 1e-3 --resnet-weight-path '/data2/herunyu/fixres_cache/20210928-085330/checkpoint.pth' > finetune-${CURRENTTIME}.log
