#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py --input-size 384 --architecture 'ResNet50' --epochs 60 --num-tasks 8 --batch 64 --learning-rate 1e-3 --resnet-weight-path '/data2/herunyu/fixres_cache/20211116-094236/checkpoint_119.pth' > finetune-${CURRENTTIME}-mytrain.log
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py --input-size 384 --architecture 'ResNet50' --epochs 56 --num-tasks 8 --batch 64 --learning-rate 1e-3 --resnet-weight-path '/data2/herunyu/fixres_cache/ResNet_no_adaptation.pth' > finetune-${CURRENTTIME}-repo.log
