#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py --input-size 384 --architecture 'ResNet50' --epochs 60 --num-tasks 4 --batch 128 --learning-rate 1e-3 --resnet-weight-path '/data2/herunyu/fixres_cache/20211008-110514/checkpoint.pth' > finetune-${CURRENTTIME}.log
