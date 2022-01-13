#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

python main_finetune.py --epochs 1 --num_tasks 1 --batch 64 --resnet-weight-path '/home/hry/FixRes_gpu/train_cache/20211218-113441/checkpoint.pth' > perf-finetune-1p.log