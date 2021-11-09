#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=5 python main_resnet50_scratch.py --epochs 160 --num_tasks 1 --batch 128 > train-${CURRENTTIME}.log
