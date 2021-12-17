#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

CUDA_VISIBLE_DEVICES=0 python main_resnet50_scratch.py --epochs 160 --num_tasks 1 --batch 64 > train-${CURRENTTIME}-1p.log
