#!/bin/bash

CURRENTTIME=`date +"%Y%m%d-%H%M%S"`

python main_resnet50_scratch.py --epochs 1 --num_tasks 1 --batch 64 > perf-${CURRENTTIME}-1p.log
