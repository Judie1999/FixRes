python -m torch.distributed.launch --nproc_per_node=2 main_resnet50_scratch.py --epochs 10 --num_tasks 2 --batch 80 > train.log
