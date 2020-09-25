#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --eval --use_subset\
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/test/detr/

#    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
