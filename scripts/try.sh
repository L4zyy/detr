#!/bin/bash

python -m torch.distributed.launch --use_env main.py --no_aux_loss --ltransformer \
    --num_groups 1 \
    --lr 1e-5 \
    --backbone resnet50 \
    --batch_size 2 \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/test

# python -m torch.distributed.launch --use_env main.py --no_aux_loss --eval \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \

# python -m torch.distributed.launch --use_env main.py --no_aux_loss --roi_head --use_subset \
#     --lr 1e-2 \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/detr_roi

# python -m torch.distributed.launch --use_env main.py --no_aux_loss --use_subset \
# python -m torch.distributed.launch --use_env main.py --no_aux_loss --ctransformer --merge --use_subset \
#     --lr 1e-5 \
#     --backbone resnet50 \
#     --batch_size 2 \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/detrs/test