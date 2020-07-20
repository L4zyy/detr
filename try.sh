#!/bin/bash
# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss --eval \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr/r101-dc5

# python -m torch.distributed.launch --use_env main.py --no_aux_loss --eval \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \

python -m torch.distributed.launch --use_env main.py --no_aux_loss \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detr_roi
