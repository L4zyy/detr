#!/bin/bash
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
python -m torch.distributed.launch --use_env main.py --no_aux_loss --ctransformer --use_subset \
    --lr 1e-5 \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/cdetr