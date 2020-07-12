#!/bin/bash
#
#SBATCH --job-name=comp_detr
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=00-12:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:4

# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss --eval \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr/r101-dc5

python -m torch.distributed.launch --use_env main.py --no_aux_loss --eval \
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr/coco/r101-dc5
