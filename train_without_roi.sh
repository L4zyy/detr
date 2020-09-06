#!/bin/bash
#
#SBATCH --job-name=train_roi
#SBATCH --output=logs/res_%j.txt  # output file
#SBATCH -e logs/res_%j.err        # File to which STDERR will be written
#SBATCH --partition=m40-long # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --time=07-00:00       # Runtime in D-HH:MM
#SBATCH --gres=gpu:4
#SBATCH --partition=m40-long
#SBATCH --mem 60G

python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --epoch 100\
    --backbone resnet101 \
    --batch_size 1 --dilation \
    --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
    --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
    --output_dir /mnt/nfs/scratch1/zhiyilai/detr_roi/

# python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --no_aux_loss --eval \
#     --backbone resnet101 \
#     --batch_size 1 --dilation \
#     --resume https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth \
#     --coco_path /mnt/nfs/scratch1/zhiyilai/coco \
#     --output_dir /mnt/nfs/scratch1/zhiyilai/comp/detr/r101-dc5
